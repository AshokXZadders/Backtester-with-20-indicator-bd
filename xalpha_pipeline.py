

import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING  (mirrors wto.py's load_data for column-name consistency)
# ============================================================================

def load_price_data(csv_file, use_volume=False):
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    df.columns = [c.strip().lower() for c in df.columns]
    needed = ['open', 'high', 'low', 'close']
    out_cols = ['Open', 'High', 'Low', 'Close']
    if use_volume:
        needed = needed + ['volume']
        out_cols = out_cols + ['Volume']
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Found: {list(df.columns)}"
                              + ("  (pass --use-volume only if your CSV actually "
                                 "has a Volume column)" if col == 'volume' else ""))
    df = df[needed].copy()
    df.columns = out_cols
    df = df.dropna().astype(float)
    df = df[(df > 0).all(axis=1)]
    return df


def compute_hmm_features(df, use_volume=False):
    """
    Emissions for the HMM: log returns, plus a volume z-score when
    --use-volume is set (matching the doc's 'rolling price variance and
    volume' regime-detection description). Uses a rolling (not global)
    mean/std for the volume z-score so it stays well-defined incrementally
    as more data arrives each cycle. hmmlearn's GaussianHMM works fine
    with either 1 or 2 feature columns, so dropping volume here needs no
    change anywhere else in the HMM stage.
    """
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    if use_volume and 'Volume' in df.columns:
        vol_roll_mean = df['Volume'].rolling(20, min_periods=5).mean()
        vol_roll_std  = df['Volume'].rolling(20, min_periods=5).std()
        vol_z = (df['Volume'] - vol_roll_mean) / vol_roll_std.replace(0, np.nan)
        feats = pd.DataFrame({'log_ret': log_ret, 'vol_z': vol_z})
    else:
        feats = pd.DataFrame({'log_ret': log_ret})
    return feats


# ============================================================================
# STAGE 1 — HMM REGIME DETECTION
# ============================================================================

def fit_hmm(features_train, n_states=3, random_state=42):
    from hmmlearn.hmm import GaussianHMM
    X = features_train.dropna().values
    if len(X) < n_states * 10:
        return None
    model = GaussianHMM(n_components=n_states, covariance_type='diag',
                        n_iter=100, random_state=random_state)
    try:
        model.fit(X)
    except Exception:
        return None
    return model


def current_regime(hmm_model, features_train):
    """Most likely hidden state as of the last observation in the training
    window — this becomes 'today's regime' for the whole upcoming block."""
    if hmm_model is None:
        return -1
    X = features_train.dropna().values
    if len(X) == 0:
        return -1
    try:
        states = hmm_model.predict(X)
        return int(states[-1])
    except Exception:
        return -1


# ============================================================================
# STAGE 2 — SARIMAX LINEAR BASELINE + RESIDUAL EXTRACTION
# ============================================================================

def fit_sarimax(log_price_train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), verbose=True):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    try:
        # Fit on a plain (date-index-stripped) array. SARIMAX's forecast
        # index machinery breaks on a DatetimeIndex with no inferred
        # frequency ("No supported index is available") — real trading
        # data (irregular sessions, holidays) essentially never has a clean
        # freq, so fit/forecast entirely positionally and re-attach the
        # real dates ourselves in sarimax_insample_residuals instead.
        values = np.asarray(log_price_train.values, dtype=np.float64)
        model = SARIMAX(values, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False)
        return fitted
    except Exception as e:
        if verbose:
            print(f"    [SARIMAX FIT FAILED] {type(e).__name__}: {e}", flush=True)
        return None


def sarimax_block_forecast(fitted, steps, verbose=True):
    """Multi-step-ahead forecast from a single fit — this is the SARIMAX
    contribution for the whole upcoming block, refreshed next cycle.
    fitted was fit on a plain array (see fit_sarimax's index-stripping
    fix), so get_forecast() returns predicted_mean as a plain ndarray,
    not a pandas Series — no .values needed (or available) on it."""
    if fitted is None:
        return np.full(steps, np.nan)
    try:
        fc = fitted.get_forecast(steps=steps)
        pred = fc.predicted_mean
        return np.asarray(pred, dtype=np.float64)
    except Exception as e:
        if verbose:
            print(f"    [SARIMAX FORECAST FAILED] {type(e).__name__}: {e}", flush=True)
        return np.full(steps, np.nan)


def sarimax_insample_residuals(fitted, log_price_train):
    """eps_t = y_t - yhat_t,SARIMAX for the training window — feeds the LSTM.
    fitted.fittedvalues is now a plain positional array (see fit_sarimax's
    index-stripping fix), so re-attach log_price_train's real dates here
    rather than relying on any index carried by the fit itself."""
    if fitted is None:
        return pd.Series(np.nan, index=log_price_train.index)
    try:
        fitted_vals = np.asarray(fitted.fittedvalues, dtype=np.float64)
        log_vals    = np.asarray(log_price_train.values, dtype=np.float64)
        n = min(len(fitted_vals), len(log_vals))
        resid_vals = np.full(len(log_vals), np.nan)
        resid_vals[:n] = log_vals[:n] - fitted_vals[:n]
        return pd.Series(resid_vals, index=log_price_train.index)
    except Exception:
        return pd.Series(np.nan, index=log_price_train.index)


# ============================================================================
# STAGE 3 — LSTM NON-LINEAR RESIDUAL CORRECTION
# ============================================================================

def _get_torch_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class ResidualLSTM:
    """
    Thin wrapper around a single-layer LSTM + linear head predicting the
    next residual from a lookback window of prior residuals. Kept small
    (default hidden=16) since this retrains every --retrain-every bars —
    a heavy network here would dominate total pipeline runtime.

    GRU swap: this class only touches nn.LSTM in _build_model(); swapping
    to nn.GRU there (same input/output shapes) is a drop-in change if you
    want to A/B LSTM vs GRU later — nothing else in this file needs to change.
    """

    def __init__(self, lookback=20, hidden=16, epochs=30, lr=0.01, device=None):
        self.lookback = lookback
        self.hidden   = hidden
        self.epochs   = epochs
        self.lr       = lr
        self.device   = device or _get_torch_device()
        self.model    = None
        self.mean     = 0.0
        self.std      = 1.0

    def _build_model(self):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.lstm   = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
                self.head   = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        return _Net(self.hidden)

    def _make_sequences(self, resid_arr):
        X, y = [], []
        for i in range(self.lookback, len(resid_arr)):
            X.append(resid_arr[i - self.lookback:i])
            y.append(resid_arr[i])
        if not X:
            return None, None
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def fit(self, residuals, verbose=True):
        import torch
        import torch.nn as nn

        resid_arr = residuals.dropna().values.astype(np.float64)
        if len(resid_arr) < self.lookback + 10:
            if verbose:
                print(f"    [LSTM SKIPPED] only {len(resid_arr)} valid residual(s), "
                      f"need >= {self.lookback + 10}", flush=True)
            self.model = None
            return self

        self.mean = float(np.mean(resid_arr))
        self.std  = float(np.std(resid_arr)) or 1.0
        norm = (resid_arr - self.mean) / self.std

        X, y = self._make_sequences(norm)
        if X is None:
            if verbose:
                print(f"    [LSTM SKIPPED] no sequences built from "
                      f"{len(resid_arr)} residuals", flush=True)
            self.model = None
            return self

        X_t = torch.tensor(X).unsqueeze(-1).to(self.device)   # (N, lookback, 1)
        y_t = torch.tensor(y).unsqueeze(-1).to(self.device)   # (N, 1)

        self.model = self._build_model().to(self.device)
        opt  = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        try:
            for _ in range(self.epochs):
                opt.zero_grad()
                pred = self.model(X_t)
                loss = loss_fn(pred, y_t)
                loss.backward()
                opt.step()
        except Exception as e:
            if verbose:
                print(f"    [LSTM TRAIN FAILED] {type(e).__name__}: {e}", flush=True)
            self.model = None

        return self

    def predict_next(self, last_window_residuals):
        """Predict the single next residual from the last `lookback` actual
        residuals. Returns np.nan if the model didn't train (insufficient
        data) or the window is incomplete."""
        import torch

        if self.model is None:
            return np.nan
        window = np.asarray(last_window_residuals, dtype=np.float64)
        if len(window) < self.lookback or np.isnan(window).any():
            return np.nan
        window = window[-self.lookback:]
        norm = (window - self.mean) / self.std
        x = torch.tensor(norm, dtype=torch.float32).view(1, self.lookback, 1).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(x).item()
        return pred_norm * self.std + self.mean


# ============================================================================
# STAGE 4 — GARCH(1,1) RISK ASSESSMENT ON MASTER-PREDICTION ERRORS
# ============================================================================

def fit_garch(errors_train, verbose=True):
    from arch import arch_model
    e = errors_train.dropna().values
    if len(e) < 30:
        if verbose:
            print(f"    [GARCH SKIPPED] only {len(e)} valid error(s), need >= 30", flush=True)
        return None
    # arch_model works best with errors scaled to roughly O(1)-O(10); price-
    # log-return-scale errors are tiny (~1e-2), so scale up for numerical
    # stability and scale the resulting variance back down afterward.
    scale = 100.0
    try:
        am = arch_model(e * scale, vol='Garch', p=1, q=1, mean='Zero', rescale=False)
        res = am.fit(disp='off')
        return res, scale
    except Exception as e2:
        if verbose:
            print(f"    [GARCH FIT FAILED] {type(e2).__name__}: {e2}", flush=True)
        return None


def garch_block_vol_forecast(fitted_scale, steps):
    """Forecast volatility (not variance) for the block, unscaled back to
    the original error units."""
    if fitted_scale is None:
        return np.full(steps, np.nan)
    fitted, scale = fitted_scale
    try:
        fc = fitted.forecast(horizon=steps, reindex=False)
        var_scaled = fc.variance.values[-1]          # shape (steps,)
        vol_scaled = np.sqrt(var_scaled)
        return vol_scaled / scale
    except Exception:
        return np.full(steps, np.nan)


# ============================================================================
# WALK-FORWARD ORCHESTRATION
# ============================================================================

def run_alpha_pipeline(df, min_train=250, retrain_every=20,
                       hmm_states=3, sarimax_order=(1, 1, 1),
                       sarimax_seasonal_order=(0, 0, 0, 0),
                       lstm_lookback=20, lstm_hidden=16, lstm_epochs=30,
                       use_volume=False):
    n = len(df)
    close      = df['Close'].values
    log_price  = pd.Series(np.log(df['Close'].values), index=df.index)
    hmm_feats  = compute_hmm_features(df, use_volume=use_volume)

    regime_out    = np.full(n, -1, dtype=int)
    masterpred_out = np.full(n, np.nan)
    predvol_out    = np.full(n, np.nan)

    # Running store of realized residuals (actual - masterpred) — needed
    # both to feed the LSTM's lookback window and to fit GARCH each cycle.
    # Populated day-by-day as we walk forward, never using future values.
    realized_resid = np.full(n, np.nan)   # SARIMAX-stage residuals (eps_t)
    realized_error = np.full(n, np.nan)   # final master-prediction errors (E_t)

    start_t = time.time()
    n_cycles = max(1, (n - min_train + retrain_every - 1) // retrain_every)
    cycle_i  = 0

    t = min_train
    while t < n:
        cycle_i += 1
        block_end = min(t + retrain_every, n)
        block_len = block_end - t

        train_log_price = log_price.iloc[:t]
        train_feats     = hmm_feats.iloc[:t]

        elapsed = time.time() - start_t
        rate    = cycle_i / max(elapsed, 0.001)
        rem     = (n_cycles - cycle_i) / max(rate, 0.001)
        print(f"  [Cycle {cycle_i:>4}/{n_cycles}] bars {t}-{block_end} "
              f"({df.index[t].date()} -> {df.index[block_end-1].date()})  "
              f"El:{elapsed:.0f}s  ETA:{rem:.0f}s", flush=True)

        # ---- Stage 1: HMM regime -------------------------------------
        hmm_model = fit_hmm(train_feats, n_states=hmm_states)
        regime    = current_regime(hmm_model, train_feats)
        regime_out[t:block_end] = regime

        # ---- Stage 2: SARIMAX baseline + in-sample residuals ----------
        sarimax_fit = fit_sarimax(train_log_price, order=sarimax_order,
                                  seasonal_order=sarimax_seasonal_order)
        block_sarimax_fc = sarimax_block_forecast(sarimax_fit, block_len)
        train_resid = sarimax_insample_residuals(sarimax_fit, train_log_price)
        # Only backfill positions that have no realized residual yet (i.e.
        # the very first cycle's warm-start window). Bars already walked
        # past in a PRIOR cycle got their residual from the genuine
        # out-of-sample block forecast (see the loop below, `actual_log -
        # sarimax_fc_k`) — those must stay immutable. Overwriting them with
        # this cycle's in-sample fitted residuals would retroactively swap
        # true walk-forward residuals for smaller, better-behaved in-sample
        # ones (SARIMAX is fit to minimize exactly those), biasing the LSTM
        # toward an optimistic training distribution it won't see live.
        new_resid = train_resid.reindex(df.index[:t]).values
        mask = np.isnan(realized_resid[:t])
        realized_resid[:t][mask] = new_resid[mask]

        # ---- Stage 3: LSTM residual correction, day by day in block --
        lstm = ResidualLSTM(lookback=lstm_lookback, hidden=lstm_hidden,
                            epochs=lstm_epochs)
        lstm.fit(pd.Series(realized_resid[:t]))

        for k in range(block_len):
            idx = t + k
            window = realized_resid[max(0, idx - lstm_lookback):idx]
            pred_resid = lstm.predict_next(window) if len(window) >= lstm_lookback else np.nan

            sarimax_fc_k = block_sarimax_fc[k] if k < len(block_sarimax_fc) else np.nan
            if np.isnan(sarimax_fc_k) or np.isnan(pred_resid):
                master_log = np.nan
            else:
                master_log = sarimax_fc_k + pred_resid

            masterpred_out[idx] = np.exp(master_log) if not np.isnan(master_log) else np.nan

            # realize this bar's actual SARIMAX-stage residual and final
            # error now that we're walking past it — legitimate, not
            # lookahead, since idx has "elapsed" in the walk-forward loop
            actual_log = log_price.iloc[idx]
            if not np.isnan(sarimax_fc_k):
                realized_resid[idx] = actual_log - sarimax_fc_k
            if not np.isnan(master_log):
                realized_error[idx] = actual_log - master_log

        # ---- Stage 4: GARCH(1,1) on master-prediction errors ----------
        garch_fit = fit_garch(pd.Series(realized_error[:t]))
        block_vol = garch_block_vol_forecast(garch_fit, block_len)
        # convert log-space vol forecast to a price-fractional vol,
        # consistent with GarchVol's convention in the existing GARCH sizing
        predvol_out[t:block_end] = block_vol[:block_len]

        # ---- Diagnostic summary: where is this cycle actually failing? --
        n_valid_pred = int(np.sum(~np.isnan(masterpred_out[t:block_end])))
        n_valid_vol  = int(np.sum(~np.isnan(predvol_out[t:block_end])))
        print(f"    -> sarimax_fit={'OK' if sarimax_fit is not None else 'FAILED'}  "
              f"lstm_model={'OK' if lstm.model is not None else 'FAILED'}  "
              f"garch_fit={'OK' if garch_fit is not None else 'FAILED'}  "
              f"valid MasterPred={n_valid_pred}/{block_len}  "
              f"valid PredVol={n_valid_vol}/{block_len}", flush=True)

        t = block_end

    out = pd.DataFrame({
        'Regime':     regime_out,
        'MasterPred': masterpred_out,
        'PredVol':    predvol_out,
    }, index=df.index)
    return out


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Offline HMM->SARIMAX->LSTM->GARCH alpha trainer")
    parser.add_argument('--file',          type=str, required=True)
    parser.add_argument('--output',        type=str, default='alpha_output.csv')
    parser.add_argument('--min-train',     type=int, default=250,
                        help='Burn-in bars before the first cycle (default 250)')
    parser.add_argument('--retrain-every', type=int, default=20,
                        help='Bars per walk-forward cycle (default 20)')
    parser.add_argument('--hmm-states',    type=int, default=3)
    parser.add_argument('--sarimax-order', type=int, nargs=3, default=[1, 1, 1],
                        metavar=('p', 'd', 'q'))
    parser.add_argument('--sarimax-seasonal-order', type=int, nargs=4,
                        default=[0, 0, 0, 0], metavar=('P', 'D', 'Q', 's'))
    parser.add_argument('--lstm-lookback', type=int, default=20)
    parser.add_argument('--lstm-hidden',   type=int, default=16)
    parser.add_argument('--lstm-epochs',   type=int, default=30)
    parser.add_argument('--use-volume',    action='store_true',
                        help='Load the Volume column and include a rolling volume '
                             'z-score in the HMM regime features. Default: off — only '
                             'OHLC is loaded/required. Errors clearly if the CSV has '
                             'no Volume column and this flag is passed.')
    parser.add_argument('--start-date',    type=str, default=None)
    parser.add_argument('--end-date',      type=str, default=None)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  ALPHA PIPELINE — HMM -> SARIMAX -> LSTM -> GARCH (walk-forward)")
    print(f"{'='*70}")

    df = load_price_data(args.file, use_volume=args.use_volume)
    if args.start_date:
        df = df[df.index >= args.start_date]
    if args.end_date:
        df = df[df.index <= args.end_date]

    print(f"  Candles       : {len(df)}")
    print(f"  Range         : {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"  Min train     : {args.min_train} bars")
    print(f"  Retrain every : {args.retrain_every} bars")
    print(f"  HMM states    : {args.hmm_states}")
    print(f"  HMM features  : log_ret" + (" + vol_z (--use-volume)" if args.use_volume else " only (pass --use-volume to add volume)"))
    print(f"  SARIMAX order : {tuple(args.sarimax_order)}  "
          f"seasonal {tuple(args.sarimax_seasonal_order)}")
    print(f"  LSTM          : lookback={args.lstm_lookback}  "
          f"hidden={args.lstm_hidden}  epochs={args.lstm_epochs}")

    try:
        import torch
        device = 'mps' if torch.backends.mps.is_available() else (
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Torch device  : {device}")
    except ImportError:
        print(f"\nERROR: torch not installed. pip install torch --break-system-packages")
        sys.exit(1)

    if len(df) < args.min_train + args.retrain_every:
        print(f"\nERROR: not enough data ({len(df)} bars) for "
              f"min_train={args.min_train} + retrain_every={args.retrain_every}")
        sys.exit(1)

    print(f"{'='*70}\n")

    out = run_alpha_pipeline(
        df, min_train=args.min_train, retrain_every=args.retrain_every,
        hmm_states=args.hmm_states,
        sarimax_order=tuple(args.sarimax_order),
        sarimax_seasonal_order=tuple(args.sarimax_seasonal_order),
        lstm_lookback=args.lstm_lookback, lstm_hidden=args.lstm_hidden,
        lstm_epochs=args.lstm_epochs,
        use_volume=args.use_volume,
    )

    n_valid = out['MasterPred'].notna().sum()
    print(f"\n{'─'*70}")
    print(f"  DONE — {n_valid}/{len(out)} bars have a valid MasterPred/PredVol "
          f"(earlier bars are burn-in)")
    out.to_csv(args.output)
    print(f"  Saved -> {args.output}")
    print(f"{'─'*70}\n")


if __name__ == '__main__':
    main()