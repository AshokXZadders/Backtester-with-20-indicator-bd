'''
sys, os, json, warnings, argparse, time, traceback, itertools, random, collections, and multiprocessing are all standard library
'''

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TQDM_DISABLE'] = '1'

_GLOBAL_FIB_CACHE        = {}
_GLOBAL_DF               = None
_GLOBAL_INDICATOR_GRID   = None
_GLOBAL_ALL_INDICATORS   = None
_GLOBAL_SHORTLIST        = None
_GLOBAL_RR_GRID          = None
_GLOBAL_MIN_TRADES_HARD  = None
_GLOBAL_MIN_TRADES_SOFT  = None
_GLOBAL_CASH             = None
_GLOBAL_COMMISSION       = None

_CHECKPOINT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "datacan5_checkpoint.json"
)


def _jsonable(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _combo_names(combo):
    if not combo:
        return []
    return [c.replace('use_', '') for c in combo]


def _stats_snapshot(stats):
    if stats is None:
        return None
    keys = [
        'Return [%]', 'Sharpe Ratio', 'Max. Drawdown [%]',
        'Equity Final [$]', '# Trades', 'Win Rate [%]',
        'Profit Factor', 'Sortino Ratio', 'Avg. Trade [%]',
        'Avg. Trade Duration', 'Buy & Hold Return [%]'
    ]
    out = {}
    for key in keys:
        try:
            out[key] = _jsonable(stats.get(key, None))
        except Exception:
            out[key] = None
    return out


def _save_checkpoint(payload):
    data = {
        'updated_at': time.time(),
        **_jsonable(payload),
    }
    tmp_path = _CHECKPOINT_FILE + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, _CHECKPOINT_FILE)


def _load_checkpoint():
    if not os.path.exists(_CHECKPOINT_FILE):
        return None
    try:
        with open(_CHECKPOINT_FILE, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except Exception:
        return None


def _delete_checkpoint():
    try:
        if os.path.exists(_CHECKPOINT_FILE):
            os.remove(_CHECKPOINT_FILE)
    except Exception:
        pass


def _print_checkpoint_summary(cp):
    if not cp:
        return
    print(f"\n{'='*70}")
    print(f"  CHECKPOINT RECOVERY")
    print(f"{'='*70}")
    print(f"  File   : {_CHECKPOINT_FILE}")
    print(f"  Stage  : {cp.get('stage', 'unknown')}")
    completed = cp.get('completed')
    total = cp.get('total')
    if completed is not None and total is not None:
        print(f"  Progress: {completed}/{total}")
    best_combo = cp.get('best_combo')
    if best_combo:
        print(f"  Best   : {'+'.join(best_combo)}")
    best_score = cp.get('best_score')
    if best_score is not None:
        try:
            print(f"  Score  : {float(best_score):.4f}")
        except Exception:
            print(f"  Score  : {best_score}")
    top_results = cp.get('top_results') or []
    if top_results:
        print(f"\n  Top saved results:")
        for row in top_results[:5]:
            combo = '+'.join(row.get('combo', []))
            score = row.get('score', None)
            trades = row.get('trades', None)
            ret = row.get('return_pct', None)
            if score is None:
                continue
            try:
                print(f"    {float(score):>8.4f}  {combo:<30}  {trades:>5} trades  {float(ret):>7.2f}%")
            except Exception:
                print(f"    {score}  {combo}")
    shortlist = cp.get('shortlist') or []
    if shortlist:
        print(f"\n  Shortlist: {shortlist}")
    print(f"{'='*70}")


def _init_worker(cache, df, indicator_grid, all_indicators, shortlist,
                 rr_grid, min_trades_hard, min_trades_soft,
                 cash, commission, warmup_bars=0):
    global _GLOBAL_FIB_CACHE, _GLOBAL_DF, _GLOBAL_INDICATOR_GRID
    global _GLOBAL_ALL_INDICATORS, _GLOBAL_SHORTLIST
    global _GLOBAL_RR_GRID, _GLOBAL_MIN_TRADES_HARD, _GLOBAL_MIN_TRADES_SOFT
    global _GLOBAL_CASH, _GLOBAL_COMMISSION
    try:
        sys.stderr = open(os.devnull, 'w')
    except Exception:
        pass
    _GLOBAL_FIB_CACHE       = cache
    _GLOBAL_DF              = df
    _GLOBAL_INDICATOR_GRID  = indicator_grid
    _GLOBAL_ALL_INDICATORS  = all_indicators
    _GLOBAL_SHORTLIST       = shortlist
    _GLOBAL_RR_GRID         = rr_grid.copy()
    _GLOBAL_RR_GRID['warmup_bars'] = [warmup_bars]
    _GLOBAL_MIN_TRADES_HARD = min_trades_hard
    _GLOBAL_MIN_TRADES_SOFT = min_trades_soft
    _GLOBAL_CASH            = cash
    _GLOBAL_COMMISSION      = commission


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_supertrend(high, low, close, period=10, factor=3.0):
    n   = len(close)
    atr = talib.ATR(high, low, close, timeperiod=period)
    hl2 = (high + low) / 2.0
    upper_b = hl2 + factor * atr
    lower_b = hl2 - factor * atr
    final_up = np.copy(upper_b)
    final_lo = np.copy(lower_b)
    for i in range(1, n):
        if np.isnan(upper_b[i]) or np.isnan(final_up[i-1]):
            continue
        final_up[i] = (upper_b[i]
                       if (upper_b[i] < final_up[i-1] or close[i-1] > final_up[i-1])
                       else final_up[i-1])
        final_lo[i] = (lower_b[i]
                       if (lower_b[i] > final_lo[i-1] or close[i-1] < final_lo[i-1])
                       else final_lo[i-1])
    direction = np.zeros(n)
    st_line   = np.full(n, np.nan)
    for i in range(period, n):
        if np.isnan(final_up[i]) or np.isnan(final_lo[i]):
            continue
        if i == period:
            direction[i] = 1 if close[i] <= final_up[i] else -1
        else:
            if   direction[i-1] == -1 and close[i] > final_up[i]: direction[i] =  1
            elif direction[i-1] ==  1 and close[i] < final_lo[i]: direction[i] = -1
            else:                                                   direction[i] = direction[i-1]
        st_line[i] = final_lo[i] if direction[i] == 1 else final_up[i]
    return st_line, direction


def compute_ichimoku(high, low, close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    n        = len(close)
    h_series = pd.Series(high)
    l_series = pd.Series(low)
    tenkan   = ((h_series.rolling(tenkan_p).max()   + l_series.rolling(tenkan_p).min())   / 2).values
    kijun    = ((h_series.rolling(kijun_p).max()    + l_series.rolling(kijun_p).min())    / 2).values
    sen_b    = ((h_series.rolling(senkou_b_p).max() + l_series.rolling(senkou_b_p).min()) / 2).values
    sen_a    = (tenkan + kijun) / 2
    sen_a_s  = np.full(n, np.nan)
    sen_b_s  = np.full(n, np.nan)
    shift    = kijun_p
    for i in range(shift, n):
        if not np.isnan(sen_a[i - shift]): sen_a_s[i] = sen_a[i - shift]
        if not np.isnan(sen_b[i - shift]): sen_b_s[i] = sen_b[i - shift]
    return tenkan, kijun, sen_a_s, sen_b_s


def compute_vwap(high, low, close, volume):
    typical = (high + low + close) / 3.0
    cumtpv  = np.cumsum(typical * volume)
    cumvol  = np.cumsum(volume)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(cumvol > 0, cumtpv / cumvol, np.nan)


def compute_obv_divergence(close, volume, lookback=20):
    obv    = talib.OBV(close, volume)
    n      = len(close)
    signal = np.zeros(n)
    for i in range(lookback, n):
        price_slope = close[i] - close[i - lookback]
        obv_slope   = obv[i]   - obv[i - lookback]
        if   price_slope < 0 and obv_slope > 0: signal[i] =  1
        elif price_slope > 0 and obv_slope < 0: signal[i] = -1
    return signal


def compute_slippage(atr_val, volume_val, avg_volume, close_price,
                     atr_mult=0.1, vol_mult=0.1, max_slip_pct=0.005):
    if atr_val is None or np.isnan(atr_val) or atr_val <= 0:
        return 0.0
    if avg_volume is None or avg_volume <= 0:
        vol_factor = 1.0
    else:
        vol_ratio  = avg_volume / max(volume_val, 1.0)
        vol_factor = np.clip(vol_ratio, 0.5, 3.0)
    raw_slip = (atr_mult * atr_val) + (vol_mult * atr_val * vol_factor)
    return float(min(raw_slip, max_slip_pct * close_price))


# ============================================================================
# SWING DETECTION
# ============================================================================

def find_swing_points_enhanced(high, low, close, base_lookback=5,
                                min_prominence_atr=0.5, min_lb=3, max_lb=20):
    n   = len(close)
    atr = talib.ATR(high, low, close, timeperiod=14)
    swing_highs   = np.full(n, np.nan)
    swing_lows    = np.full(n, np.nan)
    swing_quality = np.zeros(n)
    valid_atr = atr[~np.isnan(atr)]
    med_atr   = np.median(valid_atr) if len(valid_atr) else 1.0
    med_close = np.median(close)
    vol_ratio = med_atr / med_close if med_close > 0 else 0.01
    lb = int(np.clip(
        round(min_lb + (vol_ratio / 0.03) * (max_lb - min_lb)),
        min_lb, max_lb
    ))
    sh_idx = argrelextrema(high, np.greater_equal, order=lb)[0]
    sl_idx = argrelextrema(low,  np.less_equal,    order=lb)[0]
    for i in sh_idx:
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        lo   = max(0, i - lb)
        hi   = min(n - 1, i + lb)
        prom = (high[i] - np.mean(close[lo:hi+1])) / atr[i]
        if prom >= min_prominence_atr:
            swing_highs[i]   = high[i]
            swing_quality[i] = prom
    for i in sl_idx:
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        lo   = max(0, i - lb)
        hi   = min(n - 1, i + lb)
        prom = (np.mean(close[lo:hi+1]) - low[i]) / atr[i]
        if prom >= min_prominence_atr:
            swing_lows[i]    = low[i]
            swing_quality[i] = -prom
    return swing_highs, swing_lows, swing_quality


def find_last_swing_pair_quality(swing_highs, swing_lows, swing_quality,
                                  current_idx, min_quality=0.5):
    last_sh = last_sl = np.nan
    last_sh_idx = last_sl_idx = -1
    for i in range(current_idx - 1, -1, -1):
        if last_sh_idx == -1 and not np.isnan(swing_highs[i]):
            if abs(swing_quality[i]) >= min_quality:
                last_sh, last_sh_idx = swing_highs[i], i
        if last_sl_idx == -1 and not np.isnan(swing_lows[i]):
            if abs(swing_quality[i]) >= min_quality:
                last_sl, last_sl_idx = swing_lows[i], i
        if last_sh_idx != -1 and last_sl_idx != -1:
            break
    return last_sh, last_sh_idx, last_sl, last_sl_idx


# ============================================================================
# PARAMETER GRID
# ============================================================================

def get_indicator_grid():
    return {
        'use_rsi':        {'rsi_period': [14, 21],     'rsi_lower': [30.0, 40.0],
                           'rsi_upper': [60.0, 70.0]},
        'use_macd':       {'macd_fast': [12, 15],      'macd_slow': [26, 28],
                           'macd_signal_p': [9, 11]},
        'use_bb':         {'bb_period': [20, 25],      'bb_std': [2.0, 2.5]},
        'use_ema':        {'ema_fast': [9, 12],        'ema_slow': [21, 26]},
        'use_sma':        {'sma_fast': [20, 50],       'sma_slow': [100, 200]},
        'use_stoch':      {'stoch_k': [14, 18],        'stoch_d': [3, 5],
                           'stoch_lower': [20.0, 30.0],'stoch_upper': [70.0, 80.0]},
        'use_cci':        {'cci_period': [14, 20],     'cci_lower': [-100.0, -50.0],
                           'cci_upper': [100.0, 150.0]},
        'use_willr':      {'willr_period': [14, 20],   'willr_lower': [-80.0, -70.0],
                           'willr_upper': [-30.0, -20.0]},
        'use_supertrend': {'st_period': [10, 14],      'st_factor': [2.5, 3.5]},
        'use_ichimoku':   {'ichi_tenkan': [9, 13],     'ichi_kijun': [26, 30],
                           'ichi_senkou_b': [52, 60]},
        'use_adx':        {'adx_period': [14, 20],     'adx_threshold': [20.0, 25.0]},
        'use_multi_ema':  {'mema_fast': [12, 15],      'mema_slow': [100, 200]},
        'use_keltner':    {'kc_period': [20, 25],      'kc_mult': [1.5, 2.0]},
        'use_donchian':   {'dc_period': [20, 25],      'dc_tol': [0.5, 1.0]},
        'use_stddev':     {'sd_period': [20, 25],      'sd_mult': [2.0, 2.5]},
        'use_pivot':      {},
        'use_fib':        {'fib_lookback': [3, 5],     'fib_tol': [0.1, 0.2],
                           'fib_min_prominence': [0.5, 1.0]},
        'use_sr':         {'sr_lookback': [30, 50],    'sr_tol': [0.1, 0.3]},
        'use_mfi':        {'mfi_period': [14, 21],     'mfi_lower': [20.0, 30.0],
                           'mfi_upper': [70.0, 80.0]},
        'use_vwap':       {},
        'use_obv':        {'obv_lookback': [10, 20]},
        'use_cmf':        {'cmf_period': [14, 20]},
        'use_vol_spike':  {'vs_mult': [1.5, 2.0],      'vs_period': [14, 20]},
    }


# ============================================================================
# SCORING
# ============================================================================

MIN_TRADES_HARD = 30
MIN_TRADES_SOFT = 50
MIN_TRADES      = MIN_TRADES_SOFT

RR_GRID = {
    'use_atr':       [True],
    'rr_ratio':      [ 1.5, 3.0],
    'atr_sl_mult':   [1.5, 2.5],
    'risk_pct':      [0.02],
    'atr_period':    [14],
    'max_hold_bars': [15,20]
}


def score_stats(stats, lenient=False):
    try:
        ret      = float(stats.get('Return [%]',      0.0) or 0.0)
        trades   = int(stats.get('# Trades',          0)   or 0)
        sharpe   = float(stats.get('Sharpe Ratio',    0.0) or 0.0)
        eq_final = float(stats.get('Equity Final [$]',0.0) or 0.0)
        eq_peak  = float(stats.get('Equity Peak [$]', 0.0) or 0.0)
        pf       = float(stats.get('Profit Factor',   0.0) or 0.0)
        if trades < MIN_TRADES_HARD:
            return -1000.0
        if not lenient and ret <= 0:
            return -1000.0
        if trades < MIN_TRADES_SOFT:
            trade_penalty = 0.70 + 0.30 * (
                (trades - MIN_TRADES_HARD) / max(MIN_TRADES_SOFT - MIN_TRADES_HARD, 1)
            )
        else:
            trade_penalty = 1.0
        ret_score    = ret / 100.0
        sharpe_score = max(min(sharpe / 5.0, 1.0), 0.0)
        base_cash    = 100_000.0
        final_gain   = (eq_final - base_cash) / base_cash
        peak_gain    = (eq_peak  - base_cash) / base_cash
        trade_bonus  = min(trades / float(MIN_TRADES_SOFT), 1.0)
        pf_score     = max(min((pf - 1.0) / 2.0, 1.0), 0.0)
        raw_score = (0.30 * sharpe_score
                   + 0.30 * ret_score
                   + 0.15 * pf_score
                   + 0.10 * max(peak_gain,  0.0)
                   + 0.10 * max(final_gain, 0.0)
                   + 0.05 * trade_bonus)
        try:
            comp = compute_compounded_stats(stats)
            if comp:
                comp_ret_norm = comp['compounded_return_pct'] / 100.0
                comp_dd_pen   = abs(comp['compounded_max_dd']) / 100.0
                comp_bonus    = max(comp_ret_norm - comp_dd_pen * 0.5, 0.0) * 0.10
                raw_score    += comp_bonus
        except Exception:
            pass
        return float(raw_score * trade_penalty)
    except Exception:
        return -1000.0


def compute_compounded_stats(stats, cash=100_000, risk_pct=0.01, reinvest_rate=0.50):
    """
    FIX-2 (retained): Compounded Sharpe annualised with sqrt(252).
    """
    try:
        trades = stats._trades
        if trades is None or len(trades) == 0:
            return {}
    except AttributeError:
        return {}

    try:
        actual_rr = float(stats._strategy.rr_ratio)
    except AttributeError:
        actual_rr = 2.0

    capital      = float(cash)
    base_cash    = float(cash)
    base_risk    = base_cash * risk_pct
    equity_curve = [capital]

    for _, row in trades.iterrows():
        pnl = float(row.get('PnL', 0) or 0)
        if pnl > 0:
            capital += pnl * reinvest_rate
        else:
            capital += max(pnl, -base_risk)
        equity_curve.append(capital)

    equity_arr  = np.array(equity_curve)
    comp_return = (capital - base_cash) / base_cash * 100.0
    daily_rets  = np.diff(equity_arr) / equity_arr[:-1]

    # FIX-2: annualise with sqrt(252), not sqrt(n_trades)
    comp_sharpe = (np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
               if np.std(daily_rets) > 0 else 0.0)

    peak    = np.maximum.accumulate(equity_arr)
    comp_dd = float(np.min((equity_arr - peak) / peak * 100))

    return {
        'compounded_return_pct': round(comp_return, 2),
        'compounded_final':      round(capital, 2),
        'compounded_sharpe':     round(comp_sharpe, 3),
        'compounded_max_dd':     round(comp_dd, 2),
        'reinvest_rate':         reinvest_rate,
        'rr_ratio':              actual_rr,
    }


# ============================================================================
# STAGE 1 — SOLO INDICATOR SCREEN
# ============================================================================

def run_solo_screen(df, cash, commission_frac, signal_mode=2, warmup_bars=0):
    indicator_grid = get_indicator_grid()
    all_indicators = list(indicator_grid.keys())

    SCREEN_RR = {
        'use_atr':       True,
        'rr_ratio':      2.0,
        'atr_sl_mult':   1.5,
        'risk_pct':      0.01,
        'atr_period':    14,
        'max_hold_bars': 10,
        'warmup_bars':   warmup_bars,
    }

    print(f"\n{'='*70}")
    print(f"  STAGE 1 — SOLO INDICATOR SCREEN")
    print(f"  Testing {len(all_indicators)} indicators individually")
    print(f"  Data    : {len(df)} candles")
    print(f"  Gate    : >= {MIN_TRADES_HARD} trades  (return not required positive)")
    print(f"  FIX-1/4 : fills at next-bar open, SL/TP from actual fill price")
    print(f"{'='*70}\n")
    print(f"  {'#':>3}  {'Indicator':<16} {'Trades':>7} {'Return':>8} "
          f"{'Sharpe':>7} {'WinR':>6} {'Score':>9}  Status        ETA")
    print(f"  {'─'*78}")

    results = []
    start_t = time.time()

    try:
        for i, ind in enumerate(all_indicators):
            try:
                bt = Backtest(df, MultiIndicatorStrategy,
                              cash=cash, commission=commission_frac,
                              exclusive_orders=True)
                run_kwargs = {k: False for k in all_indicators}
                run_kwargs[ind] = True
                run_kwargs['signal_mode'] = signal_mode
                for param, vals in indicator_grid[ind].items():
                    run_kwargs[param] = vals[0]
                run_kwargs.update(SCREEN_RR)
                stats  = bt.run(**run_kwargs)
                trades = int(stats.get('# Trades',     0) or 0)
                ret    = float(stats.get('Return [%]',  0) or 0)
                sharpe = float(stats.get('Sharpe Ratio',0) or 0)
                winr   = float(stats.get('Win Rate [%]',0) or 0)
                sc     = score_stats(stats, lenient=True)
                elapsed = time.time() - start_t
                rate    = (i + 1) / elapsed if elapsed > 0 else 1
                rem     = (len(all_indicators) - i - 1) / rate
                eta     = f"{int(rem)}s" if rem < 60 else f"{int(rem/60)}m{int(rem%60):02d}s"
                if   trades == 0:                status = "✗ no trades"
                elif trades < MIN_TRADES_HARD:   status = f"✗ too few ({trades})"
                elif sc > 0.10:                  status = "✓ GOOD"
                elif ret > 0:                    status = "~ weak (pos)"
                else:                            status = "~ weak (neg)"
                print(f"  {i+1:>3}  {ind.replace('use_',''):<16} {trades:>7d} "
                      f"{ret:>7.2f}% {sharpe:>7.3f} {winr:>5.1f}% "
                      f"{sc:>9.4f}  {status:<13} {eta}", flush=True)
                results.append((sc, ind, trades, ret, sharpe))
            except Exception as e:
                elapsed = time.time() - start_t
                rate    = (i + 1) / max(elapsed, 0.001)
                rem     = (len(all_indicators) - i - 1) / rate
                eta     = f"{int(rem)}s" if rem < 60 else f"{int(rem/60)}m{int(rem%60):02d}s"
                print(f"  {i+1:>3}  {ind.replace('use_',''):<16} {'ERROR':>7}  "
                      f"{str(e)[:35]:<35}  {eta}", flush=True)
                results.append((-1000.0, ind, 0, 0.0, 0.0))

            partial_results = [
                {'score': float(sc), 'indicator': name.replace('use_', ''),
                 'trades': int(tr), 'return_pct': float(ret), 'sharpe': float(sh)}
                for sc, name, tr, ret, sh in results
            ]
            partial_valid = [row for row in partial_results if row['trades'] >= MIN_TRADES_HARD]
            partial_valid.sort(key=lambda x: x['score'], reverse=True)
            _save_checkpoint({
                'stage': 'stage1', 'status': 'running',
                'completed': i + 1, 'total': len(all_indicators),
                'results': partial_results, 'valid': partial_valid,
                'shortlist': [row['indicator'] for row in partial_valid],
                'best_score': partial_valid[0]['score'] if partial_valid else None,
                'best_combo': None,
            })
    except KeyboardInterrupt:
        partial_results = [
            {'score': float(sc), 'indicator': name.replace('use_', ''),
             'trades': int(tr), 'return_pct': float(ret), 'sharpe': float(sh)}
            for sc, name, tr, ret, sh in results
        ]
        partial_valid = [row for row in partial_results if row['trades'] >= MIN_TRADES_HARD]
        partial_valid.sort(key=lambda x: x['score'], reverse=True)
        _save_checkpoint({
            'stage': 'stage1', 'status': 'interrupted',
            'completed': len(results), 'total': len(all_indicators),
            'results': partial_results, 'valid': partial_valid,
            'shortlist': [row['indicator'] for row in partial_valid],
            'best_score': partial_valid[0]['score'] if partial_valid else None,
            'best_combo': None,
        })
        print(f"\n  Stage 1 interrupted — checkpoint saved.")
        raise

    valid   = [(sc, ind, tr, ret, sh) for sc, ind, tr, ret, sh in results if tr >= MIN_TRADES_HARD]
    invalid = [(sc, ind, tr, ret, sh) for sc, ind, tr, ret, sh in results if tr < MIN_TRADES_HARD]
    valid.sort(key=lambda x: x[0], reverse=True)

    total_time = time.time() - start_t
    print(f"\n{'─'*70}")
    print(f"  SCREEN COMPLETE  ({int(total_time)}s)")
    print(f"  Valid (>={MIN_TRADES_HARD} trades) : {len(valid)}")
    print(f"  Eliminated (< {MIN_TRADES_HARD} trades) : {len(invalid)}")

    if valid:
        print(f"\n  {'Rank':<5} {'Indicator':<18} {'Score':>8} {'Trades':>7} {'Return':>8} {'Sharpe':>8}")
        print(f"  {'─'*60}")
        for rank, (sc, ind, tr, ret, sh) in enumerate(valid, 1):
            marker = "★" if ret > 0 else " "
            print(f"  {marker}#{rank:<3} {ind.replace('use_',''):<18} "
                  f"{sc:>8.4f} {tr:>7d} {ret:>7.2f}% {sh:>8.3f}")

    if invalid:
        print(f"\n  Eliminated (zero/insufficient trades):")
        for _, ind, tr, _, _ in invalid:
            print(f"    ✗ {ind.replace('use_','')}  ({tr} trades)")

    print(f"\n{'─'*70}")
    return valid, [ind for _, ind, _, _, _ in valid]


# ============================================================================
# STAGE 2 — COMBO SEARCH
# ============================================================================

def _run_combo_worker(combo):
    try:
        from backtesting import Backtest
        MultiIndicatorStrategy._swing_cache = _GLOBAL_FIB_CACHE
        bt = Backtest(
            _GLOBAL_DF, MultiIndicatorStrategy,
            cash=_GLOBAL_CASH, commission=_GLOBAL_COMMISSION,
            exclusive_orders=True
        )
        opt_kwargs = {ind: [False] for ind in _GLOBAL_ALL_INDICATORS}
        for ind in combo:
            opt_kwargs[ind] = [True]
            for p, vals in _GLOBAL_INDICATOR_GRID[ind].items():
                opt_kwargs[p] = vals
        opt_kwargs.update(_GLOBAL_RR_GRID)
        stats = bt.optimize(
            **opt_kwargs,
            maximize=score_stats,
            return_heatmap=False
        )
        sc = score_stats(stats)
        return (sc, combo, stats)
    except Exception:
        return (-1000.0, combo, None)


def _build_fib_cache(df):
    from itertools import product as iproduct
    high  = np.ascontiguousarray(df['High'].values,  dtype=np.float64)
    low   = np.ascontiguousarray(df['Low'].values,   dtype=np.float64)
    close = np.ascontiguousarray(df['Close'].values, dtype=np.float64)
    cache = {}
    for lb, prom, min_lb, max_lb in iproduct([3, 5], [0.5, 1.0], [3], [20]):
        key        = (lb, prom, min_lb, max_lb)
        cache[key] = find_swing_points_enhanced(
            high, low, close,
            base_lookback=lb, min_prominence_atr=prom,
            min_lb=min_lb, max_lb=max_lb
        )
    return cache


def _format_time(seconds):
    if seconds > 3600: return f"{int(seconds/3600)}h{int((seconds%3600)/60):02d}m"
    if seconds > 60:   return f"{int(seconds/60)}m{int(seconds%60):02d}s"
    return f"{int(seconds)}s"


def _print_combo_line(completed, total, elapsed, rem, sc, ret, trades,
                      sharpe, winr, pf, names, best_names, best_sc, skip=False):
    el_str  = _format_time(elapsed)
    eta_str = _format_time(rem)
    pct     = completed / total * 100
    bar_w   = 18
    filled  = int(bar_w * completed / total)
    bar     = '█' * filled + '░' * (bar_w - filled)
    if skip:
        print(f"  [{bar}] {pct:5.1f}%  {completed:4d}/{total}"
              f"  El:{el_str:<7} ETA:{eta_str:<7}"
              f"  SKIP  {names}", flush=True)
    else:
        print(f"  [{bar}] {pct:5.1f}%  {completed:4d}/{total}"
              f"  El:{el_str:<7} ETA:{eta_str:<7}"
              f"  Sc:{sc:>7.4f}  Ret:{ret:>7.2f}%  Tr:{trades:>4d}"
              f"  Sh:{sharpe:>6.3f}  WR:{winr:>5.1f}%  PF:{pf:>4.2f}"
              f"  {names}", flush=True)
        if best_names:
            print(f"  {'':>{bar_w+2}}{'':9}"
                  f"  ↳ Best: {best_names}  ({best_sc:.4f})", flush=True)


def run_combo_search(df, cash, commission_frac, shortlist,
                     all_indicators, sizes=(2, 3), n_jobs=-1,
                     reinvest_rate=0.50, warmup_bars=0):
    indicator_grid = get_indicator_grid()
    sizes  = tuple(sizes)
    combos = []
    for r in sizes:
        combos.extend(list(itertools.combinations(shortlist, r)))
    random.shuffle(combos)          # interleave pairs/triples so ETA reflects real mix from the start
    total = len(combos)
    n_workers = max(1, cpu_count() - 1) if n_jobs == -1 else max(1, min(n_jobs, cpu_count()))

    print(f"\n{'='*70}")
    print(f"  STAGE 2 — COMBO SEARCH")
    print(f"  Shortlist   : {len(shortlist)} indicators → "
          f"{sum(1 for c in combos if len(c)==2)} pairs, "
          f"{sum(1 for c in combos if len(c)==3)} triples")
    print(f"  Total jobs  : {total}")
    print(f"  Workers     : {n_workers}")
    print(f"{'='*70}\n")

    print("  Pre-computing Fibonacci swing cache...", flush=True)
    fib_cache = _build_fib_cache(df)
    print(f"  Swing cache ready — {len(fib_cache)} variants.\n", flush=True)

    stage2_rr = dict(RR_GRID)
    completed   = 0
    start_time  = time.time()
    results_log = []

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(fib_cache, df, indicator_grid,
                  all_indicators, shortlist,
                  stage2_rr, MIN_TRADES_HARD, MIN_TRADES_SOFT,
                  cash, commission_frac, warmup_bars),
        maxtasksperchild=16,
    ) as pool:
        print(f"  Pool ready. Dispatching {total} jobs...", flush=True)
        result_iter = pool.imap_unordered(_run_combo_worker, combos, chunksize=1)
        first_done   = False
        recent_times = deque(maxlen=50)   # rolling window for ETA, not whole-run average
        try:
            for result in result_iter:
                if not first_done:
                    print(f"  ✓ First result in {time.time()-start_time:.1f}s\n", flush=True)
                    first_done = True
                sc, combo, stats = result
                completed += 1
                now = time.time()
                recent_times.append(now)
                elapsed = now - start_time
                if len(recent_times) >= 2:
                    rate = (len(recent_times) - 1) / max(recent_times[-1] - recent_times[0], 0.001)
                else:
                    rate = completed / max(elapsed, 0.001)
                rem = (total - completed) / max(rate, 0.001)
                names     = '+'.join(c.replace('use_', '') for c in combo)
                if stats is not None and sc > -1000.0:
                    results_log.append((sc, combo, stats))
                    try:
                        ret    = float(stats.get('Return [%]',    0) or 0)
                        trades = int(stats.get('# Trades',        0) or 0)
                        sharpe = float(stats.get('Sharpe Ratio',  0) or 0)
                        winr   = float(stats.get('Win Rate [%]',  0) or 0)
                        pf     = float(stats.get('Profit Factor', 0) or 0)
                    except Exception:
                        ret = trades = sharpe = winr = pf = 0
                    best       = max(results_log, key=lambda x: x[0])
                    best_names = '+'.join(c.replace('use_','') for c in best[1])
                    _print_combo_line(completed, total, elapsed, rem,
                                      sc, ret, trades, sharpe, winr, pf,
                                      names, best_names, best[0])
                else:
                    _print_combo_line(completed, total, elapsed, rem,
                                      0, 0, 0, 0, 0, 0, names, None, 0, skip=True)

                top_results = sorted(results_log, key=lambda x: x[0], reverse=True)[:5]
                _save_checkpoint({
                    'stage': 'stage2', 'status': 'running',
                    'completed': completed, 'total': total,
                    'shortlist': _combo_names(shortlist),
                    'best_combo': _combo_names(top_results[0][1]) if top_results else None,
                    'best_score': top_results[0][0] if top_results else None,
                    'top_results': [
                        {'score': float(sc_i), 'combo': _combo_names(combo_i),
                         'trades': int(st_i.get('# Trades', 0) or 0),
                         'return_pct': float(st_i.get('Return [%]', 0) or 0),
                         'sharpe': float(st_i.get('Sharpe Ratio', 0) or 0)}
                        for sc_i, combo_i, st_i in top_results
                    ],
                })
        except KeyboardInterrupt:
            top_results = sorted(results_log, key=lambda x: x[0], reverse=True)[:5]
            _save_checkpoint({
                'stage': 'stage2', 'status': 'interrupted',
                'completed': completed, 'total': total,
                'shortlist': _combo_names(shortlist),
                'best_combo': _combo_names(top_results[0][1]) if top_results else None,
                'best_score': top_results[0][0] if top_results else None,
            })
            print(f"\n  Stage 2 interrupted — checkpoint saved.")
            raise

    total_time = time.time() - start_time
    results_log.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'─'*70}")
    print(f"  STAGE 2 COMPLETE  ({_format_time(total_time)})")
    print(f"  Valid combos : {len(results_log)} / {total}")
    if results_log:
        print(f"\n  {'Rank':<5} {'Score':>7}  {'Return':>8}  {'Sharpe':>7}  {'Trades':>7}  Indicators")
        print(f"  {'─'*65}")
        for rank, (sc_i, combo_i, st_i) in enumerate(results_log[:5], 1):
            try:
                r  = float(st_i['Return [%]'])
                sh = float(st_i.get('Sharpe Ratio', 0) or 0)
                tr = int(st_i['# Trades'])
                nm = '+'.join(c.replace('use_','') for c in combo_i)
                print(f"  #{rank:<4} {sc_i:>7.3f}  {r:>7.1f}%  {sh:>7.3f}  {tr:>7d}  {nm}")
            except Exception:
                pass
    print(f"{'─'*70}")

    return (results_log[0][1] if results_log else None,
            results_log[0][2] if results_log else None,
            total_time,
            results_log[:5],
            reinvest_rate)


# ============================================================================
# STAGE 3 — BAYESIAN PARAMETER FINE-TUNING
# ============================================================================

def run_bayesian_finetune(df, cash, commission_frac, best_combo, best_stats,
                           all_indicators, warmup_bars=0, n_calls=150,
                           signal_mode=2):
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
    except ImportError:
        print(f"\n{'!'*70}")
        print(f"  ⚠️  STAGE 3 SKIPPED — scikit-optimize NOT INSTALLED")
        print(f"  Bayesian fine-tuning did not run.")
        print(f"  Results below are Stage 2 combo-search params, UNREFINED.")
        print(f"  Fix: pip install scikit-optimize")
        print(f"{'!'*70}\n")

        indicator_grid  = get_indicator_grid()
        fallback_params = extract_best_params(best_stats, best_combo, indicator_grid)

        # Match the real Bayesian path's return contract — best_params there
        # also includes rr_ratio/atr_sl_mult/max_hold_bars (they're optimized
        # dimensions too), so pull the Stage-2 values for those here as well,
        # otherwise downstream code (walk-forward test) silently falls back
        # to un-tuned class defaults instead of Stage 2's actual best values.
        try:
            strategy_cls = best_stats._strategy
            for p in ('rr_ratio', 'atr_sl_mult', 'max_hold_bars'):
                if hasattr(strategy_cls, p):
                    fallback_params[p] = getattr(strategy_cls, p)
        except AttributeError:
            pass

        fallback_kwargs = {ind: (ind in best_combo) for ind in all_indicators}
        fallback_kwargs.update(fallback_params)
        fallback_kwargs.update({
            'signal_mode': signal_mode, 'warmup_bars': warmup_bars,
            'use_atr': True, 'use_slippage': True,
            'risk_pct': 0.01, 'atr_period': 14,
        })
        return best_stats, fallback_params, fallback_kwargs

    indicator_grid = get_indicator_grid()

    print(f"\n{'='*70}")
    print(f"  STAGE 3 — BAYESIAN PARAMETER FINE-TUNE")
    print(f"  Combo  : {'+'.join(c.replace('use_','') for c in best_combo)}")
    print(f"  Calls  : {n_calls}")
    print(f"{'='*70}\n")

    param_space  = []
    param_names  = []

    def _widen(v_min, v_max, factor=0.3):
        """Expand [v_min, v_max] outward by `factor` on each side, sign-safe."""
        span = v_max - v_min
        pad  = span * factor if span > 0 else abs(v_min) * factor + 1e-6
        return v_min - pad, v_max + pad

    for ind in best_combo:
        for param, vals in indicator_grid[ind].items():
            v_min, v_max = min(vals), max(vals)
            lo, hi = _widen(v_min, v_max)
            if all(isinstance(v, int) for v in vals):
                lo = int(round(lo))
                hi = int(round(hi))
                if lo >= hi:
                    hi = lo + 1
                if param.endswith('_period') or 'period' in param:
                    lo = max(1, lo)   # periods can't go to 0 or negative
                param_space.append(Integer(lo, hi, name=param))
            else:
                if lo >= hi:
                    hi = lo + 1e-3
                param_space.append(Real(lo, hi, name=param))
            param_names.append(param)

    param_space += [Real(1.5, 3.5, name='rr_ratio'),
                    Real(0.5, 2.5, name='atr_sl_mult'),
                    Integer(5, 20, name='max_hold_bars')]
    param_names += ['rr_ratio', 'atr_sl_mult', 'max_hold_bars']

    stage2_params = extract_best_params(best_stats, best_combo, indicator_grid)
    x0 = []
    for name in param_names:
        sp  = next(s for s in param_space if s.name == name)
        val = stage2_params.get(name, (sp.bounds[0] + sp.bounds[1]) / 2)
        val = max(sp.bounds[0], min(sp.bounds[1], val))
        x0.append(val)

    print(f"  {len(param_space)} search dimensions. Starting from Stage 2 best params.")
    eval_count       = [0]
    best_seen        = [score_stats(best_stats)]
    best_params_seen = [dict(stage2_params)]
    start_t          = time.time()

    def objective(params):
        eval_count[0] += 1
        run_kwargs = {ind: (ind in best_combo) for ind in all_indicators}
        run_kwargs.update({
            'signal_mode': signal_mode, 'warmup_bars': warmup_bars,
            'use_atr': True, 'use_slippage': True,
            'risk_pct': 0.01, 'atr_period': 14,
        })
        for name, val in zip(param_names, params):
            sp = next(s for s in param_space if s.name == name)
            run_kwargs[name] = int(round(val)) if isinstance(sp, Integer) else float(val)
        try:
            bt    = Backtest(df, MultiIndicatorStrategy,
                             cash=cash, commission=commission_frac,
                             exclusive_orders=True)
            stats = bt.run(**run_kwargs)
            sc    = score_stats(stats)
            n     = eval_count[0]
            el    = _format_time(time.time() - start_t)
            if sc > best_seen[0]:
                best_seen[0] = sc
                best_params_seen[0] = {
                    name: int(round(val)) if isinstance(sp, Integer) else float(val)
                    for name, val, sp in zip(param_names, params, param_space)
                }
                ret    = float(stats.get('Return [%]',    0) or 0)
                trades = int(stats.get('# Trades',        0) or 0)
                sharpe = float(stats.get('Sharpe Ratio',  0) or 0)
                print(f"  [{n:3d}/{n_calls}]  ★ NEW BEST  "
                      f"Score:{sc:.4f}  Ret:{ret:.2f}%  "
                      f"Trades:{trades}  Sh:{sharpe:.3f}  El:{el}", flush=True)
            elif n % 15 == 0:
                print(f"  [{n:3d}/{n_calls}]  Score:{sc:.4f}  "
                      f"Best:{best_seen[0]:.4f}  El:{el}", flush=True)
            return -sc
        except Exception:
            return 0.0

    try:
        result = gp_minimize(
            func=objective, dimensions=param_space,
            n_calls=n_calls, n_initial_points=max(10, n_calls // 5),
            x0=[x0], y0=[-score_stats(best_stats)],
            acq_func='EI', noise=0.01, random_state=42,
        )
    except KeyboardInterrupt:
        print(f"\n  Stage 3 interrupted — checkpoint saved.")
        raise

    best_params = {}
    for name, val in zip(param_names, result.x):
        sp = next(s for s in param_space if s.name == name)
        best_params[name] = int(round(val)) if isinstance(sp, Integer) else float(val)

    final_kwargs = {ind: (ind in best_combo) for ind in all_indicators}
    final_kwargs.update(best_params)
    final_kwargs.update({
        'signal_mode': signal_mode, 'warmup_bars': warmup_bars,
        'use_atr': True, 'use_slippage': True,
        'risk_pct': 0.01, 'atr_period': 14,
    })
    bt          = Backtest(df, MultiIndicatorStrategy,
                           cash=cash, commission=commission_frac,
                           exclusive_orders=True)
    final_stats = bt.run(**final_kwargs)
    final_sc    = score_stats(final_stats)
    stage2_sc   = score_stats(best_stats)
    improvement = ((final_sc - stage2_sc) / abs(stage2_sc) * 100 if stage2_sc != 0 else 0)

    print(f"\n{'─'*70}")
    print(f"  STAGE 3 COMPLETE")
    print(f"  Stage 2 score : {stage2_sc:.4f}")
    print(f"  Stage 3 score : {final_sc:.4f}  ({improvement:+.1f}%)")
    for k, v in sorted(best_params.items()):
        s2  = stage2_params.get(k, '?')
        tag = '  ← changed' if str(v) != str(s2) else ''
        print(f"    {k:<25} {v}  (was {s2}){tag}")
    print(f"{'─'*70}")
    return final_stats, best_params, final_kwargs


# ============================================================================
# STAGE 4 — MONTE CARLO VALIDATION
# FIX-3 (retained): bootstrap resampling with replacement
# ============================================================================

def run_monte_carlo(stats, cash=100_000, n_sims=1000):
    print(f"\n{'='*70}")
    print(f"  STAGE 4 — MONTE CARLO VALIDATION  ({n_sims} simulations)")
    print(f"  Method : Bootstrap resampling with replacement (FIX-3)")
    print(f"{'='*70}\n")

    try:
        trades_df = stats._trades
        if trades_df is None or len(trades_df) < 10:
            n = len(trades_df) if trades_df is not None else 0
            print(f"  Skipped — need >= 10 trades (got {n})")
            return None

        returns = None
        for col in ('ReturnPct', 'Return [%]', 'ReturnPct [%]'):
            if col in trades_df.columns:
                raw     = trades_df[col].values.astype(float)
                returns = raw / 100.0 if abs(np.nanmean(raw)) > 1.0 else raw
                break

        if returns is None and 'PnL' in trades_df.columns and 'Size' in trades_df.columns:
            try:
                entry_val = trades_df['Size'].values * trades_df['EntryPrice'].values
                returns   = (trades_df['PnL'].values / np.where(entry_val == 0, 1, entry_val)).astype(float)
            except Exception:
                pass

        if returns is None:
            print(f"  Skipped — cannot extract per-trade returns.")
            print(f"  Available columns: {list(trades_df.columns)}")
            return None

        returns = returns[~np.isnan(returns)]
        if len(returns) < 10:
            print(f"  Skipped — too few valid returns ({len(returns)}).")
            return None

    except Exception as e:
        print(f"  Skipped — error reading trades: {e}")
        return None

    n_trades     = len(returns)
    original_ret = float(stats.get('Return [%]', 0) or 0)
    original_dd  = float(stats.get('Max. Drawdown [%]', 0) or 0)

    print(f"  Trades       : {n_trades}")
    print(f"  Backtest ret : {original_ret:.2f}%")
    print(f"  Backtest DD  : {original_dd:.2f}%\n")
    print(f"  Running {n_sims} bootstrap simulations...", flush=True)

    rng           = np.random.default_rng(42)
    final_returns = np.empty(n_sims)
    max_drawdowns = np.empty(n_sims)

    for sim in range(n_sims):
        # FIX-3: resample with replacement
        sample = rng.choice(returns, size=n_trades, replace=True)
        equity = cash * np.cumprod(1.0 + sample)
        equity = np.insert(equity, 0, cash)
        final_returns[sim] = (equity[-1] - cash) / cash * 100.0
        peak               = np.maximum.accumulate(equity)
        max_drawdowns[sim] = np.min((equity - peak) / peak * 100.0)

    pctls = [5, 25, 50, 75, 95]
    ret_p = {p: np.percentile(final_returns, p) for p in pctls}
    dd_p  = {p: np.percentile(max_drawdowns, p) for p in pctls}
    ruin  = np.mean(final_returns < -30.0) * 100.0

    print(f"\n  {'Metric':<22} {'P5':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'P95':>8}")
    print(f"  {'─'*60}")
    print(f"  {'Return [%]':<22} "
          f"{ret_p[5]:>7.1f}% {ret_p[25]:>7.1f}% "
          f"{ret_p[50]:>7.1f}% {ret_p[75]:>7.1f}% {ret_p[95]:>7.1f}%")
    print(f"  {'Max Drawdown [%]':<22} "
          f"{dd_p[5]:>7.1f}% {dd_p[25]:>7.1f}% "
          f"{dd_p[50]:>7.1f}% {dd_p[75]:>7.1f}% {dd_p[95]:>7.1f}%")

    print(f"\n  Actual backtest return: {original_ret:.2f}%")
    if   original_ret > ret_p[75]: verdict = "⚠  LUCKY — above P75 (may be sequence-dependent)"
    elif original_ret < ret_p[25]: verdict = "✓  CONSERVATIVE — below P25 (understated)"
    else:                           verdict = "✓  ROBUST — within P25-P75 normal range"
    print(f"  {verdict}")

    print(f"\n  Ruin probability (>30% loss): {ruin:.1f}%")
    if   ruin > 10: print("  ⚠  HIGH RUIN RISK — reduce position size")
    elif ruin > 5:  print("  ⚠  ELEVATED — consider reducing position size")
    else:           print("  ✓  Ruin risk acceptable")

    print(f"\n  Live trading expectations (bootstrap):")
    print(f"    Expected return (P50)  : {ret_p[50]:>7.1f}%")
    print(f"    Worst-case return (P5) : {ret_p[5]:>7.1f}%")
    print(f"    Worst-case DD (P5)     : {dd_p[5]:>7.1f}%")
    print(f"{'='*70}\n")

    return {'final_returns': final_returns, 'max_drawdowns': max_drawdowns,
            'ret_percentiles': ret_p, 'dd_percentiles': dd_p,
            'ruin_prob': ruin, 'n_trades': n_trades, 'n_sims': n_sims}


# ============================================================================
# DATA LOADER
# ============================================================================

def load_data(csv_file):
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    df.columns = [c.strip().lower() for c in df.columns]
    needed = ['open', 'high', 'low', 'close', 'volume']
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Found: {list(df.columns)}")
    df = df[needed].copy()
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df.dropna().astype(float)
    df = df[(df > 0).all(axis=1)]
    return df


def choose_data_range(df):
    WARMUP_BARS = 200
    total_rows  = len(df)
    total_range = f"{df.index[0].date()} → {df.index[-1].date()}"

    # 2 year cutoff
    cutoff_2y     = df.index[-1] - pd.DateOffset(years=2)
    cutoff_2y_pos = df.index.searchsorted(cutoff_2y)

    # 6+3 split
    cutoff_9y     = df.index[-1] - pd.DateOffset(years=9)
    cutoff_9y_pos = df.index.searchsorted(cutoff_9y)
    cutoff_3y     = df.index[-1] - pd.DateOffset(years=3)
    cutoff_3y_pos = df.index.searchsorted(cutoff_3y)

    warmup_2y = max(0, cutoff_2y_pos - WARMUP_BARS)
    warmup_9y = max(0, cutoff_9y_pos - WARMUP_BARS)

    df_2yr   = df.iloc[warmup_2y:].copy()
    df_train = df.iloc[warmup_9y:cutoff_3y_pos].copy()  # 6yr train + warmup
    df_test  = df.iloc[cutoff_3y_pos:].copy()           # 3yr test

    print(f"\n  {'─'*70}")
    print(f"  DATA RANGE SELECTION")
    print(f"  {'─'*70}")
    print(f"\n  1. FULL DATA    {total_range}    ({total_rows} rows)")
    print(f"\n  2. LAST 2 YEARS + {WARMUP_BARS}-bar warmup")
    print(f"     Trading : {df.index[cutoff_2y_pos].date()} → {df.index[-1].date()}")
    print(f"\n  3. 6+3 WALK-FORWARD")
    print(f"     Train   : {df.index[cutoff_9y_pos].date()} → {df.index[cutoff_3y_pos].date()}  (6 years)")
    print(f"     Test    : {df.index[cutoff_3y_pos].date()} → {df.index[-1].date()}  (3 years)")
    print(f"     ⚠  Run pipeline on Train, then single-run on Test")

    while True:
        print(f"\n  Choose (1, 2 or 3): ", end='', flush=True)
        choice = input().strip()
        if choice == '1':
            print(f"\n  ✓ Full data — {total_rows} rows")
            return df, 0, None
        elif choice == '2':
            print(f"\n  ✓ 2-year mode")
            return df_2yr, WARMUP_BARS, None
        elif choice == '3':
            print(f"\n  ✓ Walk-forward mode — optimize on train, validate on test")
            return df_train, WARMUP_BARS, df_test
        print("  Invalid. Enter 1, 2 or 3.")


# ============================================================================
# GARCH SIZING SUPPORT  (Option 4 — GARCH Pipeline)
# ============================================================================

def load_garch_file(garch_file):
    """
    Reads a garch_risk.py output CSV (Date index, GarchVol column — a
    fractional daily-return volatility forecast, e.g. 0.018 = 1.8%, NOT a
    price distance). Column-name matching is case-insensitive since
    different runs/tools may write it slightly differently.
    """
    df = pd.read_csv(garch_file, index_col=0, parse_dates=True)
    df.columns = [c.strip() for c in df.columns]
    col = next((c for c in df.columns if c.lower() == 'garchvol'), None)
    if col is None:
        raise ValueError(f"'GarchVol' column not found in {garch_file}. "
                          f"Found: {list(df.columns)}")
    return df[[col]].rename(columns={col: 'GarchVol'})


def merge_garch(df, garch_file):
    """
    Left-joins GarchVol onto the OHLCV df by date, forward-filling any
    gaps (e.g. GarchVol's own min_train burn-in period, or minor date
    mismatches) so every bar has a usable value once the forecast starts.
    Bars before the GARCH model's burn-in period will still be NaN —
    intentional, not filled — the strategy skips GARCH-sized entries on
    those bars rather than pretending a forecast exists.
    """
    garch = load_garch_file(garch_file)
    merged = df.join(garch, how='left')
    merged['GarchVol'] = merged['GarchVol'].ffill()
    return merged


def load_alpha_file(alpha_file):
    """
    Reads alpha_pipeline.py's output CSV (Date index, Regime / MasterPred /
    PredVol columns). MasterPred is a price-level forecast (Y-hat_{t+1}),
    PredVol is a fractional daily-return volatility forecast — same units
    and convention as GarchVol, NOT a price distance.
    """
    df = pd.read_csv(alpha_file, index_col=0, parse_dates=True)
    df.columns = [c.strip() for c in df.columns]
    needed = {}
    for want in ('Regime', 'MasterPred', 'PredVol'):
        col = next((c for c in df.columns if c.lower() == want.lower()), None)
        if col is None:
            raise ValueError(f"'{want}' column not found in {alpha_file}. "
                              f"Found: {list(df.columns)}")
        needed[want] = col
    return df[[needed['Regime'], needed['MasterPred'], needed['PredVol']]].rename(
        columns={needed['Regime']: 'Regime',
                 needed['MasterPred']: 'MasterPred',
                 needed['PredVol']: 'PredVol'})


def merge_alpha(df, alpha_file):
    """
    Left-joins Regime/MasterPred/PredVol onto the OHLCV df by date,
    forward-filling minor date-alignment gaps — same pattern as
    merge_garch(). Bars before alpha_pipeline.py's burn-in period stay
    NaN intentionally; the strategy skips alpha-gated/alpha-sized entries
    on those bars rather than pretending a forecast exists.
    """
    alpha = load_alpha_file(alpha_file)
    merged = df.join(alpha, how='left')
    for col in ('Regime', 'MasterPred', 'PredVol'):
        merged[col] = merged[col].ffill()
    return merged


# ============================================================================
# STRATEGY
#
# FIX-1 (retained): Signal fires on bar N close → pending flag set.
#                   Execution happens at bar N+1 open.
#
# FIX-4 (new):      _pending_sl_dist and _pending_tp_dist store the raw
#                   ATR-derived DISTANCES (in price units), not absolute
#                   price levels.  At execution time on bar N+1, absolute
#                   SL and TP are computed from assumed_fill:
#                       sl = assumed_fill - sl_dist
#                       tp = assumed_fill + tp_dist
#                   This means SL/TP are always the correct distance from
#                   the actual fill regardless of overnight gaps or slippage.
#
# FIX-5 (new):      Pending flags are pure instance state (set in init only).
#                   No class-level defaults that could leak across optimiser runs.
#
# SIZING-1 (new):   IMPORTANT DISCOVERY — the original next() called
#                   self.buy(sl=sl, tp=tp) with NO size= argument, so every
#                   trade used backtesting.py's default (near-full-equity)
#                   sizing. risk_pct was defined and printed but never
#                   actually used to compute a position size. sizing_mode
#                   now controls this explicitly:
#                     'legacy' — unchanged behavior (full-equity sizing,
#                                ATR-based stop placement only)
#                     'atr'    — NEW: real risk_pct-based sizing, ATR drives
#                                both the stop distance and the size
#                     'garch'  — NEW (Option 4): GarchVol drives both the
#                                stop distance and the size, in place of ATR
# ============================================================================

class MultiIndicatorStrategy(Strategy):
    use_rsi = True;        rsi_period = 14;    rsi_lower = 30.0;   rsi_upper = 70.0
    use_macd = False;      macd_fast = 12;     macd_slow = 26;     macd_signal_p = 9
    use_bb = False;        bb_period = 20;     bb_std = 2.0
    use_ema = False;       ema_fast = 9;       ema_slow = 21
    use_sma = False;       sma_fast = 20;      sma_slow = 50
    use_stoch = False;     stoch_k = 14;       stoch_d = 3;        stoch_lower = 20.0; stoch_upper = 80.0
    use_cci = False;       cci_period = 20;    cci_lower = -100.0; cci_upper = 100.0
    use_willr = False;     willr_period = 14;  willr_lower = -80.0; willr_upper = -20.0
    use_supertrend = False; st_period = 10;    st_factor = 3.0
    use_ichimoku = False;  ichi_tenkan = 9;    ichi_kijun = 26;    ichi_senkou_b = 52
    use_adx = False;       adx_period = 14;    adx_threshold = 25.0
    use_multi_ema = False; mema_fast = 12;     mema_slow = 100
    use_atr = False;       atr_period = 14;    atr_sl_mult = 1.5
    use_keltner = False;   kc_period = 20;     kc_mult = 1.5
    use_donchian = False;  dc_period = 20;     dc_tol = 0.5
    use_stddev = False;    sd_period = 20;     sd_mult = 2.0
    use_pivot = False
    use_fib = False;       fib_lookback = 5;   fib_tol = 0.1;  fib_min_prominence = 0.5
    fib_min_lb = 3;        fib_max_lb = 20
    use_sr = False;        sr_lookback = 50;   sr_tol = 0.1
    use_mfi = False;       mfi_period = 14;    mfi_lower = 20.0;  mfi_upper = 80.0
    use_vwap = False
    use_obv = False;       obv_lookback = 20
    use_cmf = False;       cmf_period = 20
    use_vol_spike = False; vs_mult = 2.0;      vs_period = 20
    use_slippage = True;   slip_atr_mult = 0.10; slip_max_pct = 0.005
    rr_ratio = 2.0;        risk_pct = 0.01;    max_hold_bars = 15
    signal_mode = 2;       warmup_bars = 0

    # SIZING-1: sizing_mode controls stop-distance source AND position size.
    #   'legacy' = original behavior (ATR stop, full-equity size)
    #   'atr'    = ATR stop + risk_pct-based size (NEW real baseline)
    #   'garch'  = GarchVol stop + risk_pct-based size (Option 4)
    #   'alpha'  = PredVol (from alpha_pipeline.py) stop + risk_pct-based
    #              size — reuses garch_sl_mult as the multiplier since both
    #              are fractional-vol-forecast-driven stops on the same
    #              conceptual footing (kept comparable via --dual-compare)
    sizing_mode   = 'legacy'
    garch_sl_mult = 1.5

    # ALPHA PIPELINE: direction_mode controls whether the HMM->SARIMAX->
    # LSTM->GARCH pipeline's predicted direction (MasterPred vs close)
    # influences trade entries, independent of sizing_mode:
    #   'off'     = predicted direction ignored entirely (sizing-only, if
    #               sizing_mode='alpha' is also set)
    #   'gate'    = an indicator-signaled buy is only taken if MasterPred
    #               also predicts a higher close next bar; if MasterPred
    #               is NaN (burn-in) the trade is blocked, not allowed
    #               through ungated
    #   'replace' = indicator votes are ignored for ENTRIES; a buy is
    #               taken purely because MasterPred predicts a higher
    #               close. Exits still use indicator sell-votes + the
    #               existing TP/SL/max_hold_bars machinery — direction_mode
    #               only ever governs entries, never exits, to avoid
    #               overriding risk controls that are already working
    direction_mode = 'off'

    def init(self):
        close  = np.array(self.data.Close)
        high   = np.array(self.data.High)
        low    = np.array(self.data.Low)
        volume = np.array(self.data.Volume)
        n      = len(close)

        # FIX-4 + FIX-5: all pending state as instance variables, distances not levels
        self._entry_bar       = -1

        if self.use_rsi:
            self._rsi = talib.RSI(close, timeperiod=int(self.rsi_period))
        if self.use_macd:
            self._macd_line, self._macd_sig, _ = talib.MACD(
                close, fastperiod=int(self.macd_fast),
                slowperiod=int(self.macd_slow), signalperiod=int(self.macd_signal_p))
        if self.use_bb:
            self._bb_up, _, self._bb_lo = talib.BBANDS(
                close, timeperiod=int(self.bb_period),
                nbdevup=float(self.bb_std), nbdevdn=float(self.bb_std))
        if self.use_ema:
            self._ema_f = talib.EMA(close, timeperiod=int(self.ema_fast))
            self._ema_s = talib.EMA(close, timeperiod=int(self.ema_slow))
        if self.use_sma:
            self._sma_f = talib.SMA(close, timeperiod=int(self.sma_fast))
            self._sma_s = talib.SMA(close, timeperiod=int(self.sma_slow))
        if self.use_stoch:
            self._sk, _ = talib.STOCH(high, low, close,
                fastk_period=int(self.stoch_k),
                slowk_period=int(self.stoch_d), slowd_period=int(self.stoch_d))
        if self.use_cci:
            self._cci_vals = talib.CCI(high, low, close, timeperiod=int(self.cci_period))
        if self.use_willr:
            self._willr_vals = talib.WILLR(high, low, close, timeperiod=int(self.willr_period))
        if self.use_atr:
            self._atr_vals = talib.ATR(high, low, close, timeperiod=int(self.atr_period))

        if self.use_fib:
            fib_max   = int(self.fib_max_lb)
            cache_key = (int(self.fib_lookback), float(self.fib_min_prominence),
                         int(self.fib_min_lb), int(self.fib_max_lb))
            if not hasattr(MultiIndicatorStrategy, '_swing_cache'):
                MultiIndicatorStrategy._swing_cache = {}
            if cache_key not in MultiIndicatorStrategy._swing_cache:
                MultiIndicatorStrategy._swing_cache[cache_key] = \
                    find_swing_points_enhanced(
                        high, low, close,
                        base_lookback=int(self.fib_lookback),
                        min_prominence_atr=float(self.fib_min_prominence),
                        min_lb=int(self.fib_min_lb), max_lb=int(self.fib_max_lb))
            self._swing_highs, self._swing_lows, self._swing_quality = \
                MultiIndicatorStrategy._swing_cache[cache_key]
            min_prom   = float(self.fib_min_prominence)
            tol_pct    = float(self.fib_tol) * 0.01
            fib_levels = [0.382, 0.500, 0.618]
            self._fib_buy       = np.zeros(n, dtype=bool)
            self._fib_sell      = np.zeros(n, dtype=bool)
            self._fib_level_hit = np.full(n, np.nan)
            for i in range(fib_max * 2, n):
                sh, sh_idx, sl, sl_idx = find_last_swing_pair_quality(
                    self._swing_highs, self._swing_lows, self._swing_quality,
                    i, min_quality=min_prom)
                if np.isnan(sh) or np.isnan(sl): continue
                rng = sh - sl
                if rng == 0: continue
                c  = close[i]
                ta = tol_pct * c
                if sl_idx < sh_idx:
                    for lvl in fib_levels:
                        if abs(c - (sh - lvl * rng)) <= ta:
                            self._fib_buy[i] = True
                            self._fib_level_hit[i] = lvl; break
                elif sh_idx < sl_idx:
                    for lvl in fib_levels:
                        if abs(c - (sl + lvl * rng)) <= ta:
                            self._fib_sell[i] = True
                            self._fib_level_hit[i] = lvl; break

        if self.use_pivot:
            self._piv    = np.full(n, np.nan)
            self._piv_r1 = np.full(n, np.nan)
            self._piv_s1 = np.full(n, np.nan)
            for i in range(1, n):
                ph = high[i-1]; pl = low[i-1]; pc = close[i-1]
                p = (ph + pl + pc) / 3.0
                self._piv[i]    = p
                self._piv_r1[i] = 2*p - pl
                self._piv_s1[i] = 2*p - ph

        if self.use_supertrend:
            self._st_line, self._st_dir = compute_supertrend(
                high, low, close, period=int(self.st_period), factor=float(self.st_factor))
        if self.use_ichimoku:
            self._ichi_t, self._ichi_k, self._ichi_sa, self._ichi_sb = compute_ichimoku(
                high, low, close, tenkan_p=int(self.ichi_tenkan),
                kijun_p=int(self.ichi_kijun), senkou_b_p=int(self.ichi_senkou_b))
        if self.use_adx:
            self._adx_vals = talib.ADX(high, low, close, timeperiod=int(self.adx_period))
            self._pdi      = talib.PLUS_DI(high, low, close, timeperiod=int(self.adx_period))
            self._mdi      = talib.MINUS_DI(high, low, close, timeperiod=int(self.adx_period))
        if self.use_multi_ema:
            self._mema_f = talib.EMA(close, timeperiod=int(self.mema_fast))
            self._mema_s = talib.EMA(close, timeperiod=int(self.mema_slow))
        if self.use_keltner:
            kc_ema = talib.EMA(close, timeperiod=int(self.kc_period))
            kc_atr = talib.ATR(high, low, close, timeperiod=int(self.kc_period))
            m = float(self.kc_mult)
            self._kc_upper = kc_ema + m * kc_atr
            self._kc_lower = kc_ema - m * kc_atr
        if self.use_donchian:
            p = int(self.dc_period)
            self._dc_upper = np.array(pd.Series(high).rolling(p).max())
            self._dc_lower = np.array(pd.Series(low).rolling(p).min())
        if self.use_stddev:
            sd_ma  = talib.SMA(close, timeperiod=int(self.sd_period))
            sd_std = talib.STDDEV(close, timeperiod=int(self.sd_period))
            m = float(self.sd_mult)
            self._sd_upper = sd_ma + m * sd_std
            self._sd_lower = sd_ma - m * sd_std
        if self.use_sr:
            lb = int(self.sr_lookback)
            self._sr_res = pd.Series(high).rolling(lb).max().values
            self._sr_sup = pd.Series(low).rolling(lb).min().values
        if self.use_mfi:
            self._mfi_vals = talib.MFI(high, low, close, volume,
                                       timeperiod=int(self.mfi_period))
        if self.use_vwap:
            self._vwap = compute_vwap(high, low, close, volume)
        if self.use_obv:
            self._obv_signal = compute_obv_divergence(
                close, volume, lookback=int(self.obv_lookback))
        if self.use_cmf:
            p   = int(self.cmf_period)
            mfv = volume * ((close - low) - (high - close)) / np.where(
                (high - low) == 0, 1e-9, (high - low))
            self._cmf_vals = (pd.Series(mfv).rolling(p).sum() /
                              pd.Series(volume).rolling(p).sum()).values
        if self.use_vol_spike:
            p      = int(self.vs_period)
            vol_ma = pd.Series(volume).rolling(p).mean().values
            with np.errstate(invalid='ignore', divide='ignore'):
                self._vs_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)

        self._volume   = volume
        self._vol_ma20 = pd.Series(volume).rolling(20).mean().values
        self._slip_atr = talib.ATR(high, low, close, timeperiod=14)

        # GARCH sizing support: GarchVol comes in as an extra column on the
        # source df (merged via merge_garch() before Backtest() is built),
        # so backtesting.py exposes it as self.data.GarchVol automatically
        # if present. Cache as a plain array here for fast indexed access,
        # same pattern as every other indicator array above.
        if hasattr(self.data, 'GarchVol'):
            self._garch_vol = np.array(self.data.GarchVol)
        else:
            self._garch_vol = None

        # Alpha pipeline (HMM->SARIMAX->LSTM->GARCH) support: same pattern
        # as GarchVol above — columns merged onto df before Backtest() is
        # built (see merge_alpha()), so backtesting.py exposes them
        # automatically as self.data.<Col> if present.
        if hasattr(self.data, 'MasterPred'):
            self._master_pred = np.array(self.data.MasterPred)
        else:
            self._master_pred = None
        if hasattr(self.data, 'PredVol'):
            self._pred_vol = np.array(self.data.PredVol)
        else:
            self._pred_vol = None

    def _risk_based_size(self, sl_dist, close_price):
        """
        SIZING-1: real risk_pct-based position sizing, used by both 'atr'
        and 'garch' sizing_mode.
            risk_amount = current_equity * risk_pct
            raw_units   = risk_amount / sl_dist
        Capped so the position cost never exceeds available equity, then
        floored to a whole share (Indian equities trade in whole shares —
        no fractional-unit or lot-step handling needed here, unlike
        XAUUSD's contract-size/lot-step case).
        Returns 0 if risk_pct at current equity can't even afford 1 share
        at this stop distance — the trade is skipped rather than forced
        through at a size that doesn't reflect the intended risk.
        """
        if sl_dist is None or sl_dist <= 0 or np.isnan(sl_dist) or close_price <= 0:
            return 0
        equity       = self.equity
        risk_amount  = equity * float(self.risk_pct)
        raw_units    = risk_amount / sl_dist
        max_afford   = equity / close_price
        units        = min(raw_units, max_afford)
        return int(units)  # floor to whole shares

    def next(self):
        idx = len(self.data) - 1
        if idx < int(self.warmup_bars):
            return

        close_price = float(self.data.Close[-1])

        # ── Max hold bar exit ─────────────────────────────────────────────
        if self.position:
            try:
                if idx - int(self._entry_bar) >= int(self.max_hold_bars):
                    self.position.close()
                    return
            except Exception:
                pass

        # ── Compute indicator votes ───────────────────────────────────────
        buy_votes = sell_votes = active = 0

        def _v(arr):
            try:
                val = float(arr[idx])
                return None if np.isnan(val) else val
            except Exception:
                return None

        if self.use_rsi:
            v = _v(self._rsi)
            if v is not None:
                active += 1
                if   v < self.rsi_lower: buy_votes  += 1
                elif v > self.rsi_upper: sell_votes += 1

        if self.use_macd:
            m, s = _v(self._macd_line), _v(self._macd_sig)
            if m is not None and s is not None:
                active += 1
                if   m > s: buy_votes  += 1
                elif m < s: sell_votes += 1

        if self.use_bb:
            bu, bl = _v(self._bb_up), _v(self._bb_lo)
            if bu is not None and bl is not None:
                active += 1
                if   close_price < bl: buy_votes  += 1
                elif close_price > bu: sell_votes += 1

        if self.use_ema:
            ef, es = _v(self._ema_f), _v(self._ema_s)
            if ef is not None and es is not None:
                active += 1
                if   ef > es: buy_votes  += 1
                elif ef < es: sell_votes += 1

        if self.use_sma:
            sf, ss = _v(self._sma_f), _v(self._sma_s)
            if sf is not None and ss is not None:
                active += 1
                if   sf > ss: buy_votes  += 1
                elif sf < ss: sell_votes += 1

        if self.use_stoch:
            k = _v(self._sk)
            if k is not None:
                active += 1
                if   k < self.stoch_lower: buy_votes  += 1
                elif k > self.stoch_upper: sell_votes += 1

        if self.use_cci:
            v = _v(self._cci_vals)
            if v is not None:
                active += 1
                if   v < self.cci_lower: buy_votes  += 1
                elif v > self.cci_upper: sell_votes += 1

        if self.use_willr:
            v = _v(self._willr_vals)
            if v is not None:
                active += 1
                if   v < self.willr_lower: buy_votes  += 1
                elif v > self.willr_upper: sell_votes += 1

        if self.use_fib:
            active += 1
            if   self._fib_buy[idx]:  buy_votes  += 1
            elif self._fib_sell[idx]: sell_votes += 1

        if self.use_pivot:
            p, r1, s1 = self._piv[idx], self._piv_r1[idx], self._piv_s1[idx]
            if not np.isnan(p):
                active += 1
                if   not np.isnan(s1) and close_price <= s1: buy_votes  += 1
                elif not np.isnan(r1) and close_price >= r1: sell_votes += 1

        if self.use_supertrend:
            d = self._st_dir[idx]
            if d != 0:
                active += 1
                if   d ==  1: buy_votes  += 1
                elif d == -1: sell_votes += 1

        if self.use_ichimoku:
            tk = _v(self._ichi_t); kj = _v(self._ichi_k)
            sa = _v(self._ichi_sa); sb = _v(self._ichi_sb)
            if all(x is not None for x in [tk, kj, sa, sb]):
                active += 1
                cloud_top = max(sa, sb); cloud_bot = min(sa, sb)
                if   close_price > cloud_top and tk > kj: buy_votes  += 1
                elif close_price < cloud_bot and tk < kj: sell_votes += 1

        if self.use_adx:
            av = _v(self._adx_vals); pdi = _v(self._pdi); mdi = _v(self._mdi)
            if all(x is not None for x in [av, pdi, mdi]) and av >= self.adx_threshold:
                active += 1
                if   pdi > mdi: buy_votes  += 1
                elif mdi > pdi: sell_votes += 1

        if self.use_multi_ema:
            mf = _v(self._mema_f); ms = _v(self._mema_s)
            if mf is not None and ms is not None:
                active += 1
                if   mf > ms: buy_votes  += 1
                elif mf < ms: sell_votes += 1

        if self.use_keltner:
            ku = _v(self._kc_upper); kl = _v(self._kc_lower)
            if ku is not None and kl is not None:
                active += 1
                if   close_price < kl: buy_votes  += 1
                elif close_price > ku: sell_votes += 1

        if self.use_donchian:
            du = self._dc_upper[idx]; dl = self._dc_lower[idx]
            if not np.isnan(du) and not np.isnan(dl):
                ta = float(self.dc_tol) * 0.01 * close_price
                active += 1
                if   abs(close_price - dl) <= ta: buy_votes  += 1
                elif abs(close_price - du) <= ta: sell_votes += 1

        if self.use_stddev:
            su = _v(self._sd_upper); sl_v = _v(self._sd_lower)
            if su is not None and sl_v is not None:
                active += 1
                if   close_price < sl_v: buy_votes  += 1
                elif close_price > su:   sell_votes += 1

        if self.use_sr:
            res = self._sr_res[idx]; sup = self._sr_sup[idx]
            if not np.isnan(res) and not np.isnan(sup):
                tol = float(self.sr_tol) * 0.01
                active += 1
                if   abs(close_price - sup) <= tol * close_price: buy_votes  += 1
                elif abs(close_price - res) <= tol * close_price: sell_votes += 1

        if self.use_mfi:
            v = _v(self._mfi_vals)
            if v is not None:
                active += 1
                if   v < self.mfi_lower: buy_votes  += 1
                elif v > self.mfi_upper: sell_votes += 1

        if self.use_vwap:
            vw = _v(self._vwap)
            if vw is not None:
                active += 1
                if   close_price > vw: buy_votes  += 1
                elif close_price < vw: sell_votes += 1

        if self.use_obv:
            sig = self._obv_signal[idx]
            if sig != 0:
                active += 1
                if   sig ==  1: buy_votes  += 1
                elif sig == -1: sell_votes += 1

        if self.use_cmf:
            v = _v(self._cmf_vals)
            if v is not None:
                active += 1
                if   v >  0.05: buy_votes  += 1
                elif v < -0.05: sell_votes += 1

        if self.use_vol_spike:
            ratio = self._vs_ratio[idx]
            if not np.isnan(ratio) and ratio >= float(self.vs_mult):
                active += 1
                if   close_price > float(self.data.Open[-1]): buy_votes  += 1
                elif close_price < float(self.data.Open[-1]): sell_votes += 1

        if active == 0:
            return

        sm = int(self.signal_mode)
        if   sm == 0:
            do_buy  = buy_votes  > 0
            do_sell = sell_votes > 0
        elif sm == 1:
            do_buy  = (buy_votes == active and buy_votes > 0)
            do_sell = (sell_votes == active and sell_votes > 0)
        else:
            do_buy  = (buy_votes > sell_votes and buy_votes > 0)
            do_sell = (sell_votes > buy_votes and sell_votes > 0)

        # ── ALPHA PIPELINE: direction_mode gates/replaces ENTRIES only ────
        # (exits are untouched — see class docstring above for rationale)
        if self.direction_mode in ('gate', 'replace'):
            pred_up = None
            if self._master_pred is not None:
                mp = _v(self._master_pred)
                if mp is not None:
                    pred_up = mp > close_price

            if self.direction_mode == 'replace':
                # indicator votes irrelevant for entries — purely Y-hat driven
                do_buy = bool(pred_up)              # False if pred_up is None (NaN/burn-in)
            elif self.direction_mode == 'gate':
                # indicator vote must ALSO agree with Y-hat; no confirmation
                # available (NaN/burn-in) -> blocked, not allowed through
                do_buy = do_buy and (pred_up is True)

        # ── Execute: fill at next bar open via backtesting.py native model ─
        # Signal fires on bar N close. backtesting.py executes market orders
        # at bar N+1 open natively — no pending flag needed.
        if not self.position and do_buy and self.use_atr:
            atr = _v(self._atr_vals)
            if atr is None:
                return

            slip = 0.0
            if self.use_slippage:
                vol_val = float(self._volume[idx])
                avg_vol = float(self._vol_ma20[idx]) if not np.isnan(
                        self._vol_ma20[idx]) else vol_val
                slip = compute_slippage(float(atr), vol_val, avg_vol, close_price,
                                        atr_mult=float(self.slip_atr_mult),
                                        max_slip_pct=float(self.slip_max_pct))

            # SIZING-1: sl_dist source depends on sizing_mode. 'garch' uses
            # GarchVol for BOTH the stop distance and the position size —
            # never mix (ATR stop + GARCH size) or the real risk taken
            # would drift silently from the intended risk_pct.
            if self.sizing_mode == 'garch':
                if self._garch_vol is None:
                    return  # no GARCH data merged for this run — skip entry
                gv = _v(self._garch_vol)
                if gv is None:
                    return  # still in GarchVol's burn-in period — skip entry
                # GarchVol is a fractional daily-return forecast (e.g. 0.018),
                # NOT a price distance like ATR — convert explicitly.
                sl_dist = float(self.garch_sl_mult) * gv * close_price
            elif self.sizing_mode == 'alpha':
                if self._pred_vol is None:
                    return  # no alpha-pipeline data merged for this run — skip entry
                pv = _v(self._pred_vol)
                if pv is None:
                    return  # still in PredVol's burn-in period — skip entry
                # PredVol is a fractional-vol forecast, same convention as
                # GarchVol — reuses garch_sl_mult for direct comparability
                sl_dist = float(self.garch_sl_mult) * pv * close_price
            else:
                sl_dist = float(self.atr_sl_mult) * atr

            sl = close_price + slip - sl_dist
            tp = close_price + slip + sl_dist * float(self.rr_ratio)

            if self.sizing_mode == 'legacy':
                # unchanged original behavior — full-equity default sizing
                self._entry_bar = idx
                self.buy(sl=sl, tp=tp)
            else:
                size = self._risk_based_size(sl_dist, close_price)
                if size >= 1:
                    self._entry_bar = idx
                    self.buy(size=size, sl=sl, tp=tp)
                # size == 0 -> risk_pct too small to afford 1 share at this
                # stop distance/equity level — trade skipped, not forced.

        elif self.position and do_sell:
            self.position.close()


# ============================================================================
# HELPERS
# ============================================================================

def extract_best_params(stats, combo, indicator_grid):
    try:
        strategy_cls = stats._strategy
    except AttributeError:
        return {}
    all_param_names = set()
    for ind in combo:
        all_param_names.update(indicator_grid[ind].keys())
    return {p: getattr(strategy_cls, p)
            for p in all_param_names if hasattr(strategy_cls, p)}


def _count_full_combos(sizes):
    from math import comb
    n = len(get_indicator_grid())
    return sum(comb(n, r) for r in sizes)


def display_results(combo, stats, total_time, top_results, indicator_grid,
                    reinvest_rate=0.50, cash=100_000):
    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION COMPLETE  ({_format_time(total_time)})")
    print(f"{'='*70}")

    if stats is None:
        print(f"\n  No valid results found.")
        return

    sc   = score_stats(stats)
    comp = compute_compounded_stats(stats, cash=cash, reinvest_rate=reinvest_rate)

    print(f"\n  🏆  BEST COMBO: {[c.replace('use_','') for c in combo]}")
    print(f"  Score: {sc:.4f}\n")
    print(f"  {'METRIC':<28} {'NORMAL':>12}  {'COMPOUNDED':>12}")
    print(f"  {'─'*55}")
    print(f"  {'Return [%]':<28} {float(stats.get('Return [%]',0)):>11.2f}%  "
          f"{comp.get('compounded_return_pct',0):>11.2f}%")
    print(f"  {'Sharpe Ratio':<28} {float(stats.get('Sharpe Ratio',0) or 0):>12.3f}  "
          f"{comp.get('compounded_sharpe',0):>12.3f}  ← annualised sqrt(252)")
    print(f"  {'Max Drawdown [%]':<28} {float(stats.get('Max. Drawdown [%]',0)):>11.2f}%  "
          f"{comp.get('compounded_max_dd',0):>11.2f}%")
    print(f"  {'Equity Final [$]':<28} ${float(stats.get('Equity Final [$]',0)):>10,.0f}  "
          f"${comp.get('compounded_final',0):>10,.0f}")
    print(f"  {'─'*55}")

    metrics = [
        ('# Trades',             'Trades',       '{:d}'),
        ('Win Rate [%]',         'Win Rate',     '{:.1f}%'),
        ('Profit Factor',        'Profit Factor','{:.2f}'),
        ('Buy & Hold Return [%]','B&H Return',   '{:.2f}%'),
        ('Sortino Ratio',        'Sortino',      '{:.3f}'),
        ('Avg. Trade [%]',       'Avg Trade',    '{:.2f}%'),
        ('Avg. Trade Duration',  'Avg Duration', '{}'),
    ]
    print(f"\n  {'─'*40}")
    for key, label, fmt in metrics:
        if key in stats:
            try:
                val = stats[key]
                txt = fmt.format(int(val) if 'd' in fmt else float(val))
                print(f"  {label:<20} {txt:>12}")
            except Exception:
                pass

    params = extract_best_params(stats, combo, indicator_grid)

    # Also extract RR/execution parameters from the strategy
    rr_param_names = ['rr_ratio', 'atr_sl_mult', 'risk_pct', 'atr_period', 'max_hold_bars']
    rr_params = {}
    try:
        strategy_cls = stats._strategy
        for p in rr_param_names:
            if hasattr(strategy_cls, p):
                rr_params[p] = getattr(strategy_cls, p)
    except AttributeError:
        pass

    if params or rr_params:
        print(f"\n  {'─'*40}")
        print(f"  OPTIMISED PARAMETERS")
        if params:
            print(f"  -- Indicator params --")
            for k, v in sorted(params.items()):
                print(f"  {k:<25} {v}")
        if rr_params:
            print(f"  -- Risk/Reward params --")
            for k in rr_param_names:           # preserve logical order
                if k in rr_params:
                    v = rr_params[k]
                    if k == 'rr_ratio':
                        print(f"  {'rr_ratio':<25} {v}  ← R:R (e.g. 2.0 = 2× reward per risk)")
                    elif k == 'atr_sl_mult':
                        print(f"  {'atr_sl_mult':<25} {v}  ← SL = {v}× ATR from fill")
                    elif k == 'risk_pct':
                        print(f"  {'risk_pct':<25} {v}  ← {float(v)*100:.1f}% capital risked per trade")
                    elif k == 'atr_period':
                        print(f"  {'atr_period':<25} {v}  ← ATR period for SL/TP sizing")
                    elif k == 'max_hold_bars':
                        print(f"  {'max_hold_bars':<25} {v}  ← forced exit after {v} bars")
                    else:
                        print(f"  {k:<25} {v}")

    if top_results:
        print(f"\n  {'─'*70}")
        print(f"  TOP 5 COMBINATIONS")
        print(f"  {'Rank':<5} {'Score':>7}  {'Return':>8}  {'CompRet':>8}  "
              f"{'Sharpe':>7}  {'Trades':>7}  Indicators")
        print(f"  {'─'*72}")
        for rank, (sc_i, combo_i, st_i) in enumerate(top_results, 1):
            try:
                r      = float(st_i['Return [%]'])
                sh     = float(st_i.get('Sharpe Ratio', 0) or 0)
                tr     = int(st_i['# Trades'])
                nm     = '+'.join(c.replace('use_','') for c in combo_i)
                comp_i = compute_compounded_stats(st_i, cash=cash, reinvest_rate=reinvest_rate)
                cr     = comp_i.get('compounded_return_pct', 0.0)
                print(f"  #{rank:<4} {sc_i:>7.3f}  {r:>7.1f}%  {cr:>7.1f}%  "
                      f"{sh:>7.3f}  {tr:>7d}  {nm}")
            except Exception:
                pass

    print(f"\n{'='*70}\n")


def print_dual_sizing_comparison(atr_stats, other_stats, cash=100_000,
                                 reinvest_rate=0.50, label_a='ATR-risk',
                                 label_b='GARCH-risk'):
    """
    Prints two sizing_mode runs SIDE BY SIDE for the exact same combo/
    indicator params/signal logic — only sizing_mode differs between the
    runs that produced these stats. label_a/label_b let callers reuse this
    for ATR-vs-GARCH (Option 4) or ATR-vs-alpha (alpha pipeline) without
    the printed labels lying about which comparison actually ran.
    """
    print(f"\n{'='*70}")
    print(f"  DUAL SIZING COMPARISON — same signals, {label_a} vs {label_b}")
    print(f"{'='*70}")

    def g(stats, key, default=0.0):
        try:
            return float(stats.get(key, default) or default)
        except Exception:
            return default

    atr_comp   = compute_compounded_stats(atr_stats,   cash=cash, reinvest_rate=reinvest_rate)
    garch_comp = compute_compounded_stats(other_stats, cash=cash, reinvest_rate=reinvest_rate)

    rows = [
        ('Return [%]',        lambda s: g(s, 'Return [%]'),          '{:>10.2f}%'),
        ('Sharpe Ratio',      lambda s: g(s, 'Sharpe Ratio'),        '{:>11.3f}'),
        ('Sortino Ratio',     lambda s: g(s, 'Sortino Ratio'),       '{:>11.3f}'),
        ('Max Drawdown [%]',  lambda s: g(s, 'Max. Drawdown [%]'),   '{:>10.2f}%'),
        ('Win Rate [%]',      lambda s: g(s, 'Win Rate [%]'),        '{:>10.1f}%'),
        ('Profit Factor',     lambda s: g(s, 'Profit Factor'),       '{:>11.2f}'),
        ('# Trades',          lambda s: g(s, '# Trades'),            '{:>11.0f}'),
        ('Equity Final [$]',  lambda s: g(s, 'Equity Final [$]'),    '{:>11,.0f}'),
    ]

    print(f"\n  {'Metric':<22} {label_a:>12} {label_b:>12}   Winner")
    print(f"  {'─'*62}")
    for label, fn, fmt in rows:
        a_val = fn(atr_stats)
        g_val = fn(other_stats)
        a_txt = fmt.format(a_val)
        g_txt = fmt.format(g_val)
        winner = label_b if g_val > a_val else (label_a if a_val > g_val else '=')
        print(f"  {label:<22} {a_txt:>12} {g_txt:>12}   {winner}")

    print(f"  {'─'*62}")
    print(f"  {'Compounded Return':<22} {atr_comp.get('compounded_return_pct',0):>11.2f}% "
          f"{garch_comp.get('compounded_return_pct',0):>11.2f}%")
    print(f"  {'Compounded Sharpe':<22} {atr_comp.get('compounded_sharpe',0):>12.3f} "
          f"{garch_comp.get('compounded_sharpe',0):>12.3f}")
    print(f"  {'Compounded Max DD':<22} {atr_comp.get('compounded_max_dd',0):>11.2f}% "
          f"{garch_comp.get('compounded_max_dd',0):>11.2f}%")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    global MIN_TRADES, MIN_TRADES_HARD, MIN_TRADES_SOFT

    parser = argparse.ArgumentParser(description="Multi-Indicator Backtester (datacan5_fixed2)")
    parser.add_argument('--file',         type=str,   default='DLF-EQ.csv')
    parser.add_argument('--cash',         type=float, default=100_000)
    parser.add_argument('--commission',   type=float, default=0.07)
    parser.add_argument('--sizes',        type=int, nargs='+', default=[2, 3])
    parser.add_argument('--single-run',   action='store_true')
    parser.add_argument('--combo',        type=str, default=None)
    parser.add_argument('--param',        action='append', default=[])
    parser.add_argument('--signal-mode',  type=int, choices=[0,1,2], default=2)
    parser.add_argument('--reinvest-rate',type=float, default=0.5)
    parser.add_argument('--n-jobs',       type=int, default=-1)
    parser.add_argument('--min-trades',   type=int, default=MIN_TRADES)
    parser.add_argument('--pipeline',     action='store_true')
    parser.add_argument('--top-n',        type=int, default=14)
    parser.add_argument('--bayes-calls',  type=int, default=150)
    parser.add_argument('--stop-after-combo', action='store_true',
                        help='Run Stage 1 (solo screen) + Stage 2 (combo search) only, '
                             'then print top-5 combos and exit — skips Bayesian '
                             'fine-tuning (Stage 3) and Monte Carlo (Stage 4). Useful '
                             'for a fast sanity-check pass before committing to a full '
                             'pipeline run.')
    parser.add_argument('--fib-debug',    action='store_true')
    parser.add_argument('--start-date',   type=str, default=None)
    parser.add_argument('--end-date',     type=str, default=None)

    # Option 4 — GARCH Pipeline
    parser.add_argument('--sizing-mode',  type=str, choices=['legacy', 'atr', 'garch', 'alpha'],
                        default='legacy',
                        help="'legacy' = original full-equity sizing (unchanged behavior); "
                             "'atr' = NEW real risk_pct-based sizing off ATR; "
                             "'garch' = Option 4: risk_pct-based sizing off a GJR-GARCH "
                             "volatility forecast (requires --garch-file); "
                             "'alpha' = risk_pct-based sizing off the HMM->SARIMAX->LSTM->"
                             "GARCH alpha pipeline's PredVol (requires --alpha-file)")
    parser.add_argument('--garch-file',   type=str, default=None,
                        help='garch_risk.py output CSV (Date, GarchVol) — required when '
                             '--sizing-mode garch')
    parser.add_argument('--garch-sl-mult', type=float, default=1.5,
                        help='Stop-loss multiple applied to GarchVol/PredVol (mirrors '
                             'atr_sl_mult, default 1.5)')
    parser.add_argument('--dual-compare', action='store_true',
                        help="After a --sizing-mode garch|alpha --pipeline run, also re-run "
                             "the SAME best combo/params under sizing_mode='atr' and print a "
                             "side-by-side comparison table")

    # Alpha pipeline (HMM -> SARIMAX -> LSTM -> GARCH)
    parser.add_argument('--alpha-file',   type=str, default=None,
                        help='alpha_pipeline.py output CSV (Date, Regime, MasterPred, '
                             'PredVol) — required when --sizing-mode alpha or '
                             '--direction-mode is not "off"')
    parser.add_argument('--direction-mode', type=str, choices=['off', 'gate', 'replace'],
                        default='off',
                        help="'off' = MasterPred direction ignored (sizing-only, if "
                             "sizing_mode=alpha); 'gate' = an indicator-signaled buy is only "
                             "taken if MasterPred also predicts a higher close; "
                             "'replace' = entries decided purely by MasterPred direction, "
                             "indicator votes ignored for entries (exits unaffected)")

    args = parser.parse_args()

    MIN_TRADES      = int(args.min_trades)
    MIN_TRADES_SOFT = MIN_TRADES
    MIN_TRADES_HARD = max(20, MIN_TRADES_SOFT - 30)

    if not os.path.exists(args.file):
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)

    if args.sizing_mode == 'garch':
        if not args.garch_file:
            print("ERROR: --sizing-mode garch requires --garch-file")
            sys.exit(2)
        if not os.path.exists(args.garch_file):
            print(f"ERROR: GARCH file not found: {args.garch_file}")
            sys.exit(1)

    needs_alpha_file = (args.sizing_mode == 'alpha') or (args.direction_mode != 'off')
    if needs_alpha_file:
        if not args.alpha_file:
            print("ERROR: --sizing-mode alpha or --direction-mode gate/replace "
                  "requires --alpha-file")
            sys.exit(2)
        if not os.path.exists(args.alpha_file):
            print(f"ERROR: Alpha file not found: {args.alpha_file}")
            sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  MULTI-INDICATOR BACKTESTER  datacan5_fixed2")
    print(f"  FIX-1: next-bar open fills")
    print(f"  FIX-2: sqrt(252) Sharpe annualisation")
    print(f"  FIX-3: bootstrap Monte Carlo")
    print(f"  FIX-4: SL/TP from actual fill (distances, not absolute levels)")
    print(f"  FIX-5: instance-level pending state (no class-level leakage)")
    if args.sizing_mode == 'garch':
        print(f"  {'─'*66}")
        print(f"  ★ OPTION 4: GARCH PIPELINE")
        print(f"     GARCH file    : {args.garch_file}")
        print(f"     garch_sl_mult : {args.garch_sl_mult}")
        print(f"     Real risk_pct-based sizing (both ATR and GARCH modes now")
        print(f"     implement this — 'legacy' mode remains for backward compat)")
    elif args.sizing_mode == 'atr':
        print(f"  {'─'*66}")
        print(f"  Sizing mode: ATR-risk (real risk_pct-based sizing, newly enabled)")
    if args.sizing_mode == 'alpha' or args.direction_mode != 'off':
        print(f"  {'─'*66}")
        print(f"  ★ ALPHA PIPELINE  (HMM -> SARIMAX -> LSTM -> GARCH)")
        print(f"     Alpha file     : {args.alpha_file}")
        print(f"     sizing_mode    : {args.sizing_mode}"
              + ("  ← PredVol drives stop + size" if args.sizing_mode == 'alpha' else ""))
        print(f"     direction_mode : {args.direction_mode}"
              + {"off": "  ← MasterPred ignored for entries",
                 "gate": "  ← indicator buy also requires MasterPred agreement",
                 "replace": "  ← entries decided purely by MasterPred direction"
                 }[args.direction_mode])
    print(f"{'='*70}")

    existing_cp = _load_checkpoint()
    if existing_cp:
        print(f"\n  Found previous checkpoint:")
        _print_checkpoint_summary(existing_cp)

    try:
        df = load_data(args.file)
        if args.start_date:
                df = df[df.index >= args.start_date]
        if args.end_date:
                df = df[df.index <= args.end_date]

        if args.sizing_mode == 'garch':
            df = merge_garch(df, args.garch_file)
            n_garch_valid = df['GarchVol'].notna().sum()
            print(f"\n  GarchVol merged — {n_garch_valid}/{len(df)} bars have a valid forecast "
                  f"(earlier bars are pre-burn-in and will skip GARCH-sized entries)")

        if needs_alpha_file:
            df = merge_alpha(df, args.alpha_file)
            n_alpha_valid = df['MasterPred'].notna().sum()
            print(f"\n  Alpha pipeline merged — {n_alpha_valid}/{len(df)} bars have a valid "
                  f"MasterPred/PredVol (earlier bars are pre-burn-in and will skip "
                  f"alpha-gated/alpha-sized entries)")

        df, warmup, df_test = choose_data_range(df)
        print(f"\n  Candles : {len(df)}")
        print(f"  Range   : {df.index[0].date()} → {df.index[-1].date()}")
        print(f"  Cash    : ${args.cash:,.0f}")
        print(f"  Comm    : {args.commission}%")
        print(f"  Warmup  : {warmup} bars")

        commission_frac = args.commission / 100.0
        MultiIndicatorStrategy.signal_mode   = int(args.signal_mode)
        MultiIndicatorStrategy.sizing_mode   = args.sizing_mode
        MultiIndicatorStrategy.garch_sl_mult = float(args.garch_sl_mult)
        MultiIndicatorStrategy.direction_mode = args.direction_mode
        all_indicators  = list(get_indicator_grid().keys())

        if args.fib_debug:
            close = np.array(df['Close']); high = np.array(df['High']); low = np.array(df['Low'])
            sh, sl, sq = find_swing_points_enhanced(high, low, close, base_lookback=5, min_prominence_atr=0.5)
            print(f"\n  Swing Highs: {int(np.sum(~np.isnan(sh)))}")
            print(f"  Swing Lows : {int(np.sum(~np.isnan(sl)))}")
            sys.exit(0)

        if args.single_run:
            if not args.combo:
                print("ERROR: --single-run requires --combo"); sys.exit(2)
            combo_list  = [c.strip() for c in args.combo.split(',') if c.strip()]
            run_kwargs  = {ind: (ind in combo_list) for ind in all_indicators}
            run_kwargs['warmup_bars'] = warmup
            for p in args.param:
                if '=' not in p: continue
                k, v = p.split('=', 1)
                k = k.strip(); v = v.strip()
                try:    vv = int(v)
                except:
                    try: vv = float(v)
                    except:
                        vv = (v.lower() == 'true') if v.lower() in ('true','false') else v
                run_kwargs[k] = vv
            bt    = Backtest(df, MultiIndicatorStrategy,
                             cash=args.cash, commission=commission_frac, exclusive_orders=True)
            stats = bt.run(**run_kwargs)
            for k, v in stats.items():
                print(f"  {k:<30} {v}")
            sys.exit(0)

        if args.pipeline:
            screen_results, shortlist = run_solo_screen(
                df, args.cash, commission_frac,
                signal_mode=int(args.signal_mode), warmup_bars=warmup)

            if len(shortlist) < 2:
                print("\n  ERROR: Fewer than 2 valid indicators.")
                sys.exit(1)

            top_n     = min(args.top_n, len(shortlist))
            shortlist = shortlist[:top_n]
            print(f"\n  Using top {top_n}: {[i.replace('use_','') for i in shortlist]}")

            best_combo, best_stats, total_time, top_results, _ = run_combo_search(
                df, args.cash, commission_frac, shortlist, all_indicators,
                sizes=args.sizes, n_jobs=args.n_jobs,
                reinvest_rate=args.reinvest_rate, warmup_bars=warmup)

            if best_combo is None:
                print("\n  ERROR: No valid combos in Stage 2.")
                sys.exit(1)

            if args.stop_after_combo:
                display_results(best_combo, best_stats, total_time, top_results,
                                get_indicator_grid(), reinvest_rate=args.reinvest_rate,
                                cash=args.cash)
                print(f"\n  --stop-after-combo set — skipping Bayesian fine-tune "
                      f"(Stage 3) and Monte Carlo (Stage 4).\n")
                _delete_checkpoint()
                return

            final_stats, best_params, final_kwargs = run_bayesian_finetune(
                df, args.cash, commission_frac, best_combo, best_stats,
                all_indicators, warmup_bars=warmup,
                n_calls=args.bayes_calls, signal_mode=int(args.signal_mode))

            run_monte_carlo(final_stats, cash=args.cash, n_sims=1000)

            display_results(best_combo, final_stats, total_time, top_results,
                            get_indicator_grid(), reinvest_rate=args.reinvest_rate,
                            cash=args.cash)

            # ── Dual comparison: same combo/params, ATR-risk vs GARCH-risk
            #    OR ATR-risk vs alpha-pipeline-risk sizing, re-using the
            #    EXACT kwargs Stage 3 landed on so entries/exits are
            #    identical and only sizing differs ──
            if args.sizing_mode in ('garch', 'alpha') and args.dual_compare:
                shadow_kwargs = dict(final_kwargs)
                original_sizing_mode = args.sizing_mode
                MultiIndicatorStrategy.sizing_mode = 'atr'
                try:
                    bt_shadow = Backtest(df, MultiIndicatorStrategy,
                                         cash=args.cash, commission=commission_frac,
                                         exclusive_orders=True)
                    atr_shadow_stats = bt_shadow.run(**shadow_kwargs)
                finally:
                    # restore even if the shadow run raises — otherwise the
                    # walk-forward test below silently inherits sizing_mode
                    # 'atr' instead of the real one, since test_kwargs never
                    # sets sizing_mode explicitly and relies on the class attr
                    MultiIndicatorStrategy.sizing_mode = original_sizing_mode
                label_b = 'GARCH-risk' if original_sizing_mode == 'garch' else 'alpha-risk'
                print_dual_sizing_comparison(atr_shadow_stats, final_stats,
                                             cash=args.cash,
                                             reinvest_rate=args.reinvest_rate,
                                             label_a='ATR-risk', label_b=label_b)

            if df_test is not None and best_combo is not None:
                    print(f"\n{'='*70}")
                    print(f"  WALK-FORWARD OUT-OF-SAMPLE TEST")
                    print(f"  Period : {df_test.index[0].date()} → {df_test.index[-1].date()}")
                    print(f"  Combo  : {'+'.join(c.replace('use_','') for c in best_combo)}")
                    print(f"  No reoptimization — same parameters from training period")
                    print(f"{'='*70}\n")

                    test_kwargs = {ind: (ind in best_combo) for ind in all_indicators}
                    test_kwargs.update(best_params)
                    test_kwargs.update({
                        'signal_mode': int(args.signal_mode),
                        'warmup_bars': 0,
                        'use_atr': True,
                        'use_slippage': True,
                        'risk_pct': 0.01,
                        'atr_period': 14,
                    })

                    bt_test    = Backtest(df_test, MultiIndicatorStrategy,
                                        cash=args.cash, commission=commission_frac,
                                        exclusive_orders=True)
                    test_stats = bt_test.run(**test_kwargs)

                    ret    = float(test_stats.get('Return [%]',            0) or 0)
                    bh     = float(test_stats.get('Buy & Hold Return [%]', 0) or 0)
                    sharpe = float(test_stats.get('Sharpe Ratio',          0) or 0)
                    trades = int(test_stats.get('# Trades',                0) or 0)
                    dd     = float(test_stats.get('Max. Drawdown [%]',     0) or 0)
                    winr   = float(test_stats.get('Win Rate [%]',          0) or 0)

                    print(f"  {'METRIC':<28} {'VALUE':>12}")
                    print(f"  {'─'*42}")
                    print(f"  {'Return [%]':<28} {ret:>11.2f}%")
                    print(f"  {'B&H Return [%]':<28} {bh:>11.2f}%")
                    print(f"  {'Alpha':<28} {ret-bh:>11.2f}%")
                    print(f"  {'Sharpe Ratio':<28} {sharpe:>12.3f}")
                    print(f"  {'Max Drawdown [%]':<28} {dd:>11.2f}%")
                    print(f"  {'Trades':<28} {trades:>12d}")
                    print(f"  {'Win Rate':<28} {winr:>11.1f}%")
                    print(f"\n  {'─'*42}")
                    if ret > bh:
                        print(f"  ✓ Strategy beat B&H by {ret-bh:.1f}% out-of-sample")
                    else:
                        print(f"  ✗ Strategy underperformed B&H by {bh-ret:.1f}% out-of-sample")
                    print(f"{'='*70}\n")
            _delete_checkpoint()
            return

        # Default: full combinatorial search
        indicator_grid = get_indicator_grid()
        sizes  = tuple(args.sizes)
        combos = []
        for r in sizes:
            combos.extend(list(itertools.combinations(all_indicators, r)))
        random.shuffle(combos)
        total = len(combos)
        n_workers = max(1, cpu_count()-1) if args.n_jobs == -1 else max(1, min(args.n_jobs, cpu_count()))

        fib_cache  = _build_fib_cache(df)
        stage_rr   = dict(RR_GRID)
        completed  = 0
        start_time = time.time()
        results_log= []

        with Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(fib_cache, df, indicator_grid,
                      all_indicators, all_indicators,
                      stage_rr, MIN_TRADES_HARD, MIN_TRADES_SOFT,
                      args.cash, commission_frac, warmup),
            maxtasksperchild=16,
        ) as pool:
            result_iter = pool.imap_unordered(_run_combo_worker, combos, chunksize=1)
            recent_times = deque(maxlen=50)
            for result in result_iter:
                sc, combo, stats = result
                completed += 1
                now = time.time()
                recent_times.append(now)
                elapsed = now - start_time
                if len(recent_times) >= 2:
                    rate = (len(recent_times) - 1) / max(recent_times[-1] - recent_times[0], 0.001)
                else:
                    rate = completed / max(elapsed, 0.001)
                rem = (total - completed) / max(rate, 0.001)
                names     = '+'.join(c.replace('use_', '') for c in combo)
                if stats is not None and sc > -1000.0:
                    results_log.append((sc, combo, stats))
                    try:
                        ret    = float(stats.get('Return [%]',   0) or 0)
                        trades = int(stats.get('# Trades',       0) or 0)
                        sharpe = float(stats.get('Sharpe Ratio', 0) or 0)
                        winr   = float(stats.get('Win Rate [%]', 0) or 0)
                        pf     = float(stats.get('Profit Factor',0) or 0)
                    except Exception:
                        ret = trades = sharpe = winr = pf = 0
                    best       = max(results_log, key=lambda x: x[0])
                    best_names = '+'.join(c.replace('use_','') for c in best[1])
                    _print_combo_line(completed, total, elapsed, rem,
                                      sc, ret, trades, sharpe, winr, pf,
                                      names, best_names, best[0])
                else:
                    _print_combo_line(completed, total, elapsed, rem,
                                      0, 0, 0, 0, 0, 0, names, None, 0, skip=True)

        results_log.sort(key=lambda x: x[0], reverse=True)
        display_results(
            results_log[0][1] if results_log else None,
            results_log[0][2] if results_log else None,
            time.time() - start_time,
            results_log[:5],
            get_indicator_grid(),
            reinvest_rate=args.reinvest_rate,
            cash=args.cash
        )
        _delete_checkpoint()

    except KeyboardInterrupt:
        cp = _load_checkpoint()
        if cp:
            _print_checkpoint_summary(cp)
        print(f"\nInterrupted. Checkpoint saved to {_CHECKPOINT_FILE}")
        sys.exit(130)
    except Exception as e:
        cp = _load_checkpoint()
        if cp:
            _print_checkpoint_summary(cp)
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
