"""
Backtest: Selling Weekend Volatility
=====================================
Replicates the strategy from:
  https://insights.deribit.com/education/option-backtest-selling-weekend-vol/

Strategy:
  - Every Friday at 16:00 UTC, sell a 0.35-delta strangle
    (short call + short put) expiring Sunday 08:00 UTC.
  - Hold to expiry (settlement).
  - Flat position size: 1 contract per side per week.
  - Starting balance: 1 BTC.
  - Fee: 0.0003 BTC per option (Deribit standard).
  - Slippage: 5% reduction on premium collected.

Data used:
  - Hourly BTC index (data/underlying/btc_index_60.parquet)
  - Daily DVOL (data/underlying/btc_dvol_1D.parquet)
  - Instruments catalog (data/instruments/btc_instruments.parquet)
  - Settlement data (data/settlements/btc_settlements.parquet)
  - Option prices are computed via Black-76 with DVOL as IV estimate.
"""

import sys
import os
import argparse
from datetime import datetime, timedelta, date, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path so we can import the pricing module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from options_backtest.pricing.black76 import call_price, put_price, delta as bs_delta


# ──────────────────────── Config ──────────────────────────
DEFAULT_TARGET_DELTA = 0.35
DEFAULT_MAX_DELTA_DIFF = 0.10
DEFAULT_SLIPPAGE = 0.05          # 5% of premium
FEE_PER_OPTION_BTC = 0.0003     # Deribit 0.03% capped at 0.0003
ENTRY_HOUR = 16                  # 16:00 UTC Friday
HOURS_TO_EXPIRY = 40             # Friday 16:00 → Sunday 08:00 (default)
T_YEARS = HOURS_TO_EXPIRY / 8760.0  # ≈ 0.004566
STARTING_BALANCE = 1.0           # 1 BTC
TRADE_SIZE = 1.0                 # 1 contract per side

# Map expire day name to (day_of_week, hours_from_friday_16, friday_offset_days)
EXPIRE_DAY_MAP = {
    "saturday": (5, 16, 1),   # Sat 08:00 = 16h,  Friday = Sat - 1
    "sunday":   (6, 40, 2),   # Sun 08:00 = 40h,  Friday = Sun - 2
    "monday":   (0, 64, 3),   # Mon 08:00 = 64h,  Friday = Mon - 3
}


def load_data(data_dir: str, expire_dow: int = 6):
    """Load all required data.

    Parameters
    ----------
    data_dir : data directory path
    expire_dow : day of week for expiry filter (5=Sat, 6=Sun, 0=Mon)
    """
    data_dir = Path(data_dir)

    # Hourly underlying
    spot_h = pd.read_parquet(data_dir / "underlying" / "btc_index_60.parquet")
    spot_h["timestamp"] = pd.to_datetime(spot_h["timestamp"], utc=True)
    spot_h = spot_h.set_index("timestamp").sort_index()

    # Daily DVOL
    dvol = pd.read_parquet(data_dir / "underlying" / "btc_dvol_1D.parquet")
    dvol["timestamp"] = pd.to_datetime(dvol["timestamp"], utc=True)
    dvol = dvol.set_index("timestamp").sort_index()

    # Instruments
    inst = pd.read_parquet(data_dir / "instruments" / "btc_instruments.parquet")
    inst["exp_dt"] = pd.to_datetime(inst["expiration_date"], utc=True)
    inst["exp_date"] = inst["exp_dt"].dt.date
    inst["exp_dow"] = inst["exp_dt"].dt.dayofweek
    # Filter by target expiry day of week
    filtered_inst = inst[inst["exp_dow"] == expire_dow].copy()

    # Settlements
    sett = pd.read_parquet(data_dir / "settlements" / "btc_settlements.parquet")
    sett["ts"] = pd.to_datetime(sett["settlement_timestamp"], utc=True)

    return spot_h, dvol, filtered_inst, sett


def get_friday_spot(spot_h: pd.DataFrame, friday_dt: datetime) -> float:
    """Get BTC spot price on Friday at 16:00 UTC."""
    target = pd.Timestamp(friday_dt).tz_localize("UTC") if friday_dt.tzinfo is None else pd.Timestamp(friday_dt)
    # Try exact match first
    if target in spot_h.index:
        return float(spot_h.loc[target, "close"])
    # Nearest within 1h
    idx = spot_h.index.get_indexer([target], method="nearest")[0]
    if idx >= 0 and abs((spot_h.index[idx] - target).total_seconds()) < 3600:
        return float(spot_h.iloc[idx]["close"])
    return np.nan


def get_friday_dvol(dvol: pd.DataFrame, friday_date: date) -> float:
    """Get DVOL for Friday (close), fallback to nearest day."""
    target = pd.Timestamp(friday_date, tz=timezone.utc)
    if target in dvol.index:
        return float(dvol.loc[target, "close"])
    # Fallback: nearest day before
    mask = dvol.index <= target
    if mask.any():
        return float(dvol.loc[dvol.index[mask][-1], "close"])
    return np.nan


def _find_best_strike(options_df, spot, sigma, option_type, target_delta,
                      max_delta_diff, price_fn, delta_fn, t_years=None):
    """Find the strike closest to target_delta. Returns (K, delta, price_btc) or None."""
    _t = t_years if t_years is not None else T_YEARS
    best = None
    best_diff = 999
    for _, row in options_df.iterrows():
        K = row["strike_price"]
        d = delta_fn(spot, K, _t, sigma, option_type=option_type)
        diff = abs(abs(d) - target_delta)
        if diff < best_diff:
            best_diff = diff
            px = price_fn(spot, K, _t, sigma)
            best = (K, d, px / spot, best_diff)
    if best is None or best[3] > max_delta_diff:
        return None
    return best[:3]  # (K, delta, price_btc)


def select_strikes(sun_inst: pd.DataFrame, expiry_date: date,
                   spot: float, iv: float, target_delta: float,
                   max_delta_diff: float, wing_delta: float = 0.0,
                   t_years: float = None):
    """
    Select call and put strikes closest to target delta for a given expiry.
    If wing_delta > 0, also selects long wing options further OTM (iron condor).

    Returns dict with keys:
      short_call, short_put: (K, delta, price_btc)
      wing_call, wing_put:   (K, delta, price_btc) or None
    Or None if insufficient strikes.
    """
    opts = sun_inst[sun_inst["exp_date"] == expiry_date]
    if len(opts) == 0:
        return None

    calls = opts[opts["option_type"].str.lower() == "call"]
    puts = opts[opts["option_type"].str.lower() == "put"]

    if len(calls) == 0 or len(puts) == 0:
        return None

    sigma = iv / 100.0  # DVOL is in % (e.g., 50 → 0.50)

    # Short legs
    sc = _find_best_strike(calls, spot, sigma, "call", target_delta,
                           max_delta_diff, call_price, bs_delta, t_years)
    sp = _find_best_strike(puts, spot, sigma, "put", target_delta,
                           max_delta_diff, put_price, bs_delta, t_years)
    if sc is None or sp is None:
        return None

    result = {"short_call": sc, "short_put": sp,
              "wing_call": None, "wing_put": None}

    # Wing legs (if requested)
    if wing_delta > 0:
        # Long call wing: further OTM calls (lower delta than short call)
        wing_calls = calls[calls["strike_price"] > sc[0]]
        wc = _find_best_strike(wing_calls, spot, sigma, "call", wing_delta,
                               0.15, call_price, bs_delta, t_years)
        # Long put wing: further OTM puts (lower |delta| than short put)
        wing_puts = puts[puts["strike_price"] < sp[0]]
        wp = _find_best_strike(wing_puts, spot, sigma, "put", wing_delta,
                               0.15, put_price, bs_delta, t_years)
        result["wing_call"] = wc
        result["wing_put"] = wp

    return result


def get_settlement_index(sett: pd.DataFrame, expiry_date: date) -> float:
    """Get the BTC index price at Sunday 08:00 UTC settlement."""
    exp_dt = pd.Timestamp(datetime.combine(expiry_date, datetime.min.time()),
                          tz=timezone.utc).replace(hour=8)
    # All options for this expiry have the same index_price
    mask = (sett["ts"].dt.date == expiry_date)
    subset = sett[mask]
    if len(subset) == 0:
        return np.nan
    return float(subset.iloc[0]["index_price"])


def run_backtest(data_dir: str = "data",
                 start_date: str = "2024-01-01",
                 end_date: str = "2024-08-31",
                 target_delta: float = DEFAULT_TARGET_DELTA,
                 max_delta_diff: float = DEFAULT_MAX_DELTA_DIFF,
                 slippage: float = DEFAULT_SLIPPAGE,
                 iv_discount: float = 0.0,
                 leverage: float = 1.0,
                 margin_mode: str = "coin",
                 wing_delta: float = 0.0,
                 expire_day: str = "sunday",
                 verbose: bool = True):
    """Run the weekend vol selling backtest.

    margin_mode:
      - 'coin': Deribit native, PnL in BTC, starting balance 1 BTC.
      - 'usd':  USD-margined. Starting balance = 1 BTC worth of USD at first trade.
                Leverage multiplies position size relative to equity.
    expire_day:
      - 'saturday': settlement Sat 08:00 (16h hold)
      - 'sunday':   settlement Sun 08:00 (40h hold, default)
      - 'monday':   settlement Mon 08:00 (64h hold)
    """

    expire_day_lower = expire_day.lower()
    if expire_day_lower not in EXPIRE_DAY_MAP:
        raise ValueError(f"expire_day must be saturday/sunday/monday, got: {expire_day}")

    expire_dow, hours_to_expiry, friday_offset = EXPIRE_DAY_MAP[expire_day_lower]
    t_years = hours_to_expiry / 8760.0

    spot_h, dvol, target_inst, sett = load_data(data_dir, expire_dow=expire_dow)

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    # Find all target day dates in the range
    all_target_dates = sorted(target_inst["exp_date"].unique())
    target_dates = [d for d in all_target_dates if start <= d <= end]

    is_usd = margin_mode.lower() == "usd"

    day_label = expire_day_lower.capitalize()
    if verbose:
        print(f"{'='*70}")
        print(f"  Backtest: Selling Weekend Volatility")
        print(f"  Period: {start_date} → {end_date}")
        print(f"  Settlement: {day_label} 08:00 UTC ({hours_to_expiry}h hold)")
        print(f"  Target delta: {target_delta}, Max diff: {max_delta_diff}")
        print(f"  Slippage: {slippage*100:.0f}%, Fee: {FEE_PER_OPTION_BTC} BTC/option")
        print(f"  IV discount: {iv_discount*100:.0f}%")
        wing_str = f", Wing delta: {wing_delta}" if wing_delta > 0 else ""
        print(f"  Margin: {margin_mode.upper()}, Leverage: {leverage:.1f}x{wing_str}")
        print(f"  {day_label} expiry dates in range: {len(target_dates)}")
        print(f"{'='*70}\n")

    trades = []
    initial_usd = None      # set on first trade for USD mode
    balance = STARTING_BALANCE  # BTC in coin mode, USD in usd mode
    peak_balance = STARTING_BALANCE
    max_dd = 0.0
    equity_curve = [(start, STARTING_BALANCE)]

    for exp_date in target_dates:
        # Corresponding Friday = expiry - friday_offset days
        fri_date = exp_date - timedelta(days=friday_offset)
        fri_dt = datetime(fri_date.year, fri_date.month, fri_date.day,
                          ENTRY_HOUR, 0, 0, tzinfo=timezone.utc)

        # Get Friday spot & IV
        spot = get_friday_spot(spot_h, fri_dt)
        if np.isnan(spot):
            if verbose:
                print(f"  {exp_date} SKIP: no spot data for {fri_date}")
            continue

        iv = get_friday_dvol(dvol, fri_date)
        if np.isnan(iv):
            if verbose:
                print(f"  {exp_date} SKIP: no DVOL for {fri_date}")
            continue

        # Apply IV discount if any
        effective_iv = iv * (1.0 - iv_discount)

        # Select strikes
        result = select_strikes(target_inst, exp_date, spot, effective_iv,
                                target_delta, max_delta_diff, wing_delta, t_years)
        if result is None:
            if verbose:
                print(f"  {exp_date} SKIP: cannot find suitable strikes (spot={spot:.0f}, IV={iv:.1f})")
            continue

        sc = result["short_call"]   # (K, delta, price_btc)
        sp = result["short_put"]
        wc = result["wing_call"]    # (K, delta, price_btc) or None
        wp = result["wing_put"]
        call_K, call_d, call_px_btc = sc
        put_K, put_d, put_px_btc = sp

        # Settlement
        settle_price = get_settlement_index(sett, exp_date)
        if np.isnan(settle_price):
            if verbose:
                print(f"  {exp_date} SKIP: no settlement data")
            continue

        # ── Position sizing with leverage ──
        if is_usd:
            if initial_usd is None:
                initial_usd = spot  # 1 BTC worth of USD at first trade
                balance = initial_usd
                peak_balance = initial_usd
                equity_curve[0] = (start, initial_usd)
            # Contracts = (equity / spot) * leverage
            qty = (balance / spot) * leverage
        else:
            # Coin mode: flat 1 contract * leverage
            qty = TRADE_SIZE * leverage

        # Premium collected (BTC) - short legs
        gross_premium = (call_px_btc + put_px_btc) * qty
        n_legs = 2

        # Wing cost (BTC) - long legs (deducted from premium)
        wing_cost = 0.0
        if wc is not None:
            wing_cost += wc[2] * qty  # wing call price * qty
            n_legs += 1
        if wp is not None:
            wing_cost += wp[2] * qty  # wing put price * qty
            n_legs += 1

        net_premium = (gross_premium - wing_cost) * (1.0 - slippage)
        total_fee = FEE_PER_OPTION_BTC * n_legs * qty

        # Intrinsic payoffs at settlement (BTC terms, Deribit convention)
        # Short legs: we pay
        call_payout = max(0, (settle_price - call_K) / settle_price)
        put_payout = max(0, (put_K - settle_price) / settle_price)
        short_payout = (call_payout + put_payout) * qty

        # Wing legs: we receive
        wing_payout = 0.0
        if wc is not None:
            wing_payout += max(0, (settle_price - wc[0]) / settle_price) * qty
        if wp is not None:
            wing_payout += max(0, (wp[0] - settle_price) / settle_price) * qty

        # Trade PnL (BTC)
        pnl_btc = net_premium - short_payout + wing_payout - total_fee

        if is_usd:
            pnl = pnl_btc * settle_price
        else:
            pnl = pnl_btc

        balance += pnl

        # Track drawdown
        peak_balance = max(peak_balance, balance)
        dd = (peak_balance - balance) / peak_balance
        max_dd = max(max_dd, dd)

        trade_record = {
            "expiry": exp_date,
            "friday": fri_date,
            "spot_entry": spot,
            "iv": iv,
            "effective_iv": effective_iv,
            "call_K": call_K,
            "call_delta": call_d,
            "call_px_btc": call_px_btc,
            "put_K": put_K,
            "put_delta": put_d,
            "put_px_btc": put_px_btc,
            "wing_call_K": wc[0] if wc else None,
            "wing_put_K": wp[0] if wp else None,
            "wing_cost": wing_cost,
            "qty": qty,
            "gross_premium": gross_premium,
            "net_premium": net_premium,
            "fee": total_fee,
            "settle_price": settle_price,
            "call_payout": call_payout,
            "put_payout": put_payout,
            "wing_payout": wing_payout,
            "pnl": pnl,
            "balance": balance,
        }
        trades.append(trade_record)
        equity_curve.append((exp_date, balance))

        if verbose:
            win = "✓" if pnl >= 0 else "✗"
            print(f"  {exp_date} {win}  spot={spot:>8.0f}  IV={effective_iv:5.1f}  "
                  f"C:{call_K:>7.0f}(Δ{call_d:.3f})  P:{put_K:>7.0f}(Δ{put_d:.3f})  "
                  f"prem={net_premium:.6f}  settle={settle_price:>8.0f}  pnl={pnl:+.6f}  "
                  f"bal={balance:.4f}")

    # ────────── Summary ──────────
    df = pd.DataFrame(trades)
    if len(df) == 0:
        print("\nNo trades executed!")
        return df

    n_trades = len(df)
    n_wins = (df["pnl"] >= 0).sum()
    n_losses = (df["pnl"] < 0).sum()
    total_pnl = df["pnl"].sum()
    starting = initial_usd if is_usd else STARTING_BALANCE
    pct_return = total_pnl / starting * 100
    years = (end - start).days / 365.25
    apr = pct_return / years if years > 0 else 0
    avg_pnl = df["pnl"].mean()
    win_rate = n_wins / n_trades * 100

    # Sharpe-like: weekly returns → annualised
    weekly_rets = df["pnl"] / starting
    sharpe = (weekly_rets.mean() / weekly_rets.std() * np.sqrt(52)
              if weekly_rets.std() > 0 else 0)

    # Profit factor
    gross_wins = df.loc[df["pnl"] >= 0, "pnl"].sum()
    gross_losses = abs(df.loc[df["pnl"] < 0, "pnl"].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Calmar ratio
    calmar = (apr / 100) / max_dd if max_dd > 0 else float("inf")

    unit = "USD" if is_usd else "BTC"
    lev_str = f" ({leverage:.0f}x)" if leverage > 1 else ""

    print(f"\n{'='*70}")
    print(f"  RESULTS: Selling Weekend Vol ({start_date} → {end_date}){lev_str}")
    print(f"{'='*70}")
    print(f"  Margin mode:       {margin_mode.upper()}, Leverage: {leverage:.1f}x")
    print(f"  Starting balance:  {starting:,.2f} {unit}")
    print(f"  Trades:            {n_trades} ({n_wins} wins, {n_losses} losses)")
    print(f"  Win rate:          {win_rate:.1f}%")
    print(f"  Net profit:        {total_pnl:+,.2f} {unit} ({pct_return:+.2f}%)")
    print(f"  APR:               {apr:.1f}%")
    print(f"  Avg PnL/trade:     {avg_pnl:,.4f} {unit}")
    print(f"  Max drawdown:      {max_dd*100:.1f}% ({max_dd*starting:,.2f} {unit})")
    print(f"  Sharpe (ann.):     {sharpe:.2f}")
    print(f"  Profit factor:     {profit_factor:.2f}")
    print(f"  Calmar ratio:      {calmar:.2f}")
    print(f"{'='*70}")

    # ────────── Equity curve chart ──────────
    try:
        _generate_report(df, equity_curve, start_date, end_date,
                         target_delta, apr, max_dd, sharpe, calmar, profit_factor,
                         margin_mode)
    except Exception as e:
        print(f"  (Report generation failed: {e})")

    return df


def _generate_report(df, equity_curve, start_date, end_date,
                     target_delta, apr, max_dd, sharpe, calmar, profit_factor,
                     margin_mode="coin"):
    """Generate an HTML equity curve report."""
    import json

    unit = "USD" if margin_mode == "usd" else "BTC"
    is_usd = margin_mode == "usd"
    pnl_precision = 2 if is_usd else 6
    net_profit = df['pnl'].sum()
    net_profit_str = f"{net_profit:+,.2f}" if is_usd else f"{net_profit:+.4f}"

    eq_dates = [str(d) for d, _ in equity_curve]
    eq_values = [round(v, pnl_precision) for _, v in equity_curve]

    # Weekly PnL bars
    pnl_dates = [str(r["expiry"]) for r in df.to_dict("records")]
    pnl_values = [round(r["pnl"], pnl_precision) for r in df.to_dict("records")]
    pnl_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in pnl_values]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Weekend Vol Selling Backtest</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #1e1e2e; color: #cdd6f4; padding: 20px; }}
  .container {{ max-width: 1100px; margin: auto; }}
  h1 {{ color: #89b4fa; }}
  .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
  .metric {{ background: #313244; border-radius: 8px; padding: 15px; text-align: center; }}
  .metric .value {{ font-size: 24px; font-weight: bold; color: #a6e3a1; }}
  .metric .label {{ font-size: 12px; color: #9399b2; margin-top: 5px; }}
  .chart-box {{ background: #313244; border-radius: 8px; padding: 20px; margin: 20px 0; }}
  canvas {{ width: 100% !important; }}
</style></head>
<body><div class="container">
<h1>Weekend Vol Selling Backtest</h1>
<p>Strategy: Sell 0.{int(target_delta*100)}-delta strangle on Friday 16:00 UTC → Sunday 08:00 UTC expiry<br>
Period: {start_date} → {end_date}</p>

<div class="metrics">
  <div class="metric"><div class="value">{apr:.1f}%</div><div class="label">APR</div></div>
  <div class="metric"><div class="value">{max_dd*100:.1f}%</div><div class="label">Max Drawdown</div></div>
  <div class="metric"><div class="value">{sharpe:.2f}</div><div class="label">Sharpe Ratio</div></div>
  <div class="metric"><div class="value">{calmar:.2f}</div><div class="label">Calmar Ratio</div></div>
  <div class="metric"><div class="value">{len(df)}</div><div class="label">Total Trades</div></div>
  <div class="metric"><div class="value">{(df['pnl']>=0).sum()}/{(df['pnl']<0).sum()}</div><div class="label">Wins/Losses</div></div>
  <div class="metric"><div class="value">{profit_factor:.2f}</div><div class="label">Profit Factor</div></div>
  <div class="metric"><div class="value">{net_profit_str}</div><div class="label">Net Profit ({unit})</div></div>
</div>

<div class="chart-box">
  <canvas id="equityChart" height="300"></canvas>
</div>
<div class="chart-box">
  <canvas id="pnlChart" height="200"></canvas>
</div>

<script>
const eqCtx = document.getElementById('equityChart').getContext('2d');
new Chart(eqCtx, {{
  type: 'line',
  data: {{
    labels: {json.dumps(eq_dates)},
    datasets: [{{ label: 'Equity ({unit})', data: {json.dumps(eq_values)},
      borderColor: '#89b4fa', backgroundColor: 'rgba(137,180,250,0.1)',
      fill: true, tension: 0.2, pointRadius: 2 }}]
  }},
  options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'Equity Curve', color: '#cdd6f4' }} }},
    scales: {{ x: {{ ticks: {{ color: '#9399b2', maxTicksLimit: 20 }} }}, y: {{ ticks: {{ color: '#9399b2' }} }} }} }}
}});

const pnlCtx = document.getElementById('pnlChart').getContext('2d');
new Chart(pnlCtx, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(pnl_dates)},
    datasets: [{{ label: 'Weekly PnL ({unit})', data: {json.dumps(pnl_values)},
      backgroundColor: {json.dumps(pnl_colors)} }}]
  }},
  options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'Weekly PnL', color: '#cdd6f4' }} }},
    scales: {{ x: {{ ticks: {{ color: '#9399b2', maxTicksLimit: 20 }} }}, y: {{ ticks: {{ color: '#9399b2' }} }} }} }}
}});
</script>
</div></body></html>"""

    out_dir = Path("reports") / "weekend_vol"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"weekend_vol_{start_date}_{end_date}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n  Report saved: {out_path}")

    # Also save trades CSV
    csv_path = out_dir / f"trades_{start_date}_{end_date}.csv"
    pd.DataFrame(df).to_csv(csv_path, index=False)
    print(f"  Trades CSV:  {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekend Vol Selling Backtest")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-08-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--delta", type=float, default=0.35, help="Target delta")
    parser.add_argument("--max-diff", type=float, default=0.10, help="Max delta deviation")
    parser.add_argument("--slippage", type=float, default=0.05, help="Slippage on premium")
    parser.add_argument("--iv-discount", type=float, default=0.0,
                        help="Discount on DVOL for weekend IV (e.g. 0.15 = 15%% lower)")
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage multiplier")
    parser.add_argument("--margin", default="coin", choices=["coin", "usd"],
                        help="Margin mode: coin (Deribit native) or usd")
    parser.add_argument("--wing-delta", type=float, default=0.0,
                        help="Wing delta for iron condor (0 = no wings / strangle)")
    parser.add_argument("--expire-day", default="sunday",
                        choices=["saturday", "sunday", "monday"],
                        help="Expiry day (default: sunday)")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-trade output")

    args = parser.parse_args()

    run_backtest(
        data_dir=args.data_dir,
        start_date=args.start,
        end_date=args.end,
        target_delta=args.delta,
        max_delta_diff=args.max_diff,
        slippage=args.slippage,
        iv_discount=args.iv_discount,
        leverage=args.leverage,
        margin_mode=args.margin,
        wing_delta=args.wing_delta,
        expire_day=args.expire_day,
        verbose=not args.quiet,
    )
