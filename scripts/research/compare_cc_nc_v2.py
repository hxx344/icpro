"""Quick comparison script for CC vs NC."""
import subprocess, sys, re

configs = {"CC": "configs/backtest/cc_eth_compare.yaml", "NC": "configs/backtest/nc_eth_compare.yaml"}
results = {}

for label, cfg in configs.items():
    proc = subprocess.run(
        [sys.executable, "-m", "options_backtest.cli", "run", "--config", cfg],
        capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    m = {}
    for line in out.splitlines():
        s = line.strip()
        for key in [
            "Total Return (Hedged USD)", "Annualized Return (Hedged USD)",
            "Max Drawdown (Hedged USD)", "Sharpe (Hedged USD)",
            "Total Return", "Annualized Return", "Max Drawdown", "Sharpe",
            "Win Rate", "Profit Factor", "Total Trades",
        ]:
            if s.startswith(key) and key not in m:
                v = re.search(r"[-+]?\d+\.?\d*", s[len(key):])
                if v:
                    m[key] = float(v.group())
                break
    results[label] = m

hdr = f"{'Metric':<35} {'CC (备兑)':>15} {'NC (裸卖)':>15}"
print()
print(hdr)
print("-" * 65)

display_keys = [
    "Total Return", "Annualized Return", "Max Drawdown", "Sharpe",
    "Win Rate", "Profit Factor", "Total Trades",
    "Total Return (Hedged USD)", "Annualized Return (Hedged USD)",
    "Max Drawdown (Hedged USD)", "Sharpe (Hedged USD)",
]

for k in display_keys:
    cc = results["CC"].get(k, "N/A")
    nc = results["NC"].get(k, "N/A")
    cc_s = f"{cc:.2f}" if isinstance(cc, float) else cc
    nc_s = f"{nc:.2f}" if isinstance(nc, float) else nc
    unit = "%" if ("Return" in k or "Drawdown" in k or "Rate" in k) else ""
    print(f"{k:<35} {cc_s:>14}{unit}  {nc_s:>14}{unit}")

print()
