"""Compare ETH Covered Call vs Naked Call over a 12-month window."""
import subprocess, sys, re

CONFIGS = {
    "CC_12m": {
        "desc": "Covered Call (12个月)",
        "cfg": "configs/backtest/cc_eth_12m.yaml",
    },
    "NC_12m": {
        "desc": "Naked Call (12个月)",
        "cfg": "configs/backtest/nc_eth_12m.yaml",
    },
}

results = {}

for label, info in CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"  Running: {info['desc']}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, "-m", "options_backtest.cli", "run", "--config", info["cfg"]]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    out = proc.stdout + proc.stderr
    
    # Extract metrics
    m = {}
    ordered_keys = [
        "Total Return (Hedged USD)", "Annualized Return (Hedged USD)",
        "Max Drawdown (Hedged USD)", "Sharpe (Hedged USD)",
        "Total Return USD", "Ann. Return USD", "Max Drawdown USD", "Sharpe Ratio USD",
        "Return Hedged", "Ann. Return Hedged", "Sharpe Hedged",
        "Total Return", "Annualised Return", "Max Drawdown", "Sharpe Ratio",
        "Win Rate", "Profit Factor", "Total Trades", "Avg Win",
    ]
    for line in out.splitlines():
        s = line.strip()
        for key in ordered_keys:
            if key in m:
                continue
            if s.startswith(key):
                v = re.search(r"[-+]?\d+\.?\d*", s[len(key):])
                if v:
                    m[key] = float(v.group())
                break
    
    results[label] = m
    # Print raw summary
    for line in out.splitlines():
        s = line.strip()
        if any(s.startswith(k) for k in ordered_keys):
            print(f"  {s}")

# Print comparison table
print(f"\n\n{'='*70}")
print("  ETH Covered Call vs Naked Call 12-month comparison (2025-03 ~ 2026-03)")
print(f"{'='*70}")

hdr = f"{'指标':<35} {'CC (备兑)':>15} {'NC (裸卖)':>15}"
print(hdr)
print("-" * 65)

display = [
    ("Base total return", "Total Return", "%"),
    ("Base annualized return", "Annualised Return", "%"),
    ("Base max drawdown", "Max Drawdown", "%"),
    ("Base Sharpe", "Sharpe Ratio", ""),
    ("胜率", "Win Rate", "%"),
    ("Profit Factor", "Profit Factor", ""),
    ("总交易数", "Total Trades", ""),
    ("USD total return", "Total Return USD", "%"),
    ("USD 年化", "Ann. Return USD", "%"),
    ("USD max drawdown", "Max Drawdown USD", "%"),
    ("USD Sharpe", "Sharpe Ratio USD", ""),
    ("对冲USD回报", "Return Hedged", "%"),
    ("对冲USD年化", "Ann. Return Hedged", "%"),
    ("对冲USD Sharpe", "Sharpe Hedged", ""),
]

for label_cn, key, unit in display:
    cc = results["CC_12m"].get(key, "N/A")
    nc = results["NC_12m"].get(key, "N/A")
    cc_s = f"{cc:.2f}{unit}" if isinstance(cc, float) else "N/A"
    nc_s = f"{nc:.2f}{unit}" if isinstance(nc, float) else "N/A"
    print(f"{label_cn:<35} {cc_s:>15} {nc_s:>15}")

print()
