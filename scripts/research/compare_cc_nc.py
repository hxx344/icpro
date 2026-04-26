"""ETH Covered Call vs Naked Call 对比回测脚本"""

import subprocess, json, sys, re

CONFIGS = {
    "CC_ETH": "configs/backtest/cc_eth_compare.yaml",
    "NC_ETH": "configs/backtest/nc_eth_compare.yaml",
}

results = {}

for label, cfg in CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"  Running: {label}  ({cfg})")
    print(f"{'='*60}")
    proc = subprocess.run(
        [sys.executable, "-m", "options_backtest.cli", "run", "--config", cfg],
        capture_output=True, text=True, cwd="."
    )
    output = proc.stdout + proc.stderr
    print(output[-3000:] if len(output) > 3000 else output)

    # Parse metrics from output �?use ordered keys to handle substring overlap
    metrics = {}
    ordered_keys = [
        "Total Return (Hedged USD)", "Annualized Return (Hedged USD)",
        "Max Drawdown (Hedged USD)", "Sharpe (Hedged USD)",
        "Total Return (BTC)", "Annualized Return (BTC)",
        "Max Drawdown (BTC)", "Sharpe (BTC)",
        "Total Return (USD)", "Annualized Return (USD)",
        "Max Drawdown (USD)", "Sharpe (USD)",
        "Total Return", "Annualized Return", "Sharpe", "Max Drawdown",
        "Win Rate", "Profit Factor", "Total Trades",
    ]
    for line in output.splitlines():
        stripped = line.strip()
        for key in ordered_keys:
            if key in metrics:
                continue
            # Exact prefix match to avoid substring collision
            if stripped.startswith(key):
                m = re.search(r"[-+]?\d+\.?\d*", stripped[len(key):])
                if m:
                    metrics[key] = float(m.group())
                break
    results[label] = metrics

print(f"\n\n{'='*70}")
print("  ETH Covered Call vs Naked Call 对比")
print(f"{'='*70}")

header = f"{'Metric':<35} {'CC_ETH':>15} {'NC_ETH':>15}"
print(header)
print("-" * len(header))

all_keys = []
for m in results.values():
    for k in m:
        if k not in all_keys:
            all_keys.append(k)

for key in all_keys:
    cc_val = results.get("CC_ETH", {}).get(key, "N/A")
    nc_val = results.get("NC_ETH", {}).get(key, "N/A")
    cc_str = f"{cc_val:.2f}" if isinstance(cc_val, float) else str(cc_val)
    nc_str = f"{nc_val:.2f}" if isinstance(nc_val, float) else str(nc_val)
    print(f"{key:<35} {cc_str:>15} {nc_str:>15}")

print()
