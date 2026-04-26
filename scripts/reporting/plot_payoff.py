"""Generate Iron Condor 0DTE payoff diagram based on current config."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import math

# ---- Current Iron Condor 0DTE parameters ----
spot = 2000           # ETH price assumption
otm_pct = 0.01        # 1% OTM
wing_width_pct = 0.08 # 8% wing width
equity = 23402
max_capital_pct = 0.30

# Strike prices
short_call_K = spot * (1 + otm_pct)                        # 2020
long_call_K  = spot * (1 + otm_pct + wing_width_pct)       # 2180
short_put_K  = spot * (1 - otm_pct)                        # 1980
long_put_K   = spot * (1 - otm_pct - wing_width_pct)       # 1820

# Quantity from compound formula
wing_width = wing_width_pct * spot   # 160
margin_per_condor = wing_width * 2   # 320
qty = math.floor(equity * max_capital_pct / margin_per_condor * 100) / 100  # 21

# Estimated net premium received per condor (conservative for 0DTE)
premium_per_condor = 8.0  # USDT

print(f"Short Call K: {short_call_K}")
print(f"Long Call K:  {long_call_K}")
print(f"Short Put K:  {short_put_K}")
print(f"Long Put K:   {long_put_K}")
print(f"Wing width:   {wing_width}")
print(f"Quantity:     {qty}")

# ---- Payoff calculation ----
prices = np.linspace(1600, 2400, 1000)

def payoff(S):
    sc = -np.maximum(S - short_call_K, 0)    # short call
    lc = np.maximum(S - long_call_K, 0)      # long call
    sp = -np.maximum(short_put_K - S, 0)     # short put
    lp = np.maximum(long_put_K - S, 0)       # long put
    return (sc + lc + sp + lp + premium_per_condor) * qty

pnl = payoff(prices)

max_profit = premium_per_condor * qty
max_loss = (wing_width - premium_per_condor) * qty
be_upper = short_call_K + premium_per_condor
be_lower = short_put_K - premium_per_condor

print(f"Max Profit: ${max_profit:.0f}")
print(f"Max Loss:   ${max_loss:.0f} ({max_loss/equity*100:.1f}% equity)")

# ---- Plot ----
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(prices, pnl, "b-", linewidth=2, label="Iron Condor P&L")
ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
ax.axvline(x=spot, color="orange", linewidth=1.2, linestyle="--", alpha=0.7,
           label=f"Spot = ${spot}")

# Shade profit/loss
ax.fill_between(prices, pnl, 0, where=(pnl > 0), alpha=0.15, color="green")
ax.fill_between(prices, pnl, 0, where=(pnl < 0), alpha=0.15, color="red")

# Mark strikes
for k, lbl, clr in [
    (short_put_K,  f"Short Put\n${short_put_K:.0f}",  "red"),
    (long_put_K,   f"Long Put\n${long_put_K:.0f}",    "green"),
    (short_call_K, f"Short Call\n${short_call_K:.0f}", "red"),
    (long_call_K,  f"Long Call\n${long_call_K:.0f}",   "green"),
]:
    ax.axvline(x=k, color=clr, linewidth=0.8, linestyle=":", alpha=0.5)
    ax.text(k, max_profit * 1.15, lbl, ha="center", fontsize=8, color=clr)

# Annotations
ax.annotate(f"Max Profit: ${max_profit:.0f}",
            xy=(spot, max_profit), xytext=(spot, max_profit + 200),
            fontsize=10, ha="center",
            arrowprops=dict(arrowstyle="->", color="green"),
            color="green", fontweight="bold")

ax.annotate(f"Max Loss: -${max_loss:.0f}\n({max_loss/equity*100:.1f}% equity)",
            xy=(long_call_K, -max_loss),
            xytext=(long_call_K + 50, -max_loss - 300),
            fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="red"), color="red")

ax.annotate(f"Max Loss: -${max_loss:.0f}\n({max_loss/equity*100:.1f}% equity)",
            xy=(long_put_K, -max_loss),
            xytext=(long_put_K - 50, -max_loss - 300),
            fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="red"), color="red")

# Breakeven
ax.axvline(x=be_upper, color="purple", linewidth=0.8, linestyle="-.", alpha=0.5)
ax.axvline(x=be_lower, color="purple", linewidth=0.8, linestyle="-.", alpha=0.5)
ax.text(be_upper, -500, f"BE ${be_upper:.0f}", ha="center", fontsize=8, color="purple")
ax.text(be_lower, -500, f"BE ${be_lower:.0f}", ha="center", fontsize=8, color="purple")

# Labels
ax.set_xlabel("ETH Price at Expiration ($)", fontsize=12)
ax.set_ylabel("Profit / Loss ($)", fontsize=12)
ax.set_title(
    f"Iron Condor 0DTE Payoff  |  ETH Spot=${spot}  |  Qty={qty:.0f}  |  "
    f"Equity=${equity:,}\n"
    f"otm={otm_pct*100:.0f}%  wing={wing_width_pct*100:.0f}%  "
    f"max_capital={max_capital_pct*100:.0f}%  "
    f"premium=${premium_per_condor}/condor (est.)",
    fontsize=11,
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(1600, 2400)

# Summary box
summary = (
    f"Strikes: {long_put_K:.0f} / {short_put_K:.0f} / "
    f"{short_call_K:.0f} / {long_call_K:.0f}\n"
    f"Qty: {qty:.0f} contracts\n"
    f"Max Profit: ${max_profit:.0f} ({max_profit/equity*100:.1f}% equity)\n"
    f"Max Loss:   ${max_loss:.0f} ({max_loss/equity*100:.1f}% equity)\n"
    f"Margin Used: ${margin_per_condor*qty:,.0f} "
    f"({margin_per_condor*qty/equity*100:.0f}% equity)\n"
    f"Leverage: {spot*qty/equity:.2f}x (single-side notional)"
)
ax.text(0.02, 0.02, summary, transform=ax.transAxes, fontsize=8.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        family="monospace")

plt.tight_layout()
out = "reports/iron_condor_payoff.png"
plt.savefig(out, dpi=150)
print(f"\nSaved to {out}")
