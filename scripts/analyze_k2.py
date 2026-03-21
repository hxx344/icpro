"""Analyze optimal K2 strike for Call Spread CC."""
import sys
sys.path.insert(0, "monitor")
from exchanges import fetch_deribit

q = fetch_deribit("ETH")
spot = [x for x in q if x.bid_usd > 0][0].underlying_price

calls_today = sorted(
    [x for x in q if x.dte < 1 and x.option_type == "call" and x.strike >= spot * 0.99],
    key=lambda x: x.strike,
)
calls_tmr = sorted(
    [x for x in q if 1 < x.dte < 2 and x.option_type == "call" and x.strike >= spot * 0.99],
    key=lambda x: x.strike,
)

print(f"Spot: ${spot:.0f}")


def analyze(calls, label):
    atm = min(calls, key=lambda x: abs(x.strike - spot))
    K1 = atm.strike
    prem = atm.bid_usd

    print(f"\n{'='*60}")
    print(f"{label}  |  Short K1={K1:.0f} (ATM)  |  Receive ${prem:.2f}")
    print(f"{'='*60}")
    print(
        f"{'K2':>7}  {'OTM%':>6}  {'Ask$':>7}  {'Cost%':>6}  {'NetPrem':>8}  "
        f"{'MaxCoinLoss':>12}  {'Status':>8}"
    )
    print("-" * 72)

    for c in calls:
        if c.strike <= K1:
            continue
        otm = (c.strike / spot - 1) * 100
        cost = c.ask_usd
        if cost <= 0:
            continue
        cost_pct = cost / prem * 100
        net = prem - cost
        # Max coin loss happens at price = K2 (worst point between K1 and K2)
        max_coin_loss = (c.strike - K1) / c.strike
        net_coin = net / spot
        # Is net premium still positive after buying protection?
        status = "OK" if net > 0 else "NET NEG"
        # Sweet spot indicator
        sweet = ""
        if 3 <= cost_pct <= 15 and net > 0:
            sweet = " <-- SWEET"
        
        print(
            f"${c.strike:>6.0f}  {otm:>+5.1f}%  ${cost:>6.2f}  {cost_pct:>5.1f}%  "
            f"${net:>7.2f}  {max_coin_loss:>10.4f} ETH  {status:>8}{sweet}"
        )


analyze(calls_today, "TODAY (0-DTE)")
analyze(calls_tmr, "TOMORROW (1-DTE)")
