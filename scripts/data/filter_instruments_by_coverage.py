"""Filter instruments to only keep those with data covering the backtest start date."""
import pandas as pd
import os

data_dir = "data/market_data/60"
backtest_start = pd.Timestamp("2025-12-13", tz="UTC")

instr = pd.read_parquet("data/instruments/eth_instruments.parquet")
print(f"Current instruments: {len(instr)}")

# Get unique expiry dates from instrument names
instr["expiry_str"] = instr["instrument_name"].apply(lambda x: x.split("-")[1])
expiries = sorted(instr["expiry_str"].unique())
print(f"Expiries: {expiries}")

# Check data coverage for each expiry
keep_expiries = []
for exp in expiries:
    exp_instr = instr[instr["expiry_str"] == exp]
    # Sample a few instruments to check data range
    sample = exp_instr.iloc[0]["instrument_name"]
    fpath = os.path.join(data_dir, f"{sample}.parquet")
    if os.path.exists(fpath):
        df = pd.read_parquet(fpath)
        data_start = df["timestamp"].min()
        data_end = df["timestamp"].max()
        covers = data_start <= backtest_start
        bars = len(df)
        print(f"  {exp}: {len(exp_instr)} instruments, data {data_start} -> {data_end} ({bars} bars), covers_start={covers}")
        if covers:
            keep_expiries.append(exp)
    else:
        print(f"  {exp}: NO DATA FILE for {sample}")

print(f"\nKeeping expiries with full coverage: {keep_expiries}")

# Filter 
filtered = instr[instr["expiry_str"].isin(keep_expiries)].drop(columns=["expiry_str"])
print(f"Filtered: {len(instr)} -> {len(filtered)} instruments")

# Also include expiries with substantial coverage (data start before Feb 15, 2026)
partial_cutoff = pd.Timestamp("2026-02-15", tz="UTC")
keep_partial = []
for exp in expiries:
    if exp in keep_expiries:
        continue
    exp_instr = instr[instr["expiry_str"] == exp]
    sample = exp_instr.iloc[0]["instrument_name"]
    fpath = os.path.join(data_dir, f"{sample}.parquet")
    if os.path.exists(fpath):
        df = pd.read_parquet(fpath)
        data_start = df["timestamp"].min()
        if data_start <= partial_cutoff:
            keep_partial.append(exp)
            print(f"  Also keeping {exp} (partial coverage from {data_start})")

all_keep = keep_expiries + keep_partial
final = instr[instr["expiry_str"].isin(all_keep)].drop(columns=["expiry_str"])
print(f"\nFinal with partial: {len(final)} instruments")
print(f"Expiries kept: {all_keep}")

# Save
final.to_parquet("data/instruments/eth_instruments.parquet", index=False)
print(f"\nSaved filtered instruments ({len(final)}) to eth_instruments.parquet")
