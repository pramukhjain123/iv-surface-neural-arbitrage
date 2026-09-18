"""
03_riskfree.py -- Phase 1.3, risk-free rate.

A genuine daily 91-day T-bill yield series could not be obtained as a free,
downloadable CSV within this session (RBI DBIE / FBIL don't expose one
easily; CEIC/Moody's/Indiastat/Investing.com gate the history behind
logins or short free windows). Per the plan's own note that this "barely
moves the IVs... not a sensitivity worth agonising over", we use the RBI
repo rate as a documented, publicly verifiable step-function proxy instead
of an arbitrary flat placeholder (the sibling project's skew script uses a
flat 6.5%, which is wrong outside 2023-2024). The 91-day T-bill yield
tracks repo closely (typically within a few tens of bps); replace this
with a real T-bill series later if the discounting sensitivity turns out
to matter more than expected (Phase 4 should include a robustness check
that swaps this out).

Source: RBI Monetary Policy Committee announcements, compiled at
https://www.basunivesh.com/rbi-repo-rate-history-from-2000/ (accessed
2026-08-27), cross-checked against well-known MPC decision dates
(including the 4 May 2022 off-cycle hike).
"""
from pathlib import Path
import pandas as pd

WINDOWS_ROOT = Path(r"E:\Research")
BRIDGE_ROOT = Path.home() / "mnt" / "Research"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT
OUT = ROOT / "iv_surface" / "02_parsed"
OUT.mkdir(parents=True, exist_ok=True)

# (effective_date, repo_rate_pct) -- only rows where the rate actually changed
CHANGES = [
    ("2019-10-04", 5.20),
    ("2020-03-27", 4.40),
    ("2020-05-22", 4.00),
    ("2022-05-04", 4.40),   # off-cycle hike
    ("2022-06-08", 4.90),
    ("2022-08-05", 5.40),
    ("2022-09-30", 5.90),
    ("2022-12-07", 6.25),
    ("2023-02-08", 6.50),
    ("2025-02-07", 6.25),
    ("2025-04-09", 6.00),
    ("2025-06-06", 5.50),
    ("2025-12-05", 5.25),
]

def build(start="2020-01-01", end="2026-01-30") -> pd.DataFrame:
    chg = pd.DataFrame(CHANGES, columns=["date", "repo_rate_pct"])
    chg["date"] = pd.to_datetime(chg["date"])
    idx = pd.date_range(start, end, freq="D")
    daily = pd.DataFrame({"date": idx})
    daily = daily.merge(chg, on="date", how="left").sort_values("date")
    daily["repo_rate_pct"] = daily["repo_rate_pct"].ffill()
    # first values before 2020-01-01's most recent change (04-Oct-2019, 5.20%)
    daily["repo_rate_pct"] = daily["repo_rate_pct"].fillna(5.20)
    daily["r"] = daily["repo_rate_pct"] / 100.0
    return daily[["date", "r"]]

if __name__ == "__main__":
    df = build()
    df.to_csv(OUT / "riskfree_daily.csv", index=False)
    print(df.head())
    print(df.tail())
    print(f"{len(df)} daily rows, {df['r'].nunique()} distinct rate levels")
