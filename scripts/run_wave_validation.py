"""已知疫情波次回溯验证。

检查已知的美国COVID-19疫情波次期间，异常检测器是否捕获了事件，
以及这些事件的标签分布，用于回应epidemic标签缺失的质疑。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from loguru import logger

from src.config import PROJECT_ROOT

# 美国已知COVID-19疫情波次
KNOWN_WAVES = {
    "Winter 2020-21 (Alpha)": ("2020-11-01", "2021-02-28"),
    "Summer 2021 (Delta)": ("2021-07-01", "2021-10-31"),
    "Winter 2021-22 (Omicron BA.1)": ("2021-12-01", "2022-02-28"),
    "Summer 2022 (BA.4/BA.5)": ("2022-05-15", "2022-08-31"),
    "Winter 2022-23 (BQ/XBB)": ("2022-11-01", "2023-02-28"),
    "Winter 2023-24 (JN.1)": ("2023-11-01", "2024-02-29"),
}

HOSP_COL = "previous_day_admission_adult_covid_confirmed"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    db = pd.read_parquet(PROJECT_ROOT / "data/processed/merged_multisource.parquet")
    db["date"] = pd.to_datetime(db["date"])

    auto = pd.read_csv(PROJECT_ROOT / "data/labeled/auto_labeled_events.csv")
    auto["peak_date"] = pd.to_datetime(auto["peak_date"])

    silver = pd.read_csv(PROJECT_ROOT / "data/labeled/labeled_events.csv")
    silver["peak_date"] = pd.to_datetime(silver["peak_date"])

    inv_path = PROJECT_ROOT / "outputs/investigation_results.csv"
    inv = pd.read_csv(inv_path) if inv_path.exists() else pd.DataFrame()
    if "anomaly_date" in inv.columns:
        inv["anomaly_date"] = pd.to_datetime(inv["anomaly_date"])

    return db, auto, silver, inv


def wave_coverage(db: pd.DataFrame) -> pd.DataFrame:
    """每个波次的数据覆盖情况。"""
    rows = []
    conc_col = "pcr_conc_lin_log1p"
    for wave, (start, end) in KNOWN_WAVES.items():
        mask = (db["date"] >= start) & (db["date"] <= end)
        w = db[mask]
        rows.append({
            "wave": wave,
            "start": start,
            "end": end,
            "n_sites": w["site_id"].nunique(),
            "n_rows": len(w),
            "n_with_conc": int(w[conc_col].notna().sum()) if conc_col in w.columns else 0,
            "avg_hosp": float(w[HOSP_COL].mean()) if HOSP_COL in w.columns else None,
            "peak_hosp": float(w[HOSP_COL].max()) if HOSP_COL in w.columns else None,
        })
    return pd.DataFrame(rows)


def wave_events(
    auto: pd.DataFrame, silver: pd.DataFrame, inv: pd.DataFrame,
) -> pd.DataFrame:
    """每个波次内的事件及其标签。"""
    rows = []
    for wave, (start, end) in KNOWN_WAVES.items():
        mask_a = (auto["peak_date"] >= start) & (auto["peak_date"] <= end)
        wave_auto = auto[mask_a]

        for _, ev in wave_auto.iterrows():
            eid = ev["event_id"]
            silver_row = silver[silver["event_id"] == eid]
            silver_label = silver_row["ground_truth_label"].iloc[0] if len(silver_row) else None

            agent_label = None
            if len(inv) and "event_id" in inv.columns:
                inv_row = inv[inv["event_id"] == eid]
                agent_label = inv_row["classification"].iloc[0] if len(inv_row) else None

            rows.append({
                "wave": wave,
                "event_id": eid,
                "site_id": ev["site_id"],
                "peak_date": ev["peak_date"].date(),
                "auto_label": ev["auto_label"],
                "silver_label": silver_label,
                "agent_label": agent_label,
            })
    return pd.DataFrame(rows)


def wave_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    """每个波次的标签分布汇总。"""
    rows = []
    for wave in KNOWN_WAVES:
        w = events_df[events_df["wave"] == wave]
        n = len(w)
        auto_epi = int((w["auto_label"] == "epidemic").sum())
        silver_epi = int((w["silver_label"] == "epidemic").sum())
        agent_epi = int((w["agent_label"] == "epidemic").sum())
        rows.append({
            "wave": wave,
            "total_events": n,
            "auto_epidemic": auto_epi,
            "silver_epidemic": silver_epi,
            "agent_epidemic": agent_epi,
            "auto_labels": dict(w["auto_label"].value_counts()) if n else {},
            "silver_labels": dict(w["silver_label"].value_counts()) if n else {},
        })
    return pd.DataFrame(rows)


def hospitalization_validation(
    db: pd.DataFrame, auto: pd.DataFrame,
) -> pd.DataFrame:
    """验证auto-epidemic事件的住院数据变化。"""
    epi = auto[auto["auto_label"] == "epidemic"].copy()
    rows = []
    for _, ev in epi.iterrows():
        site_id = ev["site_id"]
        peak = ev["peak_date"]
        site_rows = db[db["site_id"] == site_id]
        state = "?"
        if "state" in site_rows.columns and not site_rows["state"].dropna().empty:
            state = str(site_rows["state"].dropna().iloc[0])

        pre = db[
            (db["state"] == state)
            & (db["date"] >= peak - pd.Timedelta(days=7))
            & (db["date"] <= peak)
        ][HOSP_COL].dropna()
        post = db[
            (db["state"] == state)
            & (db["date"] > peak)
            & (db["date"] <= peak + pd.Timedelta(days=14))
        ][HOSP_COL].dropna()

        pre_mean = float(pre.mean()) if len(pre) else None
        post_mean = float(post.mean()) if len(post) else None
        change_pct = (
            (post_mean - pre_mean) / pre_mean * 100
            if pre_mean and pre_mean > 0 and post_mean is not None
            else None
        )

        rows.append({
            "event_id": ev["event_id"],
            "site_id": site_id,
            "state": state,
            "peak_date": peak.date(),
            "pre_hosp_mean": round(pre_mean, 1) if pre_mean else None,
            "post_hosp_mean": round(post_mean, 1) if post_mean else None,
            "hosp_change_pct": round(change_pct, 1) if change_pct else None,
            "confirms_epidemic": bool(change_pct and change_pct > 20),
        })
    return pd.DataFrame(rows)


def main() -> None:
    logger.info("Loading data...")
    db, auto, silver, inv = load_data()

    out_dir = PROJECT_ROOT / "outputs" / "wave_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Computing wave coverage...")
    cov = wave_coverage(db)
    cov.to_csv(out_dir / "wave_coverage.csv", index=False)
    logger.info(f"\n{cov.to_string()}")

    logger.info("Mapping events to waves...")
    ev = wave_events(auto, silver, inv)
    ev.to_csv(out_dir / "wave_events.csv", index=False)

    logger.info("Summarizing wave labels...")
    summary = wave_summary(ev)
    summary.to_csv(out_dir / "wave_summary.csv", index=False)
    logger.info(f"\n{summary[['wave','total_events','auto_epidemic','silver_epidemic','agent_epidemic']].to_string()}")

    logger.info("Validating hospitalization for auto-epidemic events...")
    hosp = hospitalization_validation(db, auto)
    hosp.to_csv(out_dir / "hosp_validation.csv", index=False)
    logger.info(f"\n{hosp.to_string()}")

    confirmed = hosp["confirms_epidemic"].sum()
    total = len(hosp)
    logger.info(f"\nHospitalization confirms epidemic: {confirmed}/{total} ({confirmed/total*100:.0f}%)")
    logger.info(f"All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
