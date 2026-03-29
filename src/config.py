"""
中央配置管理模块。

从 config/settings.yaml 加载所有配置项，提供类型安全的访问接口。
所有其他模块通过 `from src.config import settings` 获取配置。
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ── 项目根目录 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


# ── Pydantic 配置模型 ──────────────────────────────────────
class PathsConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    labeled_data_dir: str = "data/labeled"
    outputs_dir: str = "outputs"
    legacy_outputs_dir: str = "D:/agent/outputs"
    nwss_raw_csv: str = ""
    hhs_raw_csv: str = ""
    ghcn_daily_dir: str = ""
    usgs_cache_dir: str = ""
    variants_raw_csv: str = ""
    nwss_processed: str = ""
    hhs_processed: str = ""
    noaa_processed: str = ""
    usgs_processed: str = ""
    variants_processed: str = ""
    merged_database: str = ""
    tier1_sites: str = ""
    tier2_sites: str = ""
    tier3_sites: str = ""
    hhs_overlap: str = ""

    def resolve(self, key: str) -> Path:
        """将相对路径解析为绝对路径。"""
        return PROJECT_ROOT / getattr(self, key)


class NWSSConfig(BaseModel):
    normalization_filter: str = "flow-population"
    duplicate_strategy: str = "mean"
    zero_handling: str = "lod_half"
    default_lod: float = 100.0
    rolling_windows: list[int] = Field(default_factory=lambda: [7, 14])
    rolling_min_periods_ratio: float = 0.57


class SpatialMatchingConfig(BaseModel):
    noaa_max_distance_km: float = 50.0
    noaa_min_coverage: float = 0.80
    usgs_max_distance_km: float = 30.0
    usgs_fallback_to_precipitation: bool = True


class AnomalyMethodConfig(BaseModel):
    enabled: bool = True
    window: int | None = None
    period: int | None = None
    threshold: float | None = None
    penalty: str | None = None
    drift: float | None = None


class AnomalyDetectionConfig(BaseModel):
    methods: dict[str, AnomalyMethodConfig] = Field(default_factory=dict)
    ensemble_min_votes: int = 2
    event_merge_gap_days: int = 3


class DomainKnowledgeConfig(BaseModel):
    rainfall_dilution_threshold_mm: float = 10.0
    rainfall_significant_mm: float = 25.0
    temp_high_degradation_c: float = 25.0
    temp_low_stable_c: float = 10.0
    cso_flow_percentile: float = 90.0
    clinical_followup_threshold_pct: float = 20.0


class AgentConfig(BaseModel):
    model: str = "google/gemini-2.5-flash"
    temperature: float = 0.0
    max_tool_calls: int = 10
    api_base: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    nearby_sites_radius_km: float = 100.0
    clinical_lag_days: int = 14
    output_format: str = "json"
    domain_knowledge: DomainKnowledgeConfig = Field(
        default_factory=DomainKnowledgeConfig
    )


class EvaluationConfig(BaseModel):
    tier2_target_events: int = 50
    event_balance: dict[str, list[int]] = Field(default_factory=dict)
    baselines: list[str] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "outputs/logs"
    log_agent_traces: bool = True


class Settings(BaseModel):
    """顶层配置对象。"""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    nwss: NWSSConfig = Field(default_factory=NWSSConfig)
    spatial_matching: SpatialMatchingConfig = Field(
        default_factory=SpatialMatchingConfig
    )
    anomaly_detection: AnomalyDetectionConfig = Field(
        default_factory=AnomalyDetectionConfig
    )
    agent: AgentConfig = Field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ── 加载配置 ───────────────────────────────────────────────
def load_settings(config_path: Path | None = None) -> Settings:
    """从 YAML 文件加载配置，不存在则使用默认值。"""
    path = config_path or CONFIG_PATH
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        # 处理 anomaly_detection.methods 的嵌套结构
        if "anomaly_detection" in raw and "methods" in raw["anomaly_detection"]:
            methods = {}
            for name, params in raw["anomaly_detection"]["methods"].items():
                methods[name] = AnomalyMethodConfig(**(params or {}))
            raw["anomaly_detection"]["methods"] = methods
        return Settings(**raw)
    return Settings()


# 全局单例
settings = load_settings()


def ensure_dirs() -> None:
    """确保所有输出目录存在。"""
    for dir_key in [
        "raw_data_dir",
        "processed_data_dir",
        "labeled_data_dir",
        "outputs_dir",
    ]:
        (PROJECT_ROOT / getattr(settings.paths, dir_key)).mkdir(
            parents=True, exist_ok=True
        )
    (PROJECT_ROOT / settings.logging.log_dir).mkdir(parents=True, exist_ok=True)
