"""
Agent 输出 Schema 定义。

使用 Pydantic 模型确保 Agent 输出的结构化和可验证性。
"""

from enum import Enum

from pydantic import BaseModel, Field


class AnomalyClassification(str, Enum):
    """异常归因分类。"""

    EPIDEMIC = "epidemic"
    ENVIRONMENTAL = "environmental"
    SAMPLING = "sampling"
    MIXED = "mixed"
    UNCERTAIN = "uncertain"


class ContributionLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceFactor(BaseModel):
    """单个证据因素。"""

    factor: str = Field(description="因素名称，如 heavy_rainfall, variant_emergence")
    contribution: ContributionLevel = Field(description="贡献程度")
    evidence: str = Field(description="具体数据引用和数值")


class ReasoningStep(BaseModel):
    """推理链中的一步。"""

    step: int
    thought: str
    action: str | None = None
    observation: str | None = None


class InvestigationReport(BaseModel):
    """Agent 调查报告（完整输出）。"""

    event_id: str
    site_id: str
    anomaly_date: str
    classification: AnomalyClassification
    confidence: float = Field(ge=0.0, le=1.0)
    primary_factors: list[EvidenceFactor]
    reasoning_chain: list[ReasoningStep]
    recommendation: str = Field(description="对公共卫生决策的建议")
    data_gaps: list[str] = Field(
        default_factory=list,
        description="调查中发现的数据缺失或局限",
    )
    tool_calls_count: int = 0
    total_tokens: int = 0

    def summary(self) -> str:
        """生成一行摘要。"""
        factors = ", ".join(f.factor for f in self.primary_factors)
        return (
            f"[{self.classification.value}] {self.site_id} @ {self.anomaly_date} "
            f"(confidence={self.confidence:.2f}, factors={factors})"
        )
