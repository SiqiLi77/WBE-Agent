"""
Agent 工具集。

每个工具封装对一个数据源的查询逻辑，
接收结构化参数，返回结构化结果。

工具列表：
  - weather        : 查询气象数据（降水、气温）
  - hydrology      : 查询水文数据（流量、百分位）
  - hospitalization : 查询临床住院数据
  - nearby_sites   : 查询周边站点信号
  - variants       : 查询变异株数据
  - site_metadata  : 查询站点元数据
"""

from src.agent.tools.weather import WeatherTool
from src.agent.tools.hydrology import HydrologyTool
from src.agent.tools.hospitalization import HospitalizationTool
from src.agent.tools.nearby_sites import NearbySitesTool
from src.agent.tools.variants import VariantsTool
from src.agent.tools.site_metadata import SiteMetadataTool

ALL_TOOLS = [
    WeatherTool,
    HydrologyTool,
    HospitalizationTool,
    NearbySitesTool,
    VariantsTool,
    SiteMetadataTool,
]

__all__ = [
    "WeatherTool",
    "HydrologyTool",
    "HospitalizationTool",
    "NearbySitesTool",
    "VariantsTool",
    "SiteMetadataTool",
    "ALL_TOOLS",
]
