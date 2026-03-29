"""Agent 工具基类。"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseTool(ABC):
    """
    Agent 工具基类。

    所有工具需实现：
    - name: 工具名称（Agent 调用时使用）
    - description: 工具描述（写入 system prompt）
    - parameters_schema: 参数 JSON Schema
    - execute(): 执行查询并返回结构化结果
    """

    name: str = ""
    description: str = ""
    parameters_schema: dict = {}

    def __init__(self, database: pd.DataFrame | None = None):
        """
        Parameters
        ----------
        database : 合并后的多源数据库，工具从中查询数据
        """
        self.db = database

    @abstractmethod
    def execute(self, **kwargs) -> dict[str, Any]:
        """
        执行工具查询。

        Returns
        -------
        dict : 结构化查询结果
        """
        ...

    def to_openai_function(self) -> dict:
        """转换为 OpenAI function calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
