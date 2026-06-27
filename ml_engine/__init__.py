#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智鉴风控 - ML Engine v3.0
融合深度学习、强化学习、知识图谱与Harness Engineering的监管问询预测引擎

算法架构（v3.0）：
1. Harness Engineering安全层：输入护栏 + 输出护栏 + 审计日志 + 工具防火墙
2. 知识图谱层：监管金融KG + Graph RAG + 风险传播推理 + 证据链构建
3. 深度学习层：DeepFM + Temporal Transformer + GAT + RiskTextEncoder
4. 强化学习层：PPO自适应阈值优化 + Thompson Sampling集成权重学习
5. 混合层：规则引擎硬约束 + KG增强 + DL/RL软评分 + 安全合规校验
"""

from .predictor import HybridPredictor
from .config import MLConfig
from .safety import SafetyHarness, GuardrailResult
from .knowledge_graph import RegulatoryKG, GraphRAG

__all__ = ['HybridPredictor', 'MLConfig', 'SafetyHarness', 'RegulatoryKG', 'GraphRAG']
