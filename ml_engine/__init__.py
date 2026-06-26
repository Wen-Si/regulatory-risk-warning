#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智鉴风控 - ML Engine
融合深度学习与强化学习的监管问询预测引擎

算法架构：
1. 深度学习层：DeepFM + Temporal Transformer + GAT + FinBERT文本编码器
2. 强化学习层：PPO自适应阈值优化 + Thompson Sampling集成权重学习
3. 混合层：规则引擎硬约束 + DL/RL软评分 + LLM可解释性
"""

from .predictor import HybridPredictor
from .config import MLConfig

__all__ = ['HybridPredictor', 'MLConfig']
