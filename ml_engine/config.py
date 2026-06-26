#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Engine Configuration
深度学习与强化学习模型配置
"""
import os
import numpy as np


class MLConfig:
    """机器学习引擎全局配置"""
    
    # ==================== 路径配置 ====================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'checkpoints')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    
    # ==================== 特征配置 ====================
    # 数值型财务特征维度
    NUMERIC_FEATURE_DIM = 24
    # 类别型特征
    CATEGORICAL_FEATURES = {
        'industry': 32,      # 行业类别embedding维度
        'market': 8,         # 交易所类别embedding维度
        'company_type': 4,   # 公司类型(ST/正常)embedding维度
    }
    # 文本特征维度（FinBERT输出）
    TEXT_EMBED_DIM = 768
    # 图特征维度
    GRAPH_FEATURE_DIM = 32
    # 时序窗口长度（季度）
    TEMPORAL_WINDOW = 8
    
    # ==================== DeepFM配置 ====================
    DEEPFM_HIDDEN_DIMS = [128, 64, 32]
    DEEPFM_DROPOUT = 0.3
    DEEPFM_EMBED_DIM = 16
    DEEPFM_NUM_FIELDS = 14
    
    # ==================== Temporal Transformer配置 ====================
    TEMPORAL_D_MODEL = 64
    TEMPORAL_NHEAD = 4
    TEMPORAL_NUM_LAYERS = 2
    TEMPORAL_DIM_FF = 128
    TEMPORAL_DROPOUT = 0.2
    
    # ==================== GAT图注意力配置 ====================
    GAT_HIDDEN_DIM = 32
    GAT_NUM_HEADS = 4
    GAT_NUM_LAYERS = 2
    GAT_DROPOUT = 0.2
    
    # ==================== FinBERT文本编码器配置 ====================
    FINBERT_HIDDEN_DIM = 256
    FINBERT_MAX_SEQ_LEN = 512
    FINBERT_NUM_LAYERS = 4
    FINBERT_NUM_HEADS = 8
    
    # ==================== PPO强化学习配置 ====================
    PPO_STATE_DIM = 64        # 状态维度（融合特征表示）
    PPO_ACTION_DIM = 3        # 动作维度（30/60/90天阈值）
    PPO_HIDDEN_DIM = 64
    PPO_LR = 3e-4
    PPO_GAMMA = 0.99
    PPO_GAE_LAMBDA = 0.95
    PPO_CLIP_EPSILON = 0.2
    PPO_ENTROPY_COEF = 0.01
    PPO_VALUE_COEF = 0.5
    PPO_MAX_GRAD_NORM = 0.5
    PPO_BUFFER_SIZE = 2048
    PPO_BATCH_SIZE = 64
    PPO_EPOCHS = 10
    
    # ==================== Thompson Sampling集成配置 ====================
    TS_NUM_MODELS = 4         # 集成模型数量
    TS_PRIOR_ALPHA = 1.0      # Beta分布先验alpha
    TS_PRIOR_BETA = 1.0       # Beta分布先验beta
    TS_EXPLORATION_DECAY = 0.995  # 探索率衰减
    
    # ==================== 训练配置 ====================
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 100
    EARLY_STOP_PATIENCE = 10
    DEVICE = 'cpu'  # 默认CPU，有GPU时自动切换
    
    # ==================== 预测窗口配置 ====================
    PREDICTION_WINDOWS = [30, 60, 90]
    RISK_THRESHOLDS = {'low': 0.3, 'medium': 0.6, 'high': 0.8}
    
    # ==================== 在线学习配置 ====================
    ONLINE_LEARNING_BATCH = 32    # 累积多少反馈后更新
    ONLINE_LEARNING_LR = 5e-4     # 在线学习率
    FEEDBACK_BUFFER_SIZE = 1000   # 反馈缓存大小
    
    @classmethod
    def ensure_dirs(cls):
        for d in [cls.MODEL_DIR, cls.LOG_DIR]:
            os.makedirs(d, exist_ok=True)
