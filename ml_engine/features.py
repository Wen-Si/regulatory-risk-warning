#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Engineering Module
金融特征工程：数值特征归一化、类别特征编码、文本特征提取、图特征构建
"""
import numpy as np
import re
import hashlib
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict


class FeatureEngineer:
    """多模态特征工程器"""
    
    def __init__(self):
        # 行业映射到ID
        self.industry_map = {}
        self.market_map = {'上交所': 0, '深交所': 1, '上交所科创板': 2, '北交所': 3, '深交所创业板': 4}
        # 财务特征的统计参数（用于Z-score归一化）
        self.feature_stats = self._init_feature_stats()
        # 文本风险关键词词典
        self.risk_keywords = self._init_risk_keywords()
        # 行业相关性图（用于GAT）
        self.industry_graph = self._init_industry_graph()
        
    def _init_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """初始化财务特征统计参数（基于A股历史数据先验）"""
        return {
            'revenue_growth': {'mean': 12.0, 'std': 25.0, 'min': -50.0, 'max': 100.0},
            'profit_growth': {'mean': 10.0, 'std': 30.0, 'min': -80.0, 'max': 150.0},
            'roe': {'mean': 8.0, 'std': 8.0, 'min': -20.0, 'max': 30.0},
            'debt_ratio': {'mean': 48.0, 'std': 18.0, 'min': 10.0, 'max': 95.0},
            'account_receivable_turnover': {'mean': 8.0, 'std': 5.0, 'min': 0.5, 'max': 30.0},
            'inventory_turnover': {'mean': 5.0, 'std': 4.0, 'min': 0.2, 'max': 20.0},
            'operating_cashflow_ratio': {'mean': 1.0, 'std': 0.6, 'min': -1.0, 'max': 3.0},
            'goodwill_ratio': {'mean': 8.0, 'std': 12.0, 'min': 0.0, 'max': 60.0},
            'pledge_ratio': {'mean': 15.0, 'std': 18.0, 'min': 0.0, 'max': 90.0},
            'guarantee_ratio': {'mean': 8.0, 'std': 12.0, 'min': 0.0, 'max': 80.0},
        }
    
    def _init_risk_keywords(self) -> Dict[str, List[str]]:
        """初始化风险关键词词典（按风险类型分类）"""
        return {
            'financial_fraud': ['虚增', '造假', '财务造假', '虚增收入', '虚增利润', '伪造', '虚构'],
            'related_party': ['关联交易', '关联方', '利益输送', '非关联化', '资金占用'],
            'goodwill': ['商誉减值', '商誉', '减值测试', '业绩承诺', '业绩补偿'],
            'guarantee': ['担保', '违规担保', '对外担保', '连带担保'],
            'restructuring': ['重组', '重大资产重组', '并购', '收购', '资产置换'],
            'disclosure': ['信息披露', '补充披露', '更正', '遗漏', '虚假记载'],
            'pledge': ['股权质押', '质押', '平仓', '爆仓'],
            'cashflow': ['现金流', '经营现金流', '资金紧张', '债务违约'],
            'inventory': ['存货', '跌价准备', '积压', '滞销'],
            'receivable': ['应收账款', '坏账', '回款', '账龄'],
        }
    
    def _init_industry_graph(self) -> Dict[str, List[str]]:
        """初始化行业关联图（相关行业存在风险传导）"""
        return {
            '银行': ['保险', '房地产', '证券'],
            '保险': ['银行', '证券'],
            '白酒': ['食品饮料', '消费'],
            '新能源汽车': ['锂电池', '有色金属', '半导体'],
            '锂电池': ['新能源汽车', '有色金属', '锂电池材料'],
            '锂电池材料': ['锂电池', '有色金属'],
            '半导体': ['新能源汽车', '电子'],
            '有色金属': ['新能源汽车', '锂电池', '锂电池材料'],
            '家电': ['消费', '房地产'],
            '生物医药': ['医药', '医疗'],
            '家居用品': ['房地产', '消费'],
        }
    
    def _normalize_numeric(self, value: float, stats: Dict[str, float]) -> float:
        """Z-score归一化 + 截断"""
        z = (value - stats['mean']) / (stats['std'] + 1e-8)
        return max(-3.0, min(3.0, z)) / 3.0  # 归一化到[-1, 1]
    
    def _get_industry_id(self, industry: str) -> int:
        """行业名称哈希编码"""
        if industry not in self.industry_map:
            h = int(hashlib.md5(industry.encode()).hexdigest()[:4], 16) % 32
            self.industry_map[industry] = h
        return self.industry_map[industry]
    
    def extract_numeric_features(self, financial_data: Dict) -> np.ndarray:
        """提取归一化数值特征向量 [24维]"""
        features = []
        
        # 基础10维财务指标
        base_features = [
            'revenue_growth', 'profit_growth', 'roe', 'debt_ratio',
            'account_receivable_turnover', 'inventory_turnover',
            'operating_cashflow_ratio', 'goodwill_ratio', 'pledge_ratio', 'guarantee_ratio'
        ]
        for key in base_features:
            val = financial_data.get(key, self.feature_stats[key]['mean'])
            features.append(self._normalize_numeric(val, self.feature_stats[key]))
        
        # 衍生特征14维（交叉特征）
        rev_g = financial_data.get('revenue_growth', 10)
        prof_g = financial_data.get('profit_growth', 10)
        roe = financial_data.get('roe', 8)
        debt = financial_data.get('debt_ratio', 48)
        ar_turn = financial_data.get('account_receivable_turnover', 8)
        inv_turn = financial_data.get('inventory_turnover', 5)
        ocf = financial_data.get('operating_cashflow_ratio', 1.0)
        goodwill = financial_data.get('goodwill_ratio', 8)
        pledge = financial_data.get('pledge_ratio', 15)
        guarantee = financial_data.get('guarantee_ratio', 8)
        
        # 利润-营收背离度
        features.append(np.tanh((rev_g - prof_g) / 40.0))
        # 现金流-利润背离度
        features.append(np.tanh((ocf - 1.0) / 0.8))
        # 偿债压力综合指标
        features.append(np.tanh((debt - 40) / 30.0) * np.tanh((guarantee - 10) / 30.0))
        # 营运效率综合指标
        features.append(np.tanh((ar_turn - 6) / -5.0) * np.tanh((inv_turn - 4) / -4.0))
        # 资产质量综合指标
        features.append(np.tanh((goodwill - 15) / 20.0) + np.tanh((pledge - 30) / 30.0))
        # 盈利质量
        features.append(np.tanh((roe - 5) / -10.0) if roe < 5 else np.tanh((roe - 20) / 10.0))
        # 杠杆率异常
        features.append(1.0 if debt > 70 else (debt - 40) / 30.0)
        # 复合风险Z-score（简化Altman Z-score启发式）
        z_score = 1.2 * (1 - debt/100) + 1.4 * np.tanh(roe/10) + 3.3 * np.tanh(ocf) + 0.6 * np.tanh(rev_g/30)
        features.append(np.tanh((z_score - 2.0) / -2.0))  # 越低风险越高
        # 营收异常波动
        features.append(1.0 if abs(rev_g) > 50 else abs(rev_g) / 50.0)
        # 股权质押+担保复合风险
        features.append(np.tanh((pledge + guarantee) / 80.0))
        # 商誉+利润增长背离
        features.append(np.tanh((goodwill / 20.0) - (prof_g / 50.0)))
        # 应收账款+存货复合风险
        features.append(np.tanh((10 - ar_turn) / 8.0) + np.tanh((5 - inv_turn) / 5.0))
        # 市场情绪代理（市值小+高波动=高风险）
        features.append(0.0)  # 占位，实际由market_cap计算
        # 综合异常度
        risk_count = sum(1 for v in [debt>65, ocf<0.5, ar_turn<3, inv_turn<2, goodwill>25, pledge>50, guarantee>30] if v)
        features.append(risk_count / 7.0)
        
        return np.array(features, dtype=np.float32)
    
    def extract_categorical_features(self, company_info: Dict) -> Dict[str, int]:
        """提取类别特征ID"""
        industry = company_info.get('industry', '其他')
        market = company_info.get('market', '上交所')
        name = company_info.get('name', '')
        
        is_st = 1 if ('ST' in name or '*ST' in name) else 0
        
        return {
            'industry_id': self._get_industry_id(industry),
            'market_id': self.market_map.get(market, 0),
            'company_type_id': is_st,
        }
    
    def extract_text_features(self, text: str) -> np.ndarray:
        """提取文本特征向量（关键词频率 + 规则embedding）
        
        使用TF-IDF风格的风险关键词加权，输出128维向量
        （在完整实现中替换为FinBERT输出）
        """
        if not text:
            text = ''
        
        # 关键词频率特征（10类风险）
        keyword_freq = np.zeros(10, dtype=np.float32)
        risk_types = list(self.risk_keywords.keys())
        for i, rtype in enumerate(risk_types):
            kws = self.risk_keywords[rtype]
            count = sum(text.count(kw) for kw in kws)
            keyword_freq[i] = min(count / 5.0, 1.0)  # 截断
        
        # 文本长度特征
        text_len = min(len(text) / 2000.0, 1.0)
        
        # 数字密度（公告中数字比例，高=财务数据多）
        digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
        
        # 组合成128维向量（关键词特征 + 随机投影模拟BERT embedding）
        # 使用局部随机生成器，不修改全局numpy随机状态
        rng = np.random.default_rng(42)
        proj_matrix = rng.standard_normal((12, 118)).astype(np.float32) * 0.1
        base = np.concatenate([keyword_freq, [text_len, digit_ratio]])
        projected = base @ proj_matrix
        
        # 添加位置编码（模拟Transformer position embedding）
        pos_enc = np.sin(np.arange(118) * 0.1) * 0.05
        projected += pos_enc
        
        return np.concatenate([keyword_freq, [text_len, digit_ratio], projected[:116]]).astype(np.float32)
    
    def build_graph_features(self, company_info: Dict, all_companies: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """构建图特征和邻接矩阵
        
        Returns:
            node_features: [N, F] 节点特征矩阵
            adj_matrix: [N, N] 邻接矩阵
        """
        if all_companies is None:
            # 简化模式：返回单节点自环
            industry = company_info.get('industry', '其他')
            feat = np.zeros(32, dtype=np.float32)
            feat[self._get_industry_id(industry) % 32] = 1.0
            adj = np.array([[1.0]], dtype=np.float32)
            return feat, adj
        
        # 全模式：构建行业关联图
        companies_list = list(all_companies.items())
        n = len(companies_list)
        node_features = np.zeros((n, 32), dtype=np.float32)
        adj_matrix = np.eye(n, dtype=np.float32) * 0.5  # 自环
        
        code_to_idx = {code: i for i, (code, _) in enumerate(companies_list)}
        
        for i, (code, info) in enumerate(companies_list):
            ind = info.get('industry', '其他')
            node_features[i, self._get_industry_id(ind) % 32] = 1.0
            
            # 同行业连接
            for j, (code2, info2) in enumerate(companies_list):
                if i != j and info2.get('industry') == ind:
                    adj_matrix[i, j] = 1.0
            
            # 关联行业连接
            related = self.industry_graph.get(ind, [])
            for j, (code2, info2) in enumerate(companies_list):
                if i != j and info2.get('industry') in related:
                    adj_matrix[i, j] = max(adj_matrix[i, j], 0.5)
        
        return node_features, adj_matrix
    
    def build_temporal_features(self, financial_data: Dict, history: List[Dict] = None) -> np.ndarray:
        """构建时序特征 [T, D]
        
        当没有历史数据时，使用当前数据+统计先验模拟8个季度窗口
        """
        T = 8  # 8个季度
        D = 16  # 每步特征维度
        
        if history and len(history) > 0:
            # 使用真实历史数据
            features = np.zeros((T, D), dtype=np.float32)
            for t, hdata in enumerate(history[-T:]):
                numeric = self.extract_numeric_features(hdata)
                features[t, :min(D, len(numeric))] = numeric[:D]
            return features
        
        # 无历史数据时，模拟带噪声的趋势
        np.random.seed(hash(str(financial_data)) % (2**31))
        features = np.zeros((T, D), dtype=np.float32)
        current_numeric = self.extract_numeric_features(financial_data)
        
        for t in range(T):
            # 过去T个季度，线性外推+噪声
            decay = 0.7 + 0.3 * (t / T)
            noise = np.random.randn(D) * 0.05 * (1 + t / T)
            features[t] = current_numeric[:D] * decay + noise
            # 添加趋势
            features[t] += np.sin(np.arange(D) * 0.5 + t * 0.3) * 0.02
        
        return features
    
    def get_feature_dims(self) -> Dict[str, int]:
        """返回各特征维度"""
        return {
            'numeric': 24,
            'text': 128,
            'graph': 32,
            'temporal': (8, 16),
            'embed': {
                'industry': 32,
                'market': 8,
                'company_type': 4,
            }
        }
