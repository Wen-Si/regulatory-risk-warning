#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Predictor v3.0 - 混合预测引擎
融合Harness Engineering安全层、知识图谱、深度学习、强化学习与规则引擎

架构（v3.0）：
0. 输入安全护栏：输入验证、注入检测、PII脱敏（Harness Engineering）
1. 知识图谱层：监管KG + Graph RAG + 风险推理 + 证据链
2. 特征工程层：多模态特征提取（数值/文本/图/时序 + KG增强特征）
3. 深度学习层：DeepFM + Temporal Transformer + GAT + Text Encoder 四路并行
4. 集成层：Thompson Sampling动态加权 + KG风险评分融合
5. 强化学习层：PPO自适应阈值调整
6. 规则引擎层：硬约束gating
7. 输出安全护栏：合规校验、置信度门控、免责声明
8. 审计日志：全链路记录
"""
import numpy as np
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# 导入特征工程
from .features import FeatureEngineer
# 导入深度学习模型
from .models import DeepFM, TemporalTransformer, GATModel, RiskTextEncoder
# 导入强化学习
from .rl import PPOAgent, ThompsonSamplingEnsemble, OnlineLearner
# 导入安全合规层（Harness Engineering）
from .safety import SafetyHarness
# 导入知识图谱
from .knowledge_graph import RegulatoryKG, GraphRAG
from .config import MLConfig

# 导入规则引擎
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'rule_engine'))
from rule_engine import RuleEngine


class HybridPredictor:
    """智鉴风控 - 混合预测引擎 v3.0
    
    v3.0新增：
    - Harness Engineering安全合规层（输入/输出护栏、审计日志、工具防火墙）
    - 监管金融知识图谱（Graph RAG、风险传播推理、证据链构建）
    - KG增强的特征工程和预测融合
    - 全链路审计与合规检查
    
    v2.0保留：
    - DeepFM、Temporal Transformer、GAT、RiskTextEncoder四种DL模型
    - PPO自适应阈值优化和Thompson Sampling集成
    - 规则引擎硬约束
    """
    
    def __init__(self, use_rl=True, use_safety=True, use_kg=True):
        self.config = MLConfig
        self.config.ensure_dirs()
        
        # v3.0: Harness Engineering安全层
        self.use_safety = use_safety
        if use_safety:
            self.safety_harness = SafetyHarness()
        else:
            self.safety_harness = None
        
        # v3.0: 知识图谱
        self.use_kg = use_kg
        if use_kg:
            self.kg = RegulatoryKG()
            self.graph_rag = GraphRAG(self.kg)
        else:
            self.kg = None
            self.graph_rag = None
        
        # 特征工程
        self.feature_engineer = FeatureEngineer()
        
        # 初始化深度学习模型
        self.deepfm = DeepFM(
            num_numeric=24,
            cat_dims={'industry': 32, 'market': 8, 'company_type': 4},
            embed_dim=16,
            hidden_dims=[128, 64, 32],
            dropout=0.3
        )
        self.temporal_transformer = TemporalTransformer(
            d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.2,
            input_dim=16, num_windows=3
        )
        self.gat = GATModel(
            input_dim=32, hidden_dim=32, num_heads=4, num_layers=2, dropout=0.2
        )
        self.text_encoder = RiskTextEncoder(
            vocab_size=5000, embed_dim=128, num_heads=8, num_layers=4,
            max_seq_len=512, hidden_dim=256
        )
        
        # 规则引擎
        self.rule_engine = RuleEngine()
        
        # 强化学习组件
        self.use_rl = use_rl
        if use_rl:
            self.ppo_agent = PPOAgent(
                state_dim=64, action_dim=3, hidden_dim=64,
                lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2
            )
            self.ts_ensemble = ThompsonSamplingEnsemble(
                num_models=4, prior_alpha=1.0, prior_beta=1.0, exploration_decay=0.995
            )
        else:
            self.ppo_agent = None
            self.ts_ensemble = None
        
        # 在线学习
        self.online_learner = OnlineLearner(buffer_size=1000, batch_size=32, lr=5e-4)
        
        # 模型元信息
        self.model_info = {
            'version': '3.0.0',
            'models': ['SafetyHarness', 'KnowledgeGraph', 'DeepFM', 'TemporalTransformer', 
                      'GAT', 'RiskTextEncoder'],
            'rl_components': ['PPO', 'ThompsonSampling'] if use_rl else [],
            'safety_components': ['InputGuardrail', 'OutputGuardrail', 'AuditLogger'] if use_safety else [],
            'kg_components': ['RegulatoryKG', 'GraphRAG'] if use_kg else [],
            'rule_engine': True,
            'architecture': 'Hybrid Safety+KG+DL+RL+Rules (Harness Engineering + Graph RAG)'
        }
        
        # 加载检查点（如果存在）
        self._load_checkpoints()
    
    def _load_checkpoints(self):
        """加载模型检查点（生产环境使用，当前为初始化权重）"""
        # 在实际部署中，这里加载预训练权重
        pass
    
    def _extract_all_features(self, financial_data: Dict, announcement_text: str, 
                               company_info: Dict, all_companies: Dict = None,
                               history: List[Dict] = None) -> Dict[str, np.ndarray]:
        """提取所有模态特征"""
        features = {}
        
        # 1. 数值特征
        features['numeric'] = self.feature_engineer.extract_numeric_features(financial_data)
        
        # 2. 类别特征
        features['categorical'] = self.feature_engineer.extract_categorical_features(company_info)
        
        # 3. 文本特征
        features['text'] = self.feature_engineer.extract_text_features(announcement_text)
        
        # 4. 图特征
        node_feats, adj = self.feature_engineer.build_graph_features(company_info, all_companies)
        features['graph_nodes'] = node_feats
        features['graph_adj'] = adj
        features['graph_target_idx'] = 0  # 默认目标公司是第一个节点
        
        # 5. 时序特征
        features['temporal'] = self.feature_engineer.build_temporal_features(financial_data, history)
        
        return features
    
    def _run_deep_models(self, features: Dict, announcement_text: str = '') -> Dict[str, Any]:
        """运行所有深度学习模型"""
        results = {}
        
        # 1. DeepFM预测
        deepfm_prob = self.deepfm.predict(
            features['numeric'], 
            features['categorical']
        )
        results['deepfm_prob'] = float(deepfm_prob[0, 0])
        
        # 2. Temporal Transformer多窗口预测
        temporal_preds = self.temporal_transformer.predict(features['temporal'])
        results['temporal_30d'] = temporal_preds[30]
        results['temporal_60d'] = temporal_preds[60]
        results['temporal_90d'] = temporal_preds[90]
        
        # 3. GAT图神经网络预测
        gat_score = self.gat.predict(
            features['graph_nodes'],
            features['graph_adj'],
            target_idx=features.get('graph_target_idx', 0)
        )
        results['gat_score'] = gat_score
        
        # 4. 文本编码器预测
        text_result = self.text_encoder.encode(announcement_text)
        results['text_risk'] = text_result['risk_score']
        results['text_seq_len'] = text_result.get('seq_len', 0)
        
        return results
    
    def _ensemble_predictions(self, deep_results: Dict, rule_result: Dict) -> Dict[str, Any]:
        """集成多个模型的预测"""
        # 各模型60天预测概率
        model_probs_60d = [
            deep_results['deepfm_prob'],                          # DeepFM
            deep_results['temporal_60d'],                         # Temporal Transformer
            0.5 * deep_results['gat_score'] + 0.5 * deep_results['text_risk'],  # GAT+Text融合
            rule_result['total_score'] / 100.0,                   # 规则引擎
        ]
        
        if self.ts_ensemble:
            # Thompson Sampling动态加权
            ensemble_prob_60d, weights = self.ts_ensemble.predict(model_probs_60d)
            ensemble_weights = {
                'DeepFM': float(weights[0]),
                'TemporalTransformer': float(weights[1]),
                'GAT+Text': float(weights[2]),
                'RuleEngine': float(weights[3]),
            }
            model_performance = self.ts_ensemble.get_model_performance()
        else:
            # 固定权重（不使用RL时）
            fixed_weights = np.array([0.30, 0.30, 0.20, 0.20])
            ensemble_prob_60d = float(np.dot(model_probs_60d, fixed_weights))
            ensemble_weights = {
                'DeepFM': 0.30,
                'TemporalTransformer': 0.30,
                'GAT+Text': 0.20,
                'RuleEngine': 0.20,
            }
            model_performance = {}
        
        # 推导30天和90天概率
        # 30天：基于时序模型和基础概率的比例
        ratio_30 = deep_results['temporal_30d'] / max(deep_results['temporal_60d'], 0.01)
        ratio_90 = deep_results['temporal_90d'] / max(deep_results['temporal_60d'], 0.01)
        
        # 文本和图信号调整
        text_boost = 1.0 + 0.3 * deep_results['text_risk']
        gat_boost = 1.0 + 0.2 * deep_results['gat_score']
        
        ensemble_prob_30d = ensemble_prob_60d * min(ratio_30, 0.9) * text_boost * gat_boost
        ensemble_prob_90d = min(ensemble_prob_60d * max(ratio_90, 1.1) * text_boost * gat_boost, 0.98)
        
        # 裁剪
        ensemble_prob_30d = max(0.02, min(ensemble_prob_30d, 0.95))
        ensemble_prob_60d = max(0.02, min(ensemble_prob_60d, 0.97))
        ensemble_prob_90d = max(ensemble_prob_60d + 0.02, min(ensemble_prob_90d, 0.99))
        
        return {
            'prob_30d': ensemble_prob_30d,
            'prob_60d': ensemble_prob_60d,
            'prob_90d': ensemble_prob_90d,
            'model_probs': {
                'deepfm': deep_results['deepfm_prob'],
                'temporal_30d': deep_results['temporal_30d'],
                'temporal_60d': deep_results['temporal_60d'],
                'temporal_90d': deep_results['temporal_90d'],
                'gat': deep_results['gat_score'],
                'text': deep_results['text_risk'],
                'rule_engine': rule_result['total_score'] / 100.0,
            },
            'ensemble_weights': ensemble_weights,
            'model_performance': model_performance
        }
    
    def _apply_rl_threshold(self, ensemble: Dict, deep_results: Dict) -> Dict[str, Any]:
        """应用PPO强化学习阈值调整"""
        if not self.ppo_agent:
            # 无RL时使用固定阈值
            adjusted = np.array([ensemble['prob_30d'], ensemble['prob_60d'], ensemble['prob_90d']])
            return {
                'adjusted_probs': adjusted,
                'threshold_adjustment': np.zeros(3),
                'risk_level': self._get_risk_level(adjusted[1])
            }
        
        # 构建RL状态
        model_outputs = {
            'deepfm_prob': deep_results['deepfm_prob'],
            'temporal_30d': deep_results['temporal_30d'],
            'temporal_60d': deep_results['temporal_60d'],
            'temporal_90d': deep_results['temporal_90d'],
            'gat_score': deep_results['gat_score'],
            'text_risk': deep_results['text_risk'],
            'rule_score': ensemble['model_probs']['rule_engine'],
            'ensemble_prob': ensemble['prob_60d'],
        }
        state = self.ppo_agent.get_state(model_outputs)
        
        # PPO选择阈值调整动作
        base_probs = np.array([ensemble['prob_30d'], ensemble['prob_60d'], ensemble['prob_90d']])
        adjusted, action, log_prob, value = self.ppo_agent.adjust_thresholds(state, base_probs)
        
        risk_level = self._get_risk_level(adjusted[1])
        
        return {
            'adjusted_probs': adjusted,
            'threshold_adjustment': action,
            'risk_level': risk_level,
            'rl_state': state,
            'rl_log_prob': log_prob,
            'rl_value': value
        }
    
    def _apply_rule_gating(self, probs: np.ndarray, rule_result: Dict, 
                            fin_signals: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """规则引擎硬约束gating
        
        如果规则引擎检测到高风险信号，提升概率底线
        """
        gated_probs = probs.copy()
        gating_reasons = []
        
        # 高置信度规则触发时，提升概率
        high_risk_signals = [s for s in fin_signals if s['weight'] >= 13]
        critical_triggers = rule_result.get('trigger_events', [])
        
        if len(high_risk_signals) >= 3:
            # 多个强信号：确保至少中风险
            gated_probs = np.maximum(gated_probs, [0.35, 0.50, 0.65])
            gating_reasons.append(f"规则引擎检测到{len(high_risk_signals)}个强风险信号")
        
        if critical_triggers:
            # 触发事件（如重大重组）：提升概率
            boost = min(0.15, 0.05 * len(critical_triggers))
            gated_probs = gated_probs + boost
            gating_reasons.append(f"检测到{len(critical_triggers)}个重大触发事件")
        
        if rule_result.get('company_weight', 1.0) > 1.0:
            # ST/*ST公司额外提升
            gated_probs = gated_probs * 1.2
            gating_reasons.append("公司类型为ST/*ST，风险权重提升")
        
        # 裁剪
        gated_probs = np.clip(gated_probs, 0.01, 0.99)
        
        return gated_probs, gating_reasons
    
    def _get_risk_level(self, prob_60d: float) -> str:
        """根据60天概率确定风险等级"""
        if prob_60d >= 0.7:
            return '高风险'
        elif prob_60d >= 0.4:
            return '中风险'
        else:
            return '低风险'
    
    def _generate_attribution(self, features: Dict, deep_results: Dict, 
                               rule_result: Dict, ensemble: Dict) -> Dict[str, Any]:
        """生成可解释性归因分析"""
        attribution = {
            'model_contributions': {},
            'top_risk_factors': [],
            'risk_type_scores': {}
        }
        
        # 各模型贡献度
        for model_name, weight in ensemble['ensemble_weights'].items():
            model_prob = ensemble['model_probs'].get(
                model_name.lower().replace('+', '_'),
                ensemble['model_probs'].get('deepfm', 0.5)
            )
            attribution['model_contributions'][model_name] = {
                'weight': float(weight),
                'raw_score': float(model_prob),
                'contribution': float(weight * model_prob)
            }
        
        # 风险因子排名
        fin_signals = rule_result.get('fin_signals', [])
        comp_signals = rule_result.get('comp_signals', [])
        trigger_events = rule_result.get('trigger_events', [])
        
        all_signals = []
        for s in fin_signals:
            all_signals.append({
                'factor': s['signal'],
                'type': s['type'],
                'weight': s['weight'],
                'source': 'DeepFM+规则引擎',
                'detail': s.get('value', '')
            })
        for s in comp_signals:
            all_signals.append({
                'factor': s['signal'],
                'type': s['type'],
                'weight': s['weight'],
                'source': '文本编码器+规则引擎'
            })
        for e in trigger_events:
            all_signals.append({
                'factor': e['type'],
                'type': '触发事件',
                'weight': e['weight'],
                'source': '公告研读Agent',
                'keywords': e.get('matched_keywords', [])
            })
        
        # 添加文本风险
        if deep_results['text_risk'] > 0.3:
            all_signals.append({
                'factor': '公告文本语义风险',
                'type': '文本风险',
                'weight': int(deep_results['text_risk'] * 20),
                'source': 'RiskTextEncoder'
            })
        
        # 添加图风险
        if deep_results['gat_score'] > 0.3:
            all_signals.append({
                'factor': '行业风险传染',
                'type': '关联风险',
                'weight': int(deep_results['gat_score'] * 15),
                'source': 'GAT图神经网络'
            })
        
        # 按权重排序
        all_signals.sort(key=lambda x: -x['weight'])
        attribution['top_risk_factors'] = all_signals[:10]
        
        # 风险类型得分
        risk_types = {}
        for s in all_signals:
            rtype = s['type']
            risk_types[rtype] = risk_types.get(rtype, 0) + s['weight']
        attribution['risk_type_scores'] = risk_types
        
        return attribution
    
    def predict(self, financial_data: Dict, announcement_text: str = '', 
                company_info: Dict = None, all_companies: Dict = None,
                history: List[Dict] = None, session_id: str = '',
                user_role: str = 'public_user') -> Dict[str, Any]:
        """
        主预测函数 v3.0（安全护栏 + 知识图谱增强）
        
        流水线：
        0. 输入安全护栏检查（注入检测、PII脱敏、速率限制）
        1. 知识图谱Graph RAG检索（风险推理、证据链构建）
        2. 特征提取（含KG增强特征）
        3. 规则引擎预测
        4. 深度学习模型预测
        5. 集成预测（含KG风险评分融合）
        6. PPO强化学习阈值调整
        7. 规则引擎硬约束
        8. 输出安全护栏检查（合规校验、免责声明）
        9. 审计日志记录
        """
        company_info = company_info or {}
        start_time = datetime.now()
        company_code = company_info.get('code', financial_data.get('code', 'UNKNOWN'))
        company_name = company_info.get('name', '')
        
        # ====== Step 0: 输入安全护栏（Harness Engineering） ======
        safety_checks = {'input_passed': True, 'input_warnings': []}
        masked_text = announcement_text
        if self.safety_harness:
            input_check = self.safety_harness.check_input(
                financial_data=financial_data,
                announcement_text=announcement_text,
                company_info=company_info,
                user_role=user_role,
                session_id=session_id,
            )
            safety_checks['input_passed'] = input_check.passed and not input_check.blocked
            safety_checks['input_warnings'] = [input_check.details] if input_check.details else []
            safety_checks['blocked_reasons'] = [t for t in input_check.detected_threats]
            safety_checks['detected_threats'] = input_check.detected_threats
            safety_checks['pii_masked'] = input_check.pii_found
            safety_checks['input_risk_level'] = input_check.risk_level
            
            # 使用脱敏后文本
            masked_text = input_check.sanitized_input if input_check.sanitized_input else announcement_text
            
            # 如果被阻断，返回安全响应
            if input_check.blocked or not input_check.passed:
                return self._build_blocked_response(
                    company_code, input_check, start_time
                )
        
        # ====== Step 1: 知识图谱Graph RAG检索 ======
        kg_knowledge = None
        kg_rag_context = ''
        if self.graph_rag:
            kg_knowledge = self.graph_rag.retrieve_for_company(
                company_code, company_name, financial_data, masked_text
            )
            kg_rag_context = self.graph_rag.get_rag_prompt_context(
                company_code, company_name, financial_data, masked_text
            )
        
        # ====== Step 2: 特征提取（含KG增强特征） ======
        features = self._extract_all_features(
            financial_data, masked_text, company_info, all_companies, history
        )
        
        # KG增强特征拼接
        if self.kg and company_code:
            kg_embedding = self.kg.get_enhanced_embedding(company_code)
            features['kg_enhanced'] = kg_embedding
        
        # ====== Step 3: 规则引擎预测 ======
        rule_result = self.rule_engine.predict_inquiry_probability(
            financial_data, masked_text, company_info
        )
        
        # ====== Step 4: 深度学习模型预测 ======
        deep_results = self._run_deep_models(features, masked_text)
        
        # ====== Step 5: 集成预测（含KG风险评分融合） ======
        ensemble = self._ensemble_predictions(deep_results, rule_result)
        
        # KG风险评分融合
        kg_score = 0.0
        if kg_knowledge:
            kg_score = kg_knowledge.get('kg_risk_score', 0.0)
            # 将KG评分融入集成概率（权重0.15）
            ensemble['prob_30d'] = ensemble['prob_30d'] * 0.85 + kg_score * 0.15 * 0.8
            ensemble['prob_60d'] = ensemble['prob_60d'] * 0.85 + kg_score * 0.15
            ensemble['prob_90d'] = ensemble['prob_90d'] * 0.85 + kg_score * 0.15 * 1.2
        
        # ====== Step 6: PPO强化学习阈值调整 ======
        rl_result = self._apply_rl_threshold(ensemble, deep_results)
        
        # ====== Step 7: 规则引擎硬约束 ======
        final_probs, gating_reasons = self._apply_rule_gating(
            rl_result['adjusted_probs'], rule_result, rule_result.get('fin_signals', [])
        )
        
        # ====== Step 8: 归因分析（含KG证据链） ======
        attribution = self._generate_attribution(features, deep_results, rule_result, ensemble)
        
        # 融入KG证据链到归因
        if kg_knowledge and kg_knowledge.get('evidence_chain'):
            for ev in kg_knowledge['evidence_chain']:
                attribution['top_risk_factors'].insert(0, {
                    'factor': '知识图谱推理',
                    'type': 'KG推理',
                    'weight': int(kg_score * 20) + 5,
                    'source': 'Graph RAG',
                    'detail': ev,
                })
            attribution['top_risk_factors'] = attribution['top_risk_factors'][:12]
        
        # 确定最终风险等级和关注点
        risk_level = self._get_risk_level(final_probs[1])
        
        # 预测问询可能涉及的问题
        predicted_questions = rule_result.get('predicted_questions', [])
        if not predicted_questions:
            predicted_questions = self._generate_fallback_questions(attribution['top_risk_factors'])
        
        # 计算推理时间
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # ====== 构建原始结果 ======
        raw_result = {
            'inquiry_probability_30d': round(float(final_probs[0] * 100), 1),
            'inquiry_probability_60d': round(float(final_probs[1] * 100), 1),
            'inquiry_probability_90d': round(float(final_probs[2] * 100), 1),
            'risk_level': risk_level,
            
            'model_details': {
                'deepfm_score': round(deep_results['deepfm_prob'] * 100, 1),
                'temporal_scores': {
                    '30d': round(deep_results['temporal_30d'] * 100, 1),
                    '60d': round(deep_results['temporal_60d'] * 100, 1),
                    '90d': round(deep_results['temporal_90d'] * 100, 1),
                },
                'gat_contagion_score': round(deep_results['gat_score'] * 100, 1),
                'text_risk_score': round(deep_results['text_risk'] * 100, 1),
                'rule_engine_score': round(rule_result['total_score'], 1),
                'kg_risk_score': round(kg_score * 100, 1) if kg_knowledge else 0.0,
                'ensemble_weights': ensemble['ensemble_weights'],
                'model_performance': ensemble.get('model_performance', {}),
            },
            
            'attribution': attribution,
            'top_risk_factors': attribution['top_risk_factors'],
            'main_risk_types': list(attribution['risk_type_scores'].keys())[:5],
            
            'rule_engine': {
                'fin_score': rule_result['fin_score'],
                'comp_score': rule_result['comp_score'],
                'trigger_score': rule_result['trigger_score'],
                'fin_signals': rule_result.get('fin_signals', []),
                'comp_signals': rule_result.get('comp_signals', []),
                'trigger_events': rule_result.get('trigger_events', []),
                'predicted_questions': predicted_questions[:10],
            },
            
            'rl_adjustment': {
                'enabled': self.use_rl,
                'threshold_delta': [float(x) for x in rl_result.get('threshold_adjustment', [0, 0, 0])],
                'gating_reasons': gating_reasons,
            },
            
            # v3.0: 知识图谱结果
            'knowledge_graph': {
                'enabled': self.use_kg,
                'kg_risk_score': round(kg_score * 100, 1) if kg_knowledge else 0.0,
                'evidence_chain': kg_knowledge.get('evidence_chain', []) if kg_knowledge else [],
                'regulatory_context': kg_knowledge.get('regulatory_context', []) if kg_knowledge else [],
                'direct_risks': kg_knowledge.get('direct_risks', [])[:5] if kg_knowledge else [],
                'inferred_risks': kg_knowledge.get('inferred_risks', [])[:5] if kg_knowledge else [],
                'risk_paths': kg_knowledge.get('risk_paths', [])[:3] if kg_knowledge else [],
                'rag_context': kg_rag_context,
            },
            
            'risk_summary': self._generate_risk_summary(
                company_info, final_probs, risk_level, attribution['top_risk_factors']
            ),
            'key_evidence': [f['detail'] if 'detail' in f else f['factor'] 
                            for f in attribution['top_risk_factors'][:5]],
            
            'meta': {
                'model_version': '3.0.0',
                'architecture': 'Hybrid Safety+KG+DL+RL+Rules (Harness Engineering + Graph RAG)',
                'models_used': ['SafetyHarness', 'KnowledgeGraph', 'DeepFM', 'TemporalTransformer', 
                               'GAT', 'RiskTextEncoder'] + (['PPO', 'ThompsonSampling'] if self.use_rl else []) + ['RuleEngine'],
                'inference_time_ms': round(inference_time * 1000, 1),
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
        }
        
        # ====== Step 9: 输出安全护栏检查（Harness Engineering） ======
        if self.safety_harness:
            output_check = self.safety_harness.check_output(raw_result)
            disclaimer_added = any('disclaimer' in m.lower() or '免责' in m or '风险提示' in m 
                                   for m in output_check.modifications)
            safety_info = {
                'output_passed': output_check.passed,
                'output_warnings': output_check.warnings,
                'compliance_issues': output_check.compliance_issues,
                'modifications': output_check.modifications,
                'disclaimer_added': disclaimer_added,
                'needs_human_review': output_check.needs_human_review,
                'input_threats_blocked': len(safety_checks.get('detected_threats', [])),
            }
            # 从filtered_output获取安全处理后的结果
            safe_result = output_check.filtered_output if output_check.filtered_output else dict(raw_result)
            safe_result['safety'] = safety_info
        else:
            safe_result = dict(raw_result)
            safe_result['safety'] = {
                'output_passed': True,
                'safety_harness_enabled': False,
            }
        
        # ====== Step 10: 审计日志记录 ======
        audit_id = ''
        if self.safety_harness:
            audit_id = self.safety_harness.audit_prediction(
                company_code=company_code,
                company_info=company_info,
                input_data={
                    'financial_data': financial_data,
                    'announcement_text': masked_text[:500],  # 只记录前500字符
                    'has_history': bool(history),
                },
                prediction_result=safe_result,
                safety_checks={
                    **safety_checks,
                    'output_passed': safe_result['safety'].get('output_passed', True),
                },
            )
            safe_result['meta']['audit_id'] = audit_id
        
        return safe_result
    
    def _build_blocked_response(self, company_code: str, input_check, 
                                 start_time: datetime) -> Dict[str, Any]:
        """构建输入被阻断时的安全响应"""
        inference_time = (datetime.now() - start_time).total_seconds()
        return {
            'error': 'input_blocked',
            'blocked': True,
            'message': '输入安全检查未通过，请求已被阻断',
            'reasons': input_check.detected_threats,
            'details': input_check.details,
            'risk_level': '未评估',
            'inquiry_probability_30d': None,
            'inquiry_probability_60d': None,
            'inquiry_probability_90d': None,
            'safety': {
                'input_passed': False,
                'blocked_reasons': input_check.detected_threats,
                'detected_threats': input_check.detected_threats,
                'input_risk_level': input_check.risk_level,
            },
            'meta': {
                'model_version': '3.0.0',
                'inference_time_ms': round(inference_time * 1000, 1),
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
        }
    
    def _generate_fallback_questions(self, risk_factors: List[Dict]) -> List[str]:
        """生成问询问题"""
        questions = []
        for factor in risk_factors[:5]:
            if '财务' in factor.get('type', '') or '偿债' in factor.get('type', ''):
                questions.append(f"请说明{factor['factor']}的原因及对持续经营的影响")
            elif '关联' in factor.get('type', ''):
                questions.append(f"请补充披露{factor['factor']}的定价公允性")
            elif '商誉' in factor.get('factor', ''):
                questions.append("请说明商誉减值测试的关键参数及合理性")
            elif '触发' in factor.get('type', ''):
                questions.append(f"请详细披露{factor['factor']}的决策程序和合规性")
            else:
                questions.append(f"请说明{factor['factor']}的具体情况")
        return questions
    
    def _generate_risk_summary(self, company_info: Dict, probs: np.ndarray, 
                                risk_level: str, factors: List[Dict]) -> str:
        """生成风险总结"""
        name = company_info.get('name', '该公司')
        industry = company_info.get('industry', '')
        
        p60 = round(float(probs[1] * 100), 1)
        
        if risk_level == '高风险':
            summary = f"基于DeepFM+时序Transformer+GAT+文本编码器的深度学习融合模型，结合PPO强化学习自适应阈值和规则引擎硬约束，{name}未来60天被监管问询的概率为{p60}%，风险等级为【高风险】。"
        elif risk_level == '中风险':
            summary = f"基于多模型融合预测，{name}未来60天被监管问询的概率为{p60}%，风险等级为【中风险】。"
        else:
            summary = f"综合深度学习与规则引擎分析，{name}未来60天被监管问询的概率为{p60}%，整体风险可控。"
        
        if factors:
            top3 = [f['factor'] for f in factors[:3]]
            summary += f"主要风险因子包括：{'、'.join(top3)}。"
        
        summary += f"建议重点关注{industry}行业监管动态，持续跟踪公司公告披露情况。"
        
        return summary
    
    def provide_feedback(self, company_code: str, prediction_result: Dict, 
                         actual_inquiry: bool, risk_type: str = ''):
        """
        提供反馈用于在线学习和RL更新
        
        Args:
            company_code: 公司代码
            prediction_result: predict()返回的结果
            actual_inquiry: 是否真的被问询
            risk_type: 问询类型
        """
        # 更新Thompson Sampling
        if self.ts_ensemble:
            model_probs = [
                prediction_result['model_details']['deepfm_score'] / 100,
                prediction_result['model_details']['temporal_scores']['60d'] / 100,
                (prediction_result['model_details']['gat_contagion_score'] + 
                 prediction_result['model_details']['text_risk_score']) / 200,
                prediction_result['model_details']['rule_engine_score'] / 100,
            ]
            self.ts_ensemble.update(1 if actual_inquiry else 0, np.array(model_probs))
        
        # 存储到在线学习器
        self.online_learner.add_feedback(
            features={'company_code': company_code},
            prediction=prediction_result['inquiry_probability_60d'] / 100,
            actual=1 if actual_inquiry else 0,
            risk_type=risk_type
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            **self.model_info,
            'models_used': self.model_info['models'] + (['PPO', 'ThompsonSampling'] if self.use_rl else []),
            'online_learning_stats': self.online_learner.get_statistics(),
            'ts_model_performance': self.ts_ensemble.get_model_performance() if self.ts_ensemble else {},
        }
