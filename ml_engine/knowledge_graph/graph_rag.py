#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph RAG - 图增强检索生成模块

利用知识图谱为LLM/Agent提供结构化、可溯源的事实依据：
1. 子图检索：根据查询提取相关子图
2. 路径推理：沿关系边进行多跳推理
3. 证据链构建：将推理路径转化为可解释的证据链
4. 事实 grounding：为模型预测提供事实依据
5. 幻觉抑制：通过知识图谱验证模型输出
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
import re

from .knowledge_graph import RegulatoryKG
from .entities import Entity, EntityType, RISK_TYPE_ENTITIES, EVENT_TYPE_ENTITIES
from .relations import RelationType


class GraphRAG:
    """图增强检索引擎
    
    核心功能：
    - 基于实体的子图检索
    - 风险传播路径推理
    - 为预测提供证据链
    - 监管规则匹配
    """
    
    def __init__(self, kg: RegulatoryKG):
        self.kg = kg
        
        # 风险关键词到实体的映射（用于文本匹配）
        self.risk_keywords = self._build_risk_keyword_index()
        self.event_keywords = self._build_event_keyword_index()
        self.indicator_keywords = self._build_indicator_keyword_index()
    
    def _build_risk_keyword_index(self) -> Dict[str, str]:
        """构建风险关键词到实体ID的映射"""
        index = {}
        for rid, entity in RISK_TYPE_ENTITIES.items():
            index[entity.name] = rid
            for alias in entity.aliases:
                index[alias] = rid
        return index
    
    def _build_event_keyword_index(self) -> Dict[str, str]:
        """构建事件关键词到实体ID的映射"""
        index = {}
        for eid, entity in EVENT_TYPE_ENTITIES.items():
            index[entity.name] = eid
            for alias in entity.aliases:
                index[alias] = eid
        return index
    
    def _build_indicator_keyword_index(self) -> Dict[str, str]:
        """构建指标关键词映射"""
        return {
            '商誉': 'goodwill_ratio',
            '资产负债率': 'debt_ratio',
            '流动比率': 'current_ratio',
            '现金流': 'operating_cashflow',
            '应收账款': 'receivables_turnover',
            '存货': 'inventory_turnover',
            '其他应收款': 'other_receivables_ratio',
            'ROE': 'roe',
            '净资产收益率': 'roe',
        }
    
    def retrieve_for_company(self, company_code: str, company_name: str = '',
                             financial_data: Dict = None,
                             announcement_text: str = '') -> Dict[str, Any]:
        """
        为公司预测检索知识图谱上下文
        
        这是Graph RAG的核心接口，返回：
        1. 匹配到的风险实体
        2. 风险推理路径
        3. 监管规则依据
        4. 证据链文本
        """
        # 确保公司在图谱中
        self.kg.add_company(company_code, company_name or company_code)
        
        # 从公告文本中提取风险事件
        detected_events = self._extract_events_from_text(announcement_text)
        detected_risks = self._extract_risks_from_text(announcement_text)
        
        # 将检测到的事件和风险加入图谱
        for event_id, weight in detected_events.items():
            self.kg.add_company_event(company_code, event_id, weight)
        for risk_id, weight in detected_risks.items():
            self.kg.add_company_risk(company_code, risk_id, weight)
        
        # 检索知识
        knowledge = self.kg.retrieve_knowledge(
            company_code, financial_data, list(detected_events.keys())
        )
        
        # 构建证据链文本
        evidence_chain = self._build_evidence_chain(knowledge, company_name or company_code)
        
        # 构建监管上下文
        regulatory_context = self._build_regulatory_context(knowledge)
        
        return {
            **knowledge,
            'detected_events': detected_events,
            'detected_risks_from_text': detected_risks,
            'evidence_chain': evidence_chain,
            'regulatory_context': regulatory_context,
        }
    
    def _extract_events_from_text(self, text: str) -> Dict[str, float]:
        """从文本中提取事件类型"""
        if not text:
            return {}
        
        events = {}
        text_lower = text
        
        event_keyword_map = {
            'major_asset_restructuring': ['重大资产重组', '重组', '重大资产置换'],
            'asset_acquisition': ['收购', '资产收购', '购买资产', '对外收购'],
            'related_party_transaction': ['关联交易', '关联方', '关联往来'],
            'external_investment': ['对外投资', '设立子公司', '参股', '投资标的'],
            'equity_pledge': ['股权质押', '股份质押', '股票质押', '质押'],
            'performance_loss': ['亏损', '预亏', '业绩下滑', '净利润为负'],
            'auditor_change': ['会计师事务所变更', '审计机构变更', '换审'],
            'executive_change': ['高管离职', '董事辞职', '监事辞职', '高管变动'],
        }
        
        for event_id, keywords in event_keyword_map.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                events[event_id] = min(0.3 + count * 0.15, 0.9)
        
        return events
    
    def _extract_risks_from_text(self, text: str) -> Dict[str, float]:
        """从文本中提取风险关键词"""
        if not text:
            return {}
        
        risks = {}
        for risk_id, entity in RISK_TYPE_ENTITIES.items():
            keywords = [entity.name] + entity.aliases
            count = sum(1 for kw in keywords if kw in text)
            if count > 0:
                risks[risk_id] = min(0.3 + count * 0.15, 0.8)
        
        return risks
    
    def _build_evidence_chain(self, knowledge: Dict, company_name: str) -> List[str]:
        """构建可解释的证据链"""
        evidence = []
        
        # 指标风险证据
        for ind_risk in knowledge.get('indicator_risks', []):
            evidence.append(
                f"【指标异常】{ind_risk['indicator']}为{ind_risk['value']:.1f}"
                f"（阈值{ind_risk['threshold']}），可能引发{ind_risk['risk']}"
            )
        
        # 事件风险证据
        for evt_risk in knowledge.get('event_risks', []):
            evidence.append(
                f"【事件触发】检测到{evt_risk['event']}，"
                f"问询概率提升约{evt_risk['risk_boost']*100:.0f}个百分点"
            )
        
        # 推理路径证据
        for path_info in knowledge.get('risk_paths', [])[:3]:
            if isinstance(path_info, dict):
                path_str = ' → '.join(path_info.get('path', []))
                prob = path_info.get('probability', 0)
                evidence.append(
                    f"【风险传导】{path_str}（传播概率{prob*100:.0f}%）"
                )
        
        # 推理风险证据
        for risk in knowledge.get('inferred_risks', [])[:3]:
            evidence.append(
                f"【关联风险】通过{risk['hop_count']}跳推理发现{risk['risk_name']}"
                f"（置信度{risk['score']*100:.0f}%）"
            )
        
        return evidence
    
    def _build_regulatory_context(self, knowledge: Dict) -> List[str]:
        """构建监管上下文提示"""
        context = []
        
        kg_score = knowledge.get('kg_risk_score', 0)
        if kg_score > 0.6:
            context.append("根据知识图谱推理，该公司存在多条风险传导路径，需重点关注")
        elif kg_score > 0.3:
            context.append("知识图谱检测到部分风险信号，建议进一步核查")
        
        # 行业上下文
        for ic in knowledge.get('industry_context', []):
            context.append(
                f"{ic['industry']}行业内{ic['common_risk']}较为常见"
                f"（{ic['peer_count']}家同行业公司存在同类风险）"
            )
        
        return context
    
    def verify_claim(self, claim: str, company_code: str) -> Dict[str, Any]:
        """
        验证预测/声明是否与知识图谱一致（抑制幻觉）
        
        Returns:
            {
                'supported': bool,
                'confidence': float,
                'supporting_evidence': List[str],
                'contradicting_evidence': List[str],
            }
        """
        result = {
            'supported': True,
            'confidence': 0.5,
            'supporting_evidence': [],
            'contradicting_evidence': [],
        }
        
        # 简单的关键词匹配验证
        detected_risks = self._extract_risks_from_text(claim)
        inferred = self.kg.infer_risks(company_code)
        inferred_risk_names = {r['risk_name'] for r in inferred}
        
        for rid in detected_risks:
            entity = RISK_TYPE_ENTITIES.get(rid)
            if entity:
                if entity.name in inferred_risk_names:
                    result['supporting_evidence'].append(f"知识图谱支持{entity.name}风险")
                    result['confidence'] += 0.1
                else:
                    result['contradicting_evidence'].append(f"知识图谱未发现{entity.name}的直接证据")
                    result['confidence'] -= 0.1
        
        result['confidence'] = max(0.0, min(1.0, result['confidence']))
        result['supported'] = result['confidence'] > 0.3 and not result['contradicting_evidence']
        
        return result
    
    def get_rag_prompt_context(self, company_code: str, company_name: str = '',
                                financial_data: Dict = None,
                                announcement_text: str = '') -> str:
        """
        获取用于增强LLM prompt的知识上下文
        
        返回结构化的文本，可以直接注入到LLM prompt中
        """
        knowledge = self.retrieve_for_company(
            company_code, company_name, financial_data, announcement_text
        )
        
        context_parts = ["【知识图谱分析】"]
        
        if knowledge.get('kg_risk_score', 0) > 0:
            context_parts.append(f"知识图谱风险评分: {knowledge['kg_risk_score']*100:.0f}/100")
        
        if knowledge.get('evidence_chain'):
            context_parts.append("风险证据链:")
            for ev in knowledge['evidence_chain'][:5]:
                context_parts.append(f"- {ev}")
        
        if knowledge.get('regulatory_context'):
            context_parts.append("监管上下文:")
            for ctx in knowledge['regulatory_context'][:3]:
                context_parts.append(f"- {ctx}")
        
        direct_risks = knowledge.get('direct_risks', [])
        if direct_risks:
            risk_names = [r['risk_name'] for r in direct_risks[:3]]
            context_parts.append(f"直接风险: {', '.join(risk_names)}")
        
        return '\n'.join(context_parts)
