#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regulatory Knowledge Graph - 监管知识图谱主类

核心功能：
1. 实体管理：增删查改实体节点
2. 关系管理：增删查改关系边
3. 图谱推理：多跳推理、风险传播路径分析
4. Graph RAG：检索相关知识增强预测
5. 知识增强特征：将图谱信息融入特征向量
6. 动态更新：支持增量添加公司实体和关系
"""

import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, deque
import json

from .entities import (
    Entity, EntityType,
    RISK_TYPE_ENTITIES, FIN_INDICATOR_ENTITIES, EVENT_TYPE_ENTITIES,
)
from .relations import (
    Relation, RelationType, build_regulatory_relations,
)


class RegulatoryKG:
    """监管金融知识图谱
    
    采用属性图模型，支持：
    - 监管知识子图（风险类型、财务指标、事件类型、因果关系）
    - 公司动态子图（上市公司节点、行业关联、风险事件）
    - 图推理（风险传播路径、关联风险发现）
    - Graph RAG增强检索
    """
    
    def __init__(self):
        # 实体存储: entity_id -> Entity
        self.entities: Dict[str, Entity] = {}
        # 邻接表: entity_id -> [(relation, target_id)]
        self.adj_out: Dict[str, List[Tuple[Relation, str]]] = defaultdict(list)
        self.adj_in: Dict[str, List[Tuple[Relation, str]]] = defaultdict(list)
        
        # 公司实体索引
        self.companies: Dict[str, str] = {}  # company_code -> entity_id
        # 行业实体索引
        self.industries: Dict[str, str] = {}  # industry_name -> entity_id
        
        # 统计（必须在_init_regulatory_knowledge之前初始化）
        self.stats = {
            'total_entities': 0,
            'total_relations': 0,
            'companies': 0,
            'inference_queries': 0,
        }
        
        # 初始化监管知识子图
        self._init_regulatory_knowledge()
    
    def _init_regulatory_knowledge(self):
        """初始化预定义的监管知识"""
        # 添加风险类型实体
        for entity in RISK_TYPE_ENTITIES.values():
            self.add_entity(entity)
        
        # 添加财务指标实体
        for entity in FIN_INDICATOR_ENTITIES.values():
            self.add_entity(entity)
        
        # 添加事件类型实体
        for entity in EVENT_TYPE_ENTITIES.values():
            self.add_entity(entity)
        
        # 添加预定义关系
        relations = build_regulatory_relations()
        for rel in relations:
            self.add_relation(rel)
    
    def add_entity(self, entity: Entity) -> str:
        """添加实体"""
        self.entities[entity.id] = entity
        if entity.entity_type == EntityType.COMPANY:
            code = entity.properties.get('stock_code', '')
            if code:
                self.companies[code] = entity.id
            self.stats['companies'] += 1
        elif entity.entity_type == EntityType.INDUSTRY:
            self.industries[entity.name] = entity.id
        self.stats['total_entities'] += 1
        return entity.id
    
    def add_relation(self, relation: Relation):
        """添加关系"""
        self.adj_out[relation.source_id].append((relation, relation.target_id))
        self.adj_in[relation.target_id].append((relation, relation.source_id))
        if relation.bidirectional:
            self.adj_out[relation.target_id].append((relation, relation.source_id))
            self.adj_in[relation.source_id].append((relation, relation.target_id))
        self.stats['total_relations'] += 1
    
    def add_company(self, code: str, name: str, industry: str = '',
                    market: str = 'SH', **kwargs) -> str:
        """添加或获取公司实体"""
        if code in self.companies:
            return self.companies[code]
        
        entity = Entity(
            id=f'company_{code}',
            name=name,
            entity_type=EntityType.COMPANY,
            properties={
                'stock_code': code,
                'market': market,
                'industry': industry,
                **kwargs,
            },
            source='system',
        )
        self.add_entity(entity)
        
        # 关联到行业
        if industry:
            industry_id = self._get_or_create_industry(industry)
            if industry_id:
                rel = Relation(
                    id=f'rel_belongs_{code}_{industry}',
                    source_id=entity.id,
                    target_id=industry_id,
                    relation_type=RelationType.BELONGS_TO,
                    weight=1.0,
                )
                self.add_relation(rel)
        
        return entity.id
    
    def _get_or_create_industry(self, industry_name: str) -> Optional[str]:
        """获取或创建行业实体"""
        if industry_name in self.industries:
            return self.industries[industry_name]
        
        entity = Entity(
            id=f'industry_{len(self.industries)}',
            name=industry_name,
            entity_type=EntityType.INDUSTRY,
        )
        self.add_entity(entity)
        return entity.id
    
    def add_company_risk(self, company_code: str, risk_type_id: str,
                         weight: float = 0.5, evidence: str = ''):
        """为公司添加风险关系"""
        if company_code not in self.companies:
            return
        
        company_id = self.companies[company_code]
        if risk_type_id not in self.entities:
            return
        
        # 检查是否已存在
        for rel, tid in self.adj_out[company_id]:
            if tid == risk_type_id and rel.relation_type == RelationType.HAS_RISK:
                rel.weight = max(rel.weight, weight)
                return
        
        rel = Relation(
            id=f'rel_risk_{company_code}_{risk_type_id}',
            source_id=company_id,
            target_id=risk_type_id,
            relation_type=RelationType.HAS_RISK,
            weight=weight,
            properties={'evidence': evidence},
        )
        self.add_relation(rel)
    
    def add_company_event(self, company_code: str, event_type_id: str,
                          weight: float = 0.5):
        """为公司添加事件触发关系"""
        if company_code not in self.companies:
            return
        
        company_id = self.companies[company_code]
        if event_type_id not in self.entities:
            return
        
        rel = Relation(
            id=f'rel_event_{company_code}_{event_type_id}',
            source_id=company_id,
            target_id=event_type_id,
            relation_type=RelationType.TRIGGERS,
            weight=weight,
        )
        self.add_relation(rel)
    
    def link_companies(self, code1: str, code2: str, relation_type: RelationType,
                       weight: float = 0.5, properties: Dict = None):
        """关联两家公司（如同行业、供应链等）"""
        if code1 not in self.companies or code2 not in self.companies:
            return
        
        id1 = self.companies[code1]
        id2 = self.companies[code2]
        
        rel = Relation(
            id=f'rel_link_{code1}_{code2}',
            source_id=id1,
            target_id=id2,
            relation_type=relation_type,
            weight=weight,
            properties=properties or {},
            bidirectional=True,
        )
        self.add_relation(rel)
    
    # ============================================================
    # 图推理
    # ============================================================
    
    def infer_risks(self, company_code: str, max_hops: int = 2) -> List[Dict]:
        """
        基于知识图谱推理公司可能面临的风险
        
        推理逻辑：
        1. 直接风险（公司→风险类型）
        2. 事件触发风险（公司→事件→风险类型）
        3. 指标指示风险（财务指标异常→风险类型）
        4. 因果链风险（风险A→风险B）
        5. 关联公司风险传染（同行业/关联公司风险）
        
        Returns:
            推理出的风险列表，按置信度排序
        """
        self.stats['inference_queries'] += 1
        
        if company_code not in self.companies:
            # 添加公司实体
            self.add_company(company_code, company_code)
        
        company_id = self.companies[company_code]
        inferred_risks = {}  # risk_id -> {score, paths, evidence}
        
        # BFS多跳推理
        visited = set()
        queue = deque()
        queue.append((company_id, 0, 1.0, []))  # (node_id, hop, prob, path)
        
        while queue:
            node_id, hop, prob, path = queue.popleft()
            
            if hop > max_hops:
                continue
            
            node_key = (node_id, hop)
            if node_key in visited:
                continue
            visited.add(node_key)
            
            node = self.entities.get(node_id)
            if not node:
                continue
            
            # 如果是风险类型节点，记录
            if node.entity_type == EntityType.RISK_TYPE and hop > 0:
                if node_id not in inferred_risks or prob > inferred_risks[node_id]['score']:
                    inferred_risks[node_id] = {
                        'risk_id': node_id,
                        'risk_name': node.name,
                        'score': prob,
                        'severity': node.properties.get('severity', 'medium'),
                        'path': path + [node.name],
                        'hop_count': hop,
                    }
            
            # 遍历出边
            for rel, target_id in self.adj_out.get(node_id, []):
                target = self.entities.get(target_id)
                if not target:
                    continue
                
                # 计算传播概率
                trans_prob = prob * rel.weight
                
                # 不同关系类型的传播衰减不同
                if rel.relation_type == RelationType.HAS_RISK:
                    trans_prob *= 1.0
                elif rel.relation_type == RelationType.TRIGGERS:
                    trans_prob *= 0.8
                elif rel.relation_type == RelationType.INDICATES:
                    trans_prob *= 0.7
                elif rel.relation_type == RelationType.CAUSES:
                    trans_prob *= 0.6
                elif rel.relation_type == RelationType.CORRELATES_WITH:
                    trans_prob *= 0.4
                elif rel.relation_type == RelationType.BELONGS_TO:
                    trans_prob *= 0.3
                else:
                    trans_prob *= 0.5
                
                if trans_prob > 0.05:  # 阈值剪枝
                    queue.append((target_id, hop + 1, trans_prob, path + [node.name]))
        
        # 按分数排序
        risks = sorted(inferred_risks.values(), key=lambda x: -x['score'])
        return risks[:10]
    
    def find_risk_paths(self, company_code: str, target_risk_id: str,
                        max_hops: int = 3) -> List[List[str]]:
        """查找从公司到特定风险的传播路径"""
        if company_code not in self.companies:
            return []
        
        company_id = self.companies[company_code]
        paths = []
        
        def dfs(node_id, path, visited, prob):
            if len(path) > max_hops + 1:
                return
            if node_id == target_risk_id:
                paths.append({'path': path[:], 'probability': prob})
                return
            
            for rel, target in self.adj_out.get(node_id, []):
                if target not in visited:
                    visited.add(target)
                    path.append(self.entities.get(target, Entity('', '', EntityType.COMPANY)).name)
                    dfs(target, path, visited, prob * rel.weight)
                    path.pop()
                    visited.remove(target)
        
        dfs(company_id, [self.entities[company_id].name], {company_id}, 1.0)
        return sorted(paths, key=lambda x: -x['probability'])[:5]
    
    def get_similar_risks(self, risk_type_id: str, top_k: int = 5) -> List[Dict]:
        """获取相关风险类型"""
        similar = []
        visited = {risk_type_id}
        
        # 通过CORRELATES_WITH和CAUSES查找
        for rel, target in self.adj_out.get(risk_type_id, []):
            if target in visited:
                continue
            entity = self.entities.get(target)
            if entity and entity.entity_type == EntityType.RISK_TYPE:
                similar.append({
                    'risk_id': target,
                    'risk_name': entity.name,
                    'relation': rel.relation_type.value,
                    'weight': rel.weight,
                })
                visited.add(target)
        
        return sorted(similar, key=lambda x: -x['weight'])[:top_k]
    
    # ============================================================
    # Graph RAG - 知识增强检索
    # ============================================================
    
    def retrieve_knowledge(self, company_code: str, financial_data: Dict = None,
                           triggered_events: List[str] = None, top_k: int = 10) -> Dict:
        """
        Graph RAG: 检索相关知识用于增强预测
        
        Returns:
            {
                'direct_risks': [...],        # 直接风险
                'inferred_risks': [...],      # 推理风险
                'risk_paths': [...],          # 风险传播路径
                'related_companies': [...],   # 关联公司
                'regulatory_context': [...],  # 监管上下文
                'kg_risk_score': float,       # 知识图谱风险评分
                'evidence': [...],            # 证据链
            }
        """
        # 直接和推理风险
        all_risks = self.infer_risks(company_code)
        
        # 财务指标触发的风险
        indicator_risks = []
        if financial_data:
            indicator_risks = self._check_indicators(financial_data)
        
        # 事件触发的风险
        event_risks = []
        if triggered_events:
            event_risks = self._check_events(triggered_events)
        
        # 计算KG风险评分
        kg_score = self._calculate_kg_score(all_risks, indicator_risks, event_risks)
        
        # 风险传播路径（取最高风险的路径）
        risk_paths = []
        for risk in all_risks[:3]:
            paths = self.find_risk_paths(company_code, risk['risk_id'])
            if paths:
                risk_paths.extend(paths[:2])
        
        # 行业风险上下文
        industry_context = self._get_industry_context(company_code)
        
        # 证据链
        evidence = []
        for risk in all_risks[:5]:
            evidence.append({
                'risk': risk['risk_name'],
                'score': round(risk['score'], 3),
                'path': ' → '.join(risk['path']),
            })
        
        return {
            'direct_risks': [r for r in all_risks if r['hop_count'] == 1][:5],
            'inferred_risks': [r for r in all_risks if r['hop_count'] > 1][:5],
            'indicator_risks': indicator_risks,
            'event_risks': event_risks,
            'risk_paths': risk_paths[:5],
            'industry_context': industry_context,
            'kg_risk_score': kg_score,
            'evidence': evidence,
        }
    
    def _check_indicators(self, financial_data: Dict) -> List[Dict]:
        """检查财务指标异常触发的风险"""
        risks = []
        
        # 商誉占比
        goodwill = financial_data.get('goodwill_ratio', 0) or 0
        if goodwill > 30:
            risks.append({
                'indicator': '商誉占净资产比',
                'value': goodwill,
                'threshold': 30,
                'risk': '商誉减值风险',
                'severity': 'high' if goodwill > 50 else 'medium',
            })
        
        # 资产负债率
        debt_ratio = financial_data.get('debt_ratio', 0) or 0
        if debt_ratio > 70:
            risks.append({
                'indicator': '资产负债率',
                'value': debt_ratio,
                'threshold': 70,
                'risk': '债务违约风险',
                'severity': 'critical' if debt_ratio > 85 else 'high',
            })
        
        # 流动比率
        current_ratio = financial_data.get('current_ratio', 999) or 999
        if current_ratio < 1.0:
            risks.append({
                'indicator': '流动比率',
                'value': current_ratio,
                'threshold': 1.0,
                'risk': '短期偿债风险',
                'severity': 'critical' if current_ratio < 0.5 else 'high',
            })
        
        # 经营现金流
        ocf_ratio = financial_data.get('ocf_to_profit', 100) or 100
        if ocf_ratio < 50:
            risks.append({
                'indicator': '经营现金流/净利润',
                'value': ocf_ratio,
                'threshold': 50,
                'risk': '财务造假嫌疑',
                'severity': 'high',
            })
        
        # 其他应收款
        oth_recv = financial_data.get('other_receivables_ratio', 0) or 0
        if oth_recv > 10:
            risks.append({
                'indicator': '其他应收款/总资产',
                'value': oth_recv,
                'threshold': 10,
                'risk': '关联方资金占用',
                'severity': 'high' if oth_recv > 20 else 'medium',
            })
        
        return risks
    
    def _check_events(self, event_types: List[str]) -> List[Dict]:
        """检查事件触发的风险"""
        risks = []
        event_map = {
            'major_asset_restructuring': '重大资产重组→问询概率提升',
            'asset_acquisition': '资产收购→关注估值合理性',
            'related_party_transaction': '关联交易→关注定价公允性',
            'equity_pledge': '股权质押→关注平仓风险',
            'performance_loss': '业绩亏损→ST风险',
            'auditor_change': '审计机构变更→关注审计意见',
        }
        for evt in event_types:
            if evt in event_map:
                entity = EVENT_TYPE_ENTITIES.get(evt)
                if entity:
                    risks.append({
                        'event': entity.name,
                        'risk_boost': entity.properties.get('inquiry_probability_boost', 0.1),
                        'description': event_map[evt],
                    })
        return risks
    
    def _calculate_kg_score(self, inferred_risks, indicator_risks, event_risks) -> float:
        """计算知识图谱综合风险评分"""
        score = 0.0
        
        # 推理风险贡献
        for risk in inferred_risks:
            severity_weight = {'critical': 0.3, 'high': 0.2, 'medium': 0.1, 'low': 0.05}
            w = severity_weight.get(risk.get('severity', 'medium'), 0.1)
            score += risk['score'] * w
        
        # 指标异常贡献
        for ind in indicator_risks:
            severity_weight = {'critical': 0.25, 'high': 0.15, 'medium': 0.08}
            w = severity_weight.get(ind.get('severity', 'medium'), 0.08)
            score += w
        
        # 事件贡献
        for evt in event_risks:
            score += evt.get('risk_boost', 0.1)
        
        return min(score, 1.0)
    
    def _get_industry_context(self, company_code: str) -> List[Dict]:
        """获取行业风险上下文"""
        if company_code not in self.companies:
            return []
        
        company_id = self.companies[company_code]
        company = self.entities[company_id]
        industry = company.properties.get('industry', '')
        
        context = []
        if industry:
            # 查找同行业公司的风险
            industry_id = self.industries.get(industry)
            if industry_id:
                peer_risks = defaultdict(int)
                for rel, source_id in self.adj_in.get(industry_id, []):
                    if rel.relation_type == RelationType.BELONGS_TO:
                        for r2, tid in self.adj_out.get(source_id, []):
                            if r2.relation_type == RelationType.HAS_RISK:
                                entity = self.entities.get(tid)
                                if entity:
                                    peer_risks[entity.name] += 1
                
                for risk_name, count in sorted(peer_risks.items(), key=lambda x: -x[1])[:3]:
                    context.append({
                        'industry': industry,
                        'common_risk': risk_name,
                        'peer_count': count,
                    })
        
        return context
    
    def get_enhanced_embedding(self, company_code: str) -> np.ndarray:
        """获取知识图谱增强的实体嵌入（用于GAT模型输入增强）"""
        # 生成简化的KG嵌入：风险类型one-hot + 关系权重
        embedding = np.zeros(32, dtype=np.float32)
        
        if company_code not in self.companies:
            return embedding
        
        company_id = self.companies[company_code]
        
        # 编码直接风险
        risk_idx = 0
        for rel, target_id in self.adj_out.get(company_id, []):
            if rel.relation_type == RelationType.HAS_RISK and risk_idx < 16:
                embedding[risk_idx] = rel.weight
                risk_idx += 1
            elif rel.relation_type == RelationType.TRIGGERS and risk_idx < 24:
                embedding[16 + (risk_idx - 16) % 8] = rel.weight
                risk_idx += 1
        
        # 编码行业信息
        industry = self.entities[company_id].properties.get('industry', '')
        if industry:
            embedding[24] = hash(industry) % 100 / 100.0
        
        return embedding
    
    def get_stats(self) -> Dict[str, Any]:
        """获取图谱统计"""
        type_counts = defaultdict(int)
        for entity in self.entities.values():
            type_counts[entity.entity_type.value] += 1
        
        return {
            **self.stats,
            'entity_types': dict(type_counts),
        }
