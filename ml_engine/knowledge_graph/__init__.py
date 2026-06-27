#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智鉴风控 - 监管金融知识图谱

基于Graph RAG理念构建的金融监管知识图谱，提升AI Agent精准性：
1. 实体类型：公司、行业、风险类型、监管规则、财务指标、事件类型
2. 关系类型：has_risk, belongs_to, triggers, regulated_by, impacts, correlates_with
3. 图谱推理：风险传播路径推理、关联风险发现、因果链分析
4. Graph RAG增强：检索相关知识为预测提供事实依据
5. 监管规则图谱：将法规条款建模为可推理的知识网络

参考：
- FinKario: Event-Enhanced Financial Knowledge Graph (ACL 2026)
- FIBO (Financial Industry Business Ontology)
- Graph RAG: 图增强检索生成
- 合规知识图谱(CKG)架构
"""

from .knowledge_graph import RegulatoryKG
from .entities import Entity, EntityType
from .relations import Relation, RelationType
from .graph_rag import GraphRAG

__all__ = ['RegulatoryKG', 'Entity', 'EntityType', 'Relation', 'RelationType', 'GraphRAG']
