#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关系定义 - 知识图谱边类型

关系类型体系：
- BELONGS_TO: 公司属于某行业
- HAS_RISK: 公司/行业存在某类风险
- TRIGGERS: 事件/指标触发某类风险
- REGULATED_BY: 风险受某法规监管
- IMPACTS: 某因素影响某指标
- CORRELATES_WITH: 实体间存在相关性
- CAUSES: 因果关系
- SIMILAR_TO: 相似案例关系
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import uuid


class RelationType(Enum):
    """关系类型枚举"""
    BELONGS_TO = 'belongs_to'           # 属于（公司→行业）
    HAS_RISK = 'has_risk'               # 具有风险（公司→风险类型）
    TRIGGERS = 'triggers'               # 触发（事件→风险，指标→风险）
    REGULATED_BY = 'regulated_by'       # 受监管（风险→法规）
    IMPACTS = 'impacts'                 # 影响（因素→指标，事件→指标）
    CORRELATES_WITH = 'correlates_with' # 相关（公司↔公司，指标↔指标）
    CAUSES = 'causes'                   # 因果关系
    SIMILAR_TO = 'similar_to'           # 相似（案例↔案例）
    INDICATES = 'indicates'             # 指示（指标→风险）
    PART_OF = 'part_of'                 # 组成部分


@dataclass
class Relation:
    """知识图谱关系（边）"""
    id: str
    source_id: str          # 源实体ID
    target_id: str          # 目标实体ID
    relation_type: RelationType
    weight: float = 1.0     # 关系权重（置信度/强度）
    properties: Dict[str, Any] = field(default_factory=dict)
    bidirectional: bool = False
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'source': self.source_id,
            'target': self.target_id,
            'type': self.relation_type.value,
            'weight': self.weight,
            'properties': self.properties,
        }


def build_regulatory_relations():
    """构建监管知识图谱的预定义关系"""
    relations = []
    
    # === 财务指标 → 风险类型 (INDICATES) ===
    indicator_risk_map = {
        'fi_goodwill': ['rt_goodwill'],
        'fi_debt': ['rt_debt'],
        'fi_current': ['rt_debt'],
        'fi_ocf': ['rt_fin_fraud'],
        'fi_recv': ['rt_fin_fraud', 'rt_related'],
        'fi_inv': ['rt_fin_fraud'],
        'fi_roe': ['rt_fin_fraud', 'rt_st'],
        'fi_othrecv': ['rt_related'],
    }
    
    for indicator_id, risk_ids in indicator_risk_map.items():
        for risk_id in risk_ids:
            relations.append(Relation(
                id=f'rel_ind_{indicator_id}_{risk_id}',
                source_id=indicator_id,
                target_id=risk_id,
                relation_type=RelationType.INDICATES,
                weight=0.8,
                properties={'mechanism': 'threshold_violation', 'direction': 'positive'},
            ))
    
    # === 事件类型 → 风险类型 (TRIGGERS) ===
    event_risk_map = {
        'et_mar': ['rt_restruct', 'rt_fin_fraud', 'rt_disclosure'],
        'et_acq': ['rt_restruct', 'rt_goodwill'],
        'et_rpt': ['rt_related', 'rt_disclosure'],
        'et_inv': ['rt_restruct', 'rt_governance'],
        'et_pledge': ['rt_governance', 'rt_debt'],
        'et_loss': ['rt_st', 'rt_debt'],
        'et_audit': ['rt_fin_fraud', 'rt_disclosure'],
        'et_exec': ['rt_governance'],
    }
    
    for event_id, risk_ids in event_risk_map.items():
        for risk_id in risk_ids:
            relations.append(Relation(
                id=f'rel_evt_{event_id}_{risk_id}',
                source_id=event_id,
                target_id=risk_id,
                relation_type=RelationType.TRIGGERS,
                weight=0.7,
                properties={'mechanism': 'event_trigger'},
            ))
    
    # === 风险类型间因果关系 (CAUSES) ===
    causal_chains = [
        ('rt_fin_fraud', 'rt_st', 0.9),       # 财务造假→ST
        ('rt_debt', 'rt_st', 0.8),             # 债务危机→ST
        ('rt_related', 'rt_fin_fraud', 0.6),   # 关联交易→财务造假
        ('rt_disclosure', 'rt_fin_fraud', 0.7), # 信披违规→财务造假
        ('rt_goodwill', 'rt_fin_fraud', 0.5),  # 商誉减值→财务造假
        ('rt_governance', 'rt_related', 0.6),  # 治理缺陷→关联交易
    ]
    
    for src, tgt, w in causal_chains:
        relations.append(Relation(
            id=f'rel_cause_{src}_{tgt}',
            source_id=src,
            target_id=tgt,
            relation_type=RelationType.CAUSES,
            weight=w,
            properties={'causal_strength': w},
        ))
    
    # === 风险类型间相关性 (CORRELATES_WITH) ===
    correlations = [
        ('rt_fin_fraud', 'rt_disclosure', 0.8),
        ('rt_related', 'rt_governance', 0.7),
        ('rt_debt', 'rt_goodwill', 0.5),
        ('rt_restruct', 'rt_goodwill', 0.7),
    ]
    
    for src, tgt, w in correlations:
        relations.append(Relation(
            id=f'rel_corr_{src}_{tgt}',
            source_id=src,
            target_id=tgt,
            relation_type=RelationType.CORRELATES_WITH,
            weight=w,
            bidirectional=True,
        ))
    
    return relations
