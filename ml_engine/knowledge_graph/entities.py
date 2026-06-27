#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实体定义 - 知识图谱节点类型

实体类型体系（参考FIBO金融行业业务本体）：
- Company: 上市公司
- Industry: 行业
- RiskType: 风险类型
- Regulation: 监管规则/法规
- FinancialIndicator: 财务指标
- EventType: 事件类型
- InquiryPattern: 问询模式/案例
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import uuid


class EntityType(Enum):
    """实体类型枚举"""
    COMPANY = 'company'                # 上市公司
    INDUSTRY = 'industry'              # 行业
    RISK_TYPE = 'risk_type'            # 风险类型
    REGULATION = 'regulation'          # 监管规则
    FINANCIAL_INDICATOR = 'fin_indicator'  # 财务指标
    EVENT_TYPE = 'event_type'          # 事件类型
    INQUIRY_PATTERN = 'inquiry_pattern'    # 问询模式
    PERSON = 'person'                  # 关联人员/高管


@dataclass
class Entity:
    """知识图谱实体（节点）"""
    id: str                           # 唯一标识
    name: str                         # 实体名称
    entity_type: EntityType           # 实体类型
    properties: Dict[str, Any] = field(default_factory=dict)  # 属性
    aliases: List[str] = field(default_factory=list)  # 别名
    confidence: float = 1.0           # 置信度
    source: str = ''                  # 数据来源
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.entity_type.value,
            'properties': self.properties,
            'aliases': self.aliases,
            'confidence': self.confidence,
        }


# ============================================================
# 预定义实体（监管知识图谱核心节点）
# ============================================================

# 风险类型实体
RISK_TYPE_ENTITIES = {
    'financial_fraud': Entity(
        id='rt_fin_fraud', name='财务造假', entity_type=EntityType.RISK_TYPE,
        properties={
            'severity': 'critical', 'category': 'financial',
            'description': '虚增收入、利润造假、虚假记载等财务欺诈行为',
            'typical_indicators': ['收入增长率异常', '毛利率异常', '经营现金流为负'],
        },
        aliases=['财务舞弊', '会计造假', '虚假陈述'],
    ),
    'goodwill_impairment': Entity(
        id='rt_goodwill', name='商誉减值风险', entity_type=EntityType.RISK_TYPE,
        properties={
            'severity': 'high', 'category': 'financial',
            'description': '高商誉占比导致的减值风险',
            'typical_indicators': ['商誉/净资产>30%', '业绩承诺未完成', '标的公司经营恶化'],
        },
        aliases=['商誉暴雷', '商誉减值'],
    ),
    'related_party': Entity(
        id='rt_related', name='关联交易风险', entity_type=EntityType.RISK_TYPE,
        properties={
            'severity': 'high', 'category': 'governance',
            'description': '关联方资金占用、关联交易非公允等',
            'typical_indicators': ['其他应收款异常', '关联交易占比高', '资金往来频繁'],
        },
        aliases=['关联方占用', '关联交易违规'],
    ),
    'debt_crisis': Entity(
        id='rt_debt', name='债务违约风险', entity_type=EntityType.RISK_TYPE,
        properties={
            'severity': 'critical', 'category': 'financial',
            'description': '高负债率、短期偿债压力大导致的违约风险',
            'typical_indicators': ['资产负债率>70%', '流动比率<1', '利息保障倍数<1'],
        },
        aliases=['债务危机', '偿债风险', '违约风险'],
    ),
    'restructuring_risk': Entity(
        id='rt_restruct', name='重大资产重组风险', entity_type=EntityType.RISK_TYPE,
        properties={
            'severity': 'medium', 'category': 'event',
            'description': '重大资产重组、跨界并购相关风险',
            'typical_indicators': ['跨界并购', '高溢价收购', '业绩对赌'],
        },
        aliases=['重组风险', '并购风险'],
    ),
    'governance_risk': Entity(
        id='rt_governance', name='公司治理风险', entity_type=EntityType.RISK_TYPE,
        properties={
            'severity': 'medium', 'category': 'governance',
            'description': '股权结构、内部控制、董监高相关风险',
            'typical_indicators': ['股权质押高', '高管频繁变动', '内控缺陷'],
        },
        aliases=['治理缺陷', '内控风险'],
    ),
    'information_disclosure': Entity(
        id='rt_disclosure', name='信息披露违规', entity_type=EntityType.RISK_TYPE,
        properties={
            'severity': 'high', 'category': 'compliance',
            'description': '信息披露不及时、不准确、不完整',
            'typical_indicators': ['延迟披露', '前后矛盾', '重大遗漏'],
        },
        aliases=['信披违规', '披露违规'],
    ),
    'st_designation': Entity(
        id='rt_st', name='ST/*ST风险', entity_type=EntityType.RISK_TYPE,
        properties={
            'severity': 'critical', 'category': 'financial',
            'description': '连续亏损导致ST/退市风险警示',
            'typical_indicators': ['连续两年亏损', '净资产为负', '审计意见非标'],
        },
        aliases=['ST风险', '退市风险'],
    ),
}

# 财务指标实体
FIN_INDICATOR_ENTITIES = {
    'goodwill_ratio': Entity(
        id='fi_goodwill', name='商誉占净资产比', entity_type=EntityType.FINANCIAL_INDICATOR,
        properties={
            'unit': '%', 'threshold_high': 30, 'threshold_critical': 50,
            'direction': 'higher_risk',
            'related_risk_types': ['rt_goodwill'],
        },
    ),
    'debt_ratio': Entity(
        id='fi_debt', name='资产负债率', entity_type=EntityType.FINANCIAL_INDICATOR,
        properties={
            'unit': '%', 'threshold_high': 70, 'threshold_critical': 85,
            'direction': 'higher_risk',
            'related_risk_types': ['rt_debt'],
        },
    ),
    'current_ratio': Entity(
        id='fi_current', name='流动比率', entity_type=EntityType.FINANCIAL_INDICATOR,
        properties={
            'unit': 'x', 'threshold_low': 1.0, 'threshold_critical': 0.5,
            'direction': 'lower_risk',
            'related_risk_types': ['rt_debt'],
        },
    ),
    'operating_cashflow': Entity(
        id='fi_ocf', name='经营现金流/净利润', entity_type=EntityType.FINANCIAL_INDICATOR,
        properties={
            'unit': '%', 'threshold_low': 50, 'threshold_critical': 0,
            'direction': 'lower_risk',
            'related_risk_types': ['rt_fin_fraud'],
        },
    ),
    'receivables_turnover': Entity(
        id='fi_recv', name='应收账款周转率', entity_type=EntityType.FINANCIAL_INDICATOR,
        properties={
            'unit': '次', 'threshold_low': 3, 'threshold_critical': 1,
            'direction': 'lower_risk',
            'related_risk_types': ['rt_fin_fraud', 'rt_related'],
        },
    ),
    'inventory_turnover': Entity(
        id='fi_inv', name='存货周转率', entity_type=EntityType.FINANCIAL_INDICATOR,
        properties={
            'unit': '次', 'threshold_low': 2, 'threshold_critical': 0.5,
            'direction': 'lower_risk',
            'related_risk_types': ['rt_fin_fraud'],
        },
    ),
    'roe': Entity(
        id='fi_roe', name='净资产收益率(ROE)', entity_type=EntityType.FINANCIAL_INDICATOR,
        properties={
            'unit': '%', 'threshold_low': 0, 'threshold_critical': -10,
            'direction': 'lower_risk',
            'related_risk_types': ['rt_fin_fraud', 'rt_st'],
        },
    ),
    'other_receivables_ratio': Entity(
        id='fi_othrecv', name='其他应收款/总资产', entity_type=EntityType.FINANCIAL_INDICATOR,
        properties={
            'unit': '%', 'threshold_high': 10, 'threshold_critical': 20,
            'direction': 'higher_risk',
            'related_risk_types': ['rt_related'],
        },
    ),
}

# 事件类型实体
EVENT_TYPE_ENTITIES = {
    'major_asset_restructuring': Entity(
        id='et_mar', name='重大资产重组', entity_type=EntityType.EVENT_TYPE,
        properties={'risk_multiplier': 1.5, 'inquiry_probability_boost': 0.2},
        aliases=['重组', '重大重组'],
    ),
    'asset_acquisition': Entity(
        id='et_acq', name='资产收购', entity_type=EntityType.EVENT_TYPE,
        properties={'risk_multiplier': 1.3, 'inquiry_probability_boost': 0.15},
        aliases=['收购资产', '对外收购'],
    ),
    'related_party_transaction': Entity(
        id='et_rpt', name='关联交易', entity_type=EntityType.EVENT_TYPE,
        properties={'risk_multiplier': 1.4, 'inquiry_probability_boost': 0.18},
        aliases=['关联方交易'],
    ),
    'external_investment': Entity(
        id='et_inv', name='对外投资', entity_type=EntityType.EVENT_TYPE,
        properties={'risk_multiplier': 1.2, 'inquiry_probability_boost': 0.1},
    ),
    'equity_pledge': Entity(
        id='et_pledge', name='股权质押', entity_type=EntityType.EVENT_TYPE,
        properties={'risk_multiplier': 1.3, 'inquiry_probability_boost': 0.12},
        aliases=['股票质押', '股份质押'],
    ),
    'performance_loss': Entity(
        id='et_loss', name='业绩亏损', entity_type=EntityType.EVENT_TYPE,
        properties={'risk_multiplier': 1.8, 'inquiry_probability_boost': 0.3},
        aliases=['亏损', '预亏'],
    ),
    'auditor_change': Entity(
        id='et_audit', name='审计机构变更', entity_type=EntityType.EVENT_TYPE,
        properties={'risk_multiplier': 1.5, 'inquiry_probability_boost': 0.2},
        aliases=['换审', '会计师事务所变更'],
    ),
    'executive_change': Entity(
        id='et_exec', name='高管变动', entity_type=EntityType.EVENT_TYPE,
        properties={'risk_multiplier': 1.1, 'inquiry_probability_boost': 0.05},
        aliases=['董监高变动', '高管离职'],
    ),
}
