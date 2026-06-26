#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智鉴风控 - 监管问询规则引擎
基于从上海证券交易所抓取的真实问询函构建的规则库
"""
import json
import re
import os
from datetime import datetime
from typing import Dict, List, Any

# 加载规则库
RULES_PATH = os.path.join(os.path.dirname(__file__), 'inquiry_rules.json')
with open(RULES_PATH, 'r', encoding='utf-8') as f:
    RULES = json.load(f)


class RuleEngine:
    """监管问询规则引擎"""
    
    def __init__(self):
        self.rules = RULES
        self.categories = self.rules['rule_categories']
    
    def check_trigger_events(self, text: str) -> List[Dict]:
        """检测触发事件类型"""
        results = []
        for rule in self.categories['trigger_events']['rules']:
            match_count = sum(1 for kw in rule['keywords'] if kw in text)
            if match_count > 0:
                results.append({
                    'rule_id': rule['id'],
                    'type': rule['type'],
                    'weight': rule['weight'] * match_count,
                    'matched_keywords': [kw for kw in rule['keywords'] if kw in text],
                    'trigger_probability': rule['trigger_probability'],
                    'common_questions': rule['common_questions']
                })
        return sorted(results, key=lambda x: -x['weight'])
    
    def check_financial_signals(self, financial_data: Dict) -> List[Dict]:
        """检测财务异常信号"""
        signals = []
        rules = self.categories['financial_signals']['rules']
        
        # 商誉占净资产
        if financial_data.get('goodwill_ratio', 0) > 25:
            signals.append({'rule_id': 'FS001', 'signal': '商誉占净资产>25%', 'weight': 15, 'type': '商誉减值风险', 'value': f"{financial_data['goodwill_ratio']:.1f}%"})
        # 资产负债率
        if financial_data.get('debt_ratio', 0) > 65:
            signals.append({'rule_id': 'FS003', 'signal': '资产负债率>65%', 'weight': 15, 'type': '偿债压力', 'value': f"{financial_data['debt_ratio']:.1f}%"})
        # 经营现金流
        if financial_data.get('operating_cashflow_ratio', 1) < 0.5:
            signals.append({'rule_id': 'FS004', 'signal': '经营现金流/净利润<0.5', 'weight': 13, 'type': '盈利质量存疑', 'value': f"{financial_data['operating_cashflow_ratio']:.2f}"})
        # 应收账款周转
        if financial_data.get('account_receivable_turnover', 10) < 3:
            signals.append({'rule_id': 'FS005', 'signal': '应收账款周转率<3', 'weight': 12, 'type': '回款风险', 'value': f"{financial_data['account_receivable_turnover']:.1f}次"})
        # 存货周转
        if financial_data.get('inventory_turnover', 5) < 2:
            signals.append({'rule_id': 'FS006', 'signal': '存货周转率<2', 'weight': 10, 'type': '存货积压风险', 'value': f"{financial_data['inventory_turnover']:.1f}次"})
        # 营收增长率异常
        rg = abs(financial_data.get('revenue_growth', 10))
        if rg > 50 or financial_data.get('revenue_growth', 0) < -20:
            signals.append({'rule_id': 'FS007', 'signal': '营收增长率异常', 'weight': 10, 'type': '经营波动', 'value': f"{financial_data['revenue_growth']:.1f}%"})
        # 利润与营收背离
        rg_v = financial_data.get('revenue_growth', 0)
        pg_v = financial_data.get('profit_growth', 0)
        if abs(rg_v - pg_v) > 40:
            signals.append({'rule_id': 'FS008', 'signal': '净利润与营收背离', 'weight': 13, 'type': '财务真实性', 'value': f"营收{rg_v:.1f}% vs 净利{pg_v:.1f}%"})
        # 股权质押
        if financial_data.get('pledge_ratio', 0) > 50:
            signals.append({'rule_id': 'FS017', 'signal': '股权质押比例>50%', 'weight': 10, 'type': '控制权风险', 'value': f"{financial_data['pledge_ratio']:.1f}%"})
        # 对外担保
        if financial_data.get('guarantee_ratio', 0) > 30:
            signals.append({'rule_id': 'FS018', 'signal': '对外担保>30%', 'weight': 10, 'type': '或有负债风险', 'value': f"{financial_data['guarantee_ratio']:.1f}%"})
        
        return sorted(signals, key=lambda x: -x['weight'])
    
    def check_compliance_signals(self, text: str, company_info: Dict) -> List[Dict]:
        """检测合规性信号"""
        signals = []
        text_lower = text
        
        # 关联交易
        if '关联交易' in text_lower and ('未充分披露' in text_lower or '未识别' in text_lower):
            signals.append({'rule_id': 'CS001', 'signal': '关联交易未充分披露', 'weight': 15, 'type': '关联交易合规性'})
        # 担保超限
        if '担保' in text_lower and ('超净资产' in text_lower or '超出' in text_lower):
            signals.append({'rule_id': 'CS003', 'signal': '对外担保超净资产', 'weight': 14, 'type': '担保合规性'})
        # 资金占用
        if '资金占用' in text_lower or ('非经营性资金占用' in text_lower):
            signals.append({'rule_id': 'CS004', 'signal': '资金占用嫌疑', 'weight': 18, 'type': '资金占用'})
        # 内幕信息
        if '内幕信息' in text_lower and ('提前泄露' in text_lower or '登记不全' in text_lower):
            signals.append({'rule_id': 'CS007', 'signal': '内幕信息管理不规范', 'weight': 10, 'type': '内幕信息管理'})
        # 跨界投资
        if '跨界' in text_lower or '与主业' in text_lower or '与主营业务' in text_lower:
            signals.append({'rule_id': 'CS008', 'signal': '跨界投资与主营不相关', 'weight': 13, 'type': '投资合理性'})
        # 客户供应商同一控制
        if ('客户与供应商' in text_lower and '同一' in text_lower) or '资金闭环' in text_lower:
            signals.append({'rule_id': 'CS011', 'signal': '客户与供应商关联/资金闭环', 'weight': 16, 'type': '交易真实性'})
        # 股东减持配合
        if '股东减持' in text_lower and '披露配合' in text_lower:
            signals.append({'rule_id': 'CS006', 'signal': '股东减持与披露配合', 'weight': 15, 'type': '内幕交易'})
        
        return sorted(signals, key=lambda x: -x['weight'])
    
    def get_pattern_match(self, company_code: str, company_name: str) -> List[Dict]:
        """匹配历史案例模式"""
        results = []
        patterns = self.rules.get('case_patterns', [])
        for p in patterns:
            results.append({
                'case_id': p['case_id'],
                'pattern': p['pattern'],
                'risk_level': p['risk_level'],
                'probability': p['probability'],
                'example': p['example'],
                'applicable_rules': p['rules']
            })
        return results
    
    def get_company_type_weight(self, company_name: str) -> float:
        """获取公司类型权重"""
        weights = self.categories['risk_weights']['company_type_weight']
        if '*ST' in company_name or 'ST' in company_name:
            return weights['*ST/ST公司']
        return weights['正常运营公司']
    
    def predict_inquiry_probability(self, financial_data: Dict, text: str = '', company_info: Dict = None) -> Dict:
        """综合预测问询概率"""
        company_info = company_info or {}
        company_name = company_info.get('name', '')
        
        # 1. 财务信号评分
        fin_signals = self.check_financial_signals(financial_data)
        fin_score = sum(s['weight'] for s in fin_signals)
        
        # 2. 合规信号评分
        comp_signals = self.check_compliance_signals(text, company_info)
        comp_score = sum(s['weight'] for s in comp_signals)
        
        # 3. 触发事件评分
        trigger_events = self.check_trigger_events(text)
        trigger_score = sum(e['weight'] for e in trigger_events) if trigger_events else 0
        
        # 4. 公司类型权重
        company_weight = self.get_company_type_weight(company_name)
        
        # 综合评分
        total_score = (fin_score + comp_score + trigger_score) * company_weight
        total_score = min(total_score, 100)
        
        # 风险等级
        if total_score >= 70:
            risk_level = '高风险'
        elif total_score >= 40:
            risk_level = '中风险'
        else:
            risk_level = '低风险'
        
        # 预测的问询可能内容
        predicted_questions = []
        for e in trigger_events[:3]:
            predicted_questions.extend(e['common_questions'][:2])
        for s in fin_signals[:3]:
            predicted_questions.append(f"关注{s['type']}：{s['signal']}")
        for s in comp_signals[:3]:
            predicted_questions.append(f"合规问询：{s['type']}")
        
        return {
            'inquiry_probability_30d': max(int(total_score * 0.7), 5),
            'inquiry_probability_60d': int(total_score),
            'inquiry_probability_90d': min(int(total_score * 1.1), 98),
            'risk_level': risk_level,
            'total_score': total_score,
            'fin_score': fin_score,
            'comp_score': comp_score,
            'trigger_score': trigger_score,
            'company_weight': company_weight,
            'fin_signals': fin_signals,
            'comp_signals': comp_signals,
            'trigger_events': trigger_events,
            'predicted_questions': list(set(predicted_questions))[:10],
            'matched_patterns': self.get_pattern_match(company_info.get('code', ''), company_name)[:3]
        }
    
    def generate_inquiry_content(self, prediction: Dict, company_info: Dict) -> str:
        """基于预测生成模拟的问询函内容"""
        name = company_info.get('name', '该公司')
        code = company_info.get('code', '')
        today = datetime.now().strftime('%Y年%m月%d日')
        
        questions = prediction.get('predicted_questions', [])
        risk_level = prediction.get('risk_level', '中风险')
        
        content = f"""上 海 证 券 交 易 所
上证公函【{datetime.now().strftime('%Y%m%d')}】XXXX号

关于{name}({code})相关事项的问询函

{name}：

依据《股票上市规则》第13.1.1条等有关规定，针对公司近期公告及财务状况，鉴于以下事项对投资者影响重大，现请你公司核实并披露如下事项：

一、关于触发事项及合规性
请公司补充披露：
1. {questions[0] if questions else '近期重大事项的合规性'}
2. {questions[1] if len(questions) > 1 else '信息披露的完整性'}
3. {questions[2] if len(questions) > 2 else '相关决策程序'}

二、关于财务真实性
"""
        if prediction['fin_signals']:
            content += f"经分析公司财务指标，发现以下异常信号：\n"
            for i, s in enumerate(prediction['fin_signals'][:3], 1):
                content += f"  ({i}) {s['signal']}（{s.get('value', '')}）\n"
            content += "请公司结合上述指标说明原因及对持续经营的影响。\n\n"
        
        content += f"""三、关于内幕信息管理
请公司补充披露：
1. 内幕信息知情人登记情况
2. 自查相关信息是否存在提前泄露情形
3. 控股股东、实际控制人、董监高近期股票交易情况

请公司收到本问询函后立即披露，并于5个交易日内披露对本问询函的回复。
请律师、独立董事、年审会计师发表意见。

上海证券交易所上市公司管理一部
{today}
"""
        return content


if __name__ == '__main__':
    # 测试
    engine = RuleEngine()
    
    # 测试数据
    financial_data = {
        'revenue_growth': 65.0,
        'profit_growth': 20.0,
        'roe': 5.0,
        'debt_ratio': 75.0,
        'account_receivable_turnover': 2.0,
        'inventory_turnover': 1.5,
        'operating_cashflow_ratio': 0.3,
        'goodwill_ratio': 32.0,
        'pledge_ratio': 60.0,
        'guarantee_ratio': 25.0
    }
    
    text = "公司拟进行重大资产购买，同时近期股价异动，股东减持与公告配合"
    company_info = {'code': '600XXX', 'name': '*ST某公司', 'industry': '综合'}
    
    result = engine.predict_inquiry_probability(financial_data, text, company_info)
    print(f"问询概率: {result['inquiry_probability_60d']}%")
    print(f"风险等级: {result['risk_level']}")
    print(f"综合评分: {result['total_score']}")
    print(f"\n财务信号: {len(result['fin_signals'])}个")
    for s in result['fin_signals']:
        print(f"  - {s['signal']} (权重{s['weight']})")
    print(f"\n合规信号: {len(result['comp_signals'])}个")
    print(f"\n触发事件: {len(result['trigger_events'])}个")
    print(f"\n预测问询问题: {result['predicted_questions']}")
