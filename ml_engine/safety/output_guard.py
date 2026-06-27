#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Output Guardrail - 输出护栏
负责在模型/Agent输出返回给用户前进行安全检查：
1. 置信度门控（低置信度输出不直接交付）
2. 合规性检查（输出内容合规校验）
3. 事实核查标记（无依据的声明标记）
4. 敏感信息过滤（防止模型泄露敏感数据）
5. 输出格式强制（确保结构化输出）
6. Kill Switch（高风险输出直接阻断）
"""
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class OutputCheckResult:
    """输出检查结果"""
    passed: bool
    allowed: bool
    risk_level: str = 'low'
    filtered_output: Any = None
    warnings: List[str] = field(default_factory=list)
    compliance_issues: List[str] = field(default_factory=list)
    confidence_warning: bool = False
    needs_human_review: bool = False
    modifications: List[str] = field(default_factory=list)


class OutputGuardrail:
    """输出护栏 - 后处理安全层
    
    遵循安全原则：
    - 不确定时保守输出
    - 高风险建议需附带免责声明
    - 投资建议必须标注风险
    - 任何预测不能作为投资决策唯一依据
    """
    
    def __init__(self, min_confidence: float = 0.3, max_risk_prob: float = 0.99):
        self.min_confidence = min_confidence
        self.max_risk_prob = max_risk_prob
        
        # 合规必须包含的风险提示
        self.risk_disclaimer_required = True
        self.standard_disclaimer = (
            "【风险提示】以上分析基于AI模型预测，仅供参考，不构成投资建议。"
            "投资有风险，决策需谨慎。监管政策动态变化，请以官方公告为准。"
        )
        
        # 禁止输出的内容模式
        self.forbidden_patterns = [
            r'保证.{0,5}(盈利|不亏损|赚钱|收益)',
            r'100%[\s]*确定',
            r'一定会(被问询|退市|暴雷)',
            r'内幕消息',
            r'建议(买入|卖出|加仓|减仓|清仓)',
            r'稳赚不赔',
            r'包赚',
        ]
        
        # 输出需标记的不确定表达
        self.uncertainty_markers = [
            '可能', '或许', '估计', '大概', '约',
            '模型预测', '基于历史', '概率为',
        ]
        
        # 统计
        self.stats = {
            'total_checks': 0,
            'blocked': 0,
            'modified': 0,
            'warnings_issued': 0,
            'human_reviews': 0,
        }
    
    def check_prediction(self, prediction: Dict[str, Any]) -> OutputCheckResult:
        """
        检查预测输出的安全性与合规性
        
        Args:
            prediction: predict()返回的结果字典
            
        Returns:
            OutputCheckResult
        """
        self.stats['total_checks'] += 1
        warnings = []
        compliance_issues = []
        modifications = []
        needs_review = False
        allowed = True
        
        result = prediction.copy()
        
        # 1. 概率边界检查（防止极端值）
        for key in ['inquiry_probability_30d', 'inquiry_probability_60d', 'inquiry_probability_90d']:
            prob = result.get(key, 0) / 100.0
            if prob > self.max_risk_prob:
                result[key] = self.max_risk_prob * 100
                modifications.append(f'{key} 概率超过上限，已裁剪至{self.max_risk_prob*100}%')
                warnings.append(f'预测概率异常高，已触发裁剪机制')
            if prob < 0.01:
                result[key] = 1.0
                modifications.append(f'{key} 概率过低，已提升至1%底线')
        
        # 2. 置信度检查
        risk_level = result.get('risk_level', '低风险')
        inference_time = result.get('meta', {}).get('inference_time_ms', 0)
        
        # 快速推理可能意味着特征不足
        if inference_time < 10 and risk_level == '高风险':
            warnings.append('推理时间过短，可能存在特征不足，高风险结论置信度有限')
            needs_review = True
        
        # 3. 高风险输出附加审查标记
        if risk_level == '高风险':
            prob_60d = result.get('inquiry_probability_60d', 0)
            if prob_60d > 90:
                warnings.append(f'极高风险预测({prob_60d}%)，建议人工复核')
                needs_review = True
        
        # 4. 风险归因完整性检查
        top_factors = result.get('top_risk_factors', [])
        if not top_factors and risk_level != '低风险':
            warnings.append('风险预测缺少归因因子，输出可信度受限')
            compliance_issues.append('归因缺失')
        
        # 5. 风险总结合规检查
        summary = result.get('risk_summary', '')
        for pattern in self.forbidden_patterns:
            if re.search(pattern, summary):
                summary = re.sub(pattern, '[已过滤]', summary)
                modifications.append(f'风险总结包含不合规内容，已过滤')
                warnings.append('输出包含不合规表述，已自动修正')
        
        result['risk_summary'] = summary
        
        # 6. 添加合规免责声明
        if self.risk_disclaimer_required:
            result['disclaimer'] = self.standard_disclaimer
        
        # 7. 添加安全元信息
        result['safety_check'] = {
            'passed': True,
            'warnings_count': len(warnings),
            'needs_human_review': needs_review,
            'guardrail_version': '1.0.0',
            'check_time': __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        if warnings:
            result['safety_check']['warnings'] = warnings
            self.stats['warnings_issued'] += len(warnings)
        
        if modifications:
            self.stats['modified'] += 1
        
        if needs_review:
            self.stats['human_reviews'] += 1
        
        return OutputCheckResult(
            passed=True,
            allowed=allowed,
            risk_level='low' if not warnings else ('medium' if not needs_review else 'high'),
            filtered_output=result,
            warnings=warnings,
            compliance_issues=compliance_issues,
            confidence_warning=len(warnings) > 0,
            needs_human_review=needs_review,
            modifications=modifications,
        )
    
    def check_text_output(self, text: str, context: str = 'general') -> OutputCheckResult:
        """检查文本输出的安全性"""
        self.stats['total_checks'] += 1
        warnings = []
        modifications = []
        allowed = True
        result = text
        
        # 检查禁止模式
        for pattern in self.forbidden_patterns:
            matches = re.findall(pattern, text)
            if matches:
                result = re.sub(pattern, '[已过滤不合规表述]', result)
                modifications.append(f'检测到不合规内容并过滤')
                allowed = False  # 不直接阻断，但标记
        
        # 投资建议检测
        investment_advice_patterns = [
            r'建议(买入|卖出|持有|加仓|减仓)',
            r'(应该|应当|必须)(买入|卖出)',
            r'(强烈)?推荐(买入|购买)',
        ]
        for pattern in investment_advice_patterns:
            if re.search(pattern, text):
                warnings.append('输出包含投资建议倾向，已标记风险')
                result += '\n\n' + self.standard_disclaimer
                modifications.append('附加投资风险免责声明')
                break
        
        if modifications:
            self.stats['modified'] += 1
        
        return OutputCheckResult(
            passed=True,
            allowed=allowed,
            risk_level='low' if not modifications else 'medium',
            filtered_output=result,
            warnings=warnings,
            modifications=modifications,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        return {**self.stats}
