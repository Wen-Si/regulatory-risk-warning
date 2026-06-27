#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safety Harness - 安全缰绳主控模块
集成输入护栏、输出护栏、审计日志，形成完整的安全合规体系。

遵循SafeHARNESS四层架构：
1. 输入处理层(Input Layer): 对抗性输入过滤、PII脱敏、注入检测
2. 决策层(Decision Layer): 置信度检查、权限控制、推理链审计
3. 执行层(Execution Layer): 工具调用防火墙、最小权限原则
4. 状态更新层(State Layer): 安全回滚、异常降级、审计记录
"""
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from .input_guard import InputGuardrail, InputCheckResult
from .output_guard import OutputGuardrail, OutputCheckResult
from .audit_logger import AuditLogger


@dataclass
class GuardrailResult:
    """完整护栏检查结果"""
    allowed: bool
    action: str  # allow/block/sanitize/review
    input_result: Optional[InputCheckResult] = None
    output_result: Optional[OutputCheckResult] = None
    audit_id: str = ''
    warnings: List[str] = None
    processing_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class SafetyHarness:
    """安全缰绳 - AI Agent安全合规主控
    
    使用方式：
    ```python
    harness = SafetyHarness()
    
    # 预测前检查输入
    input_check = harness.check_input(financial_data, announcement_text)
    if not input_check.passed:
        return error_response
    
    # 执行预测...
    
    # 预测后检查输出
    output_check = harness.check_output(prediction_result)
    safe_result = output_check.filtered_output
    
    # 记录审计
    harness.audit_prediction(company_code, input_data, safe_result)
    ```
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 初始化各组件
        self.input_guard = InputGuardrail(
            max_input_length=self.config.get('max_input_length', 50000),
            rate_limit_per_min=self.config.get('rate_limit_per_min', 60),
        )
        self.output_guard = OutputGuardrail(
            min_confidence=self.config.get('min_confidence', 0.3),
            max_risk_prob=self.config.get('max_risk_prob', 0.99),
        )
        self.audit = AuditLogger(
            log_dir=self.config.get('log_dir'),
        )
        
        # 安全策略配置
        self.policies = {
            'block_on_injection': True,       # 检测到注入直接阻断
            'mask_pii': True,                 # PII自动脱敏
            'require_disclaimer': True,       # 强制风险免责声明
            'log_all_predictions': True,      # 记录所有预测
            'log_security_events': True,      # 记录安全事件
            'human_review_threshold': 'high', # 高风险触发人工复核标记
            'enable_tool_firewall': True,     # 工具调用防火墙
        }
        
        # 权限矩阵（最小权限原则）
        self.permissions = {
            'public_user': {
                'max_companies_per_min': 10,
                'can_access_history': False,
                'can_export_data': False,
                'risk_level_access': ['low', 'medium', 'high'],
            },
            'authenticated_user': {
                'max_companies_per_min': 30,
                'can_access_history': True,
                'can_export_data': False,
                'risk_level_access': ['low', 'medium', 'high'],
            },
            'enterprise_user': {
                'max_companies_per_min': 100,
                'can_access_history': True,
                'can_export_data': True,
                'risk_level_access': ['low', 'medium', 'high', 'critical'],
            },
            'admin': {
                'max_companies_per_min': 1000,
                'can_access_history': True,
                'can_export_data': True,
                'can_view_audit': True,
                'risk_level_access': ['low', 'medium', 'high', 'critical'],
            },
        }
        
        # 异常降级计数器
        self._error_counts: Dict[str, int] = {}
        self._circuit_breaker = {
            'consecutive_errors': 0,
            'max_consecutive_errors': 5,
            'open': False,
            'open_since': None,
            'cooldown_seconds': 60,
        }
    
    def check_input(self, financial_data: Dict, announcement_text: str,
                    company_info: Dict = None, user_role: str = 'public_user',
                    session_id: str = '') -> InputCheckResult:
        """检查预测输入的安全性"""
        start = time.time()
        
        # 熔断器检查
        if self._circuit_breaker['open']:
            elapsed = time.time() - (self._circuit_breaker['open_since'] or 0)
            if elapsed < self._circuit_breaker['cooldown_seconds']:
                return InputCheckResult(
                    passed=False, blocked=True,
                    risk_level='high',
                    details='系统正在进行安全检查，请稍后再试',
                    detected_threats=['circuit_breaker_open']
                )
            else:
                # 冷却期结束，重置
                self._circuit_breaker['open'] = False
                self._circuit_breaker['consecutive_errors'] = 0
        
        # 检查公告文本（主要注入风险来源）
        text_check = self.input_guard.check(
            announcement_text, source='user', session_id=session_id
        )
        
        # 检查财务数据中的文本字段
        for key, value in financial_data.items():
            if isinstance(value, str) and len(value) > 10:
                field_check = self.input_guard.check(
                    value, source='user', session_id=session_id
                )
                if not field_check.passed:
                    text_check.passed = False
                    text_check.blocked = text_check.blocked or field_check.blocked
                    text_check.detected_threats.extend(
                        [f'{key}: {t}' for t in field_check.detected_threats]
                    )
        
        # 权限检查
        role_perms = self.permissions.get(user_role, self.permissions['public_user'])
        
        # 记录安全事件
        if text_check.detected_threats and self.policies['log_security_events']:
            self.audit.log_security_event(
                event_type='input_threat_detected',
                severity=text_check.risk_level,
                details={
                    'threats': text_check.detected_threats,
                    'blocked': text_check.blocked,
                    'pii_count': len(text_check.pii_found),
                    'user_role': user_role,
                }
            )
        
        return text_check
    
    def check_output(self, prediction: Dict) -> OutputCheckResult:
        """检查预测输出的安全性与合规性"""
        return self.output_guard.check_prediction(prediction)
    
    def check_text_output(self, text: str, context: str = 'general') -> OutputCheckResult:
        """检查文本输出的安全性"""
        return self.output_guard.check_text_output(text, context)
    
    def audit_prediction(self, company_code: str, company_info: Dict,
                         input_data: Dict, prediction_result: Dict,
                         safety_checks: Dict = None) -> str:
        """记录预测审计日志"""
        if not self.policies['log_all_predictions']:
            return ''
        return self.audit.log_prediction(
            company_code, company_info, input_data,
            prediction_result, safety_checks
        )
    
    def audit_agent_action(self, agent_name: str, action: str,
                           input_summary: str, output_summary: str,
                           duration_ms: float, success: bool):
        """记录Agent动作审计（推理链审计）"""
        self.audit.log_agent_action(
            agent_name, action, input_summary, output_summary,
            duration_ms, success
        )
    
    def check_tool_permission(self, tool_name: str, user_role: str,
                              params: Dict = None) -> Dict:
        """工具调用防火墙 - 检查工具调用权限"""
        result = {
            'allowed': True,
            'sanitized_params': params or {},
            'warnings': [],
        }
        
        role_perms = self.permissions.get(user_role, self.permissions['public_user'])
        
        # 高危工具需要企业用户以上权限
        high_risk_tools = ['export_data', 'bulk_analysis', 'admin_config']
        if tool_name in high_risk_tools and user_role not in ['enterprise_user', 'admin']:
            result['allowed'] = False
            result['warnings'].append(f'权限不足: {tool_name} 需要企业版权限')
            return result
        
        # 参数净化（防止注入）
        if params:
            sanitized = {}
            for k, v in params.items():
                if isinstance(v, str):
                    # 移除潜在的注入字符
                    v = v.replace('__', '_').replace('..', '.')
                    # 限制字符串长度
                    v = v[:1000]
                sanitized[k] = v
            result['sanitized_params'] = sanitized
        
        return result
    
    def record_error(self):
        """记录错误（用于熔断器）"""
        self._circuit_breaker['consecutive_errors'] += 1
        if self._circuit_breaker['consecutive_errors'] >= self._circuit_breaker['max_consecutive_errors']:
            self._circuit_breaker['open'] = True
            self._circuit_breaker['open_since'] = time.time()
            self.audit.log_security_event(
                event_type='circuit_breaker_opened',
                severity='high',
                details={'consecutive_errors': self._circuit_breaker['consecutive_errors']}
            )
    
    def record_success(self):
        """记录成功（重置熔断器）"""
        self._circuit_breaker['consecutive_errors'] = 0
        self._circuit_breaker['open'] = False
    
    def get_safety_report(self) -> Dict[str, Any]:
        """获取安全报告"""
        return {
            'input_guard_stats': self.input_guard.get_stats(),
            'output_guard_stats': self.output_guard.get_stats(),
            'audit_stats': self.audit.get_statistics(),
            'circuit_breaker': {
                'status': 'open' if self._circuit_breaker['open'] else 'closed',
                'consecutive_errors': self._circuit_breaker['consecutive_errors'],
            },
            'policies': self.policies,
        }
