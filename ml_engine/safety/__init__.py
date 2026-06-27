#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智鉴风控 - Harness Engineering 安全合规层

基于Harness Engineering(缰绳工程)方法论构建的AI Agent安全合规框架：
1. 输入护栏(Input Guardrails): 输入验证、注入检测、PII脱敏
2. 决策护栏(Decision Guardrails): 推理链审计、权限检查、最小特权
3. 输出护栏(Output Guardrails): 内容过滤、合规校验、置信度门控
4. 监控审计层(Monitoring): 全链路日志、审计轨迹、异常检测
5. 工具防火墙(Tool Firewall): 工具输入净化、工具输出消毒

参考架构：
- SafeHARNESS: Lifecycle-Integrated Security Architecture
- NVIDIA NeMo Guardrails三阶段模型
- LlamaFirewall思维链审计
- AWS/OpenAI输入输出护栏最佳实践
"""

from .guardrails import SafetyHarness, GuardrailResult
from .input_guard import InputGuardrail
from .output_guard import OutputGuardrail
from .audit_logger import AuditLogger

__all__ = [
    'SafetyHarness',
    'GuardrailResult',
    'InputGuardrail',
    'OutputGuardrail',
    'AuditLogger',
]
