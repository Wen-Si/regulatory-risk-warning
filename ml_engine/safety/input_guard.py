#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input Guardrail - 输入护栏
负责在输入到达模型/Agent前进行安全检查：
1. 输入格式验证与清洗
2. 提示注入(Prompt Injection)检测
3. PII(个人身份信息)检测与脱敏
4. 敏感关键词过滤
5. 速率限制
6. 输入大小限制
"""
import re
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class InputCheckResult:
    """输入检查结果"""
    passed: bool
    risk_level: str = 'low'  # low/medium/high/critical
    blocked: bool = False
    sanitized_input: str = ''
    detected_threats: List[str] = field(default_factory=list)
    pii_found: List[Dict] = field(default_factory=list)
    confidence: float = 1.0
    details: str = ''


class InputGuardrail:
    """输入护栏 - 前处理安全层
    
    实现多层次防御：
    - L1: 静态规则过滤（正则/关键词黑名单）
    - L2: 启发式检测（模式匹配/异常检测）
    - L3: 语义注入检测（基于规则的分类器）
    """
    
    def __init__(self, max_input_length: int = 50000, rate_limit_per_min: int = 60):
        self.max_input_length = max_input_length
        self.rate_limit_per_min = rate_limit_per_min
        
        # 提示注入模式（直接注入）
        self.direct_injection_patterns = [
            r'ignore\s+(all\s+)?(previous|prior|above|system)\s+(instructions?|prompts?|messages?)',
            r'disregard\s+(all\s+)?(previous|prior|above|system)',
            r'forget\s+(all\s+)?(previous|prior|above|system|your)\s+(instructions?|prompts?|rules?)',
            r'override\s+(system|safety|security|alignment)',
            r'you\s+are\s+now\s+(a|an|DAN|unrestricted|unfiltered)',
            r'jailbreak|DAN mode|do anything now',
            r'忽略(之前|以上|所有|系统)(的)?(指令|提示|规则|约束)',
            r'无视(之前|以上|所有|系统)(的)?(指令|提示|规则|安全)',
            r'你现在(是|扮演)(一个)?(无限制|无审查|越狱|黑客)',
            r'解除(所有|安全|内容)(限制|过滤|约束)',
            r'system\s*prompt.*(?:reveal|show|print|output|disclose)',
            r'reveal\s+(your|the)\s+(system|secret|hidden)\s+(prompt|instructions?)',
        ]
        
        # 间接注入模式（隐藏在数据中）
        self.indirect_injection_patterns = [
            r'<\s*(?:script|iframe|object|embed|form)',  # HTML注入
            r'javascript\s*:',
            r'on(?:load|error|click|mouseover)\s*=',
            r'eval\s*\(.*\)',
            r'exec\s*\(.*\)',
            r'__import__\s*\(.*\)',
            r'os\.system\s*\(.*\)',
            r'subprocess\.(?:call|Popen|run)',
        ]
        
        # PII模式
        self.pii_patterns = {
            'id_card': r'[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]',
            'phone': r'(?:\+?86)?1[3-9]\d{9}',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'bank_card': r'62[0-9]{14,17}',
        }
        
        # 金融敏感关键词（非禁止但需标记）
        self.sensitive_keywords = [
            '内幕交易', '操纵市场', '老鼠仓', '坐庄', '洗钱',
            '财务造假', '虚增收入', '虚增利润', '账外账', '两套账',
            '利益输送', '掏空上市公司',
        ]
        
        # 禁止内容（直接阻断）
        self.blocked_keywords = [
            '如何造假', '如何规避监管', '如何操纵', '如何洗钱',
            '如何逃税', '内幕信息',
        ]
        
        # 速率限制
        self._rate_window: Dict[str, List[float]] = {}
        
        # 统计
        self.stats = {
            'total_checks': 0,
            'blocked': 0,
            'pii_detected': 0,
            'injection_attempts': 0,
            'sanitized': 0,
        }
    
    def check(self, text: str, source: str = 'user', 
              session_id: str = '', context: Dict = None) -> InputCheckResult:
        """
        对输入进行完整安全检查
        
        Args:
            text: 输入文本
            source: 输入来源（user/system/tool/web/document）
            session_id: 会话ID（用于速率限制）
            context: 上下文信息
            
        Returns:
            InputCheckResult
        """
        self.stats['total_checks'] += 1
        threats = []
        pii_found = []
        blocked = False
        risk_level = 'low'
        sanitized = text
        
        if not text:
            return InputCheckResult(
                passed=True, sanitized_input='',
                details='空输入'
            )
        
        # L0: 长度检查
        if len(text) > self.max_input_length:
            threats.append(f'输入长度超限: {len(text)} > {self.max_input_length}')
            sanitized = text[:self.max_input_length]
            risk_level = 'medium'
        
        # L0: 速率限制
        if session_id and not self._check_rate_limit(session_id):
            threats.append('请求频率超限')
            blocked = True
            risk_level = 'high'
        
        # L1: 直接注入检测
        direct_matches = self._check_direct_injection(text)
        if direct_matches:
            threats.extend([f'提示注入检测: {m}' for m in direct_matches])
            self.stats['injection_attempts'] += 1
            risk_level = 'high'
            blocked = True  # 直接注入直接阻断
        
        # L1: 间接注入检测（所有来源均检测，纵深防御）
        indirect_matches = self._check_indirect_injection(text)
        if indirect_matches:
            threats.extend([f'间接注入检测: {m}' for m in indirect_matches])
            risk_level = max(risk_level, 'medium')
            sanitized = self._sanitize_injection(sanitized, indirect_matches)
            self.stats['sanitized'] += 1
        
        # L2: PII检测与脱敏
        pii_found = self._detect_pii(text)
        if pii_found:
            threats.append(f'检测到{len(pii_found)}处PII信息')
            sanitized = self._mask_pii(sanitized, pii_found)
            self.stats['pii_detected'] += 1
            risk_level = max(risk_level, 'medium')
        
        # L2: 禁止关键词检测
        blocked_matches = self._check_blocked_keywords(text)
        if blocked_matches:
            threats.extend([f'禁止内容: {k}' for k in blocked_matches])
            blocked = True
            risk_level = 'critical'
        
        # L2: 敏感关键词标记（不阻断但标记）
        sensitive_matches = self._check_sensitive_keywords(text)
        if sensitive_matches:
            threats.append(f'敏感关键词: {", ".join(sensitive_matches[:3])}')
            risk_level = max(risk_level, 'medium')
        
        if blocked:
            self.stats['blocked'] += 1
        
        return InputCheckResult(
            passed=not blocked,
            risk_level=risk_level,
            blocked=blocked,
            sanitized_input=sanitized,
            detected_threats=threats,
            pii_found=pii_found,
            confidence=0.95 if blocked else (0.8 if risk_level == 'medium' else 0.6),
            details=f'检测到{len(threats)}个安全问题' if threats else '输入安全检查通过'
        )
    
    def _check_direct_injection(self, text: str) -> List[str]:
        """检测直接提示注入"""
        matches = []
        text_lower = text.lower()
        for pattern in self.direct_injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches.append(pattern[:30] + '...')
        return matches
    
    def _check_indirect_injection(self, text: str) -> List[str]:
        """检测间接注入（代码/HTML/命令注入）"""
        matches = []
        for pattern in self.indirect_injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern[:30])
        return matches
    
    def _detect_pii(self, text: str) -> List[Dict]:
        """检测PII信息"""
        found = []
        for pii_type, pattern in self.pii_patterns.items():
            for m in re.finditer(pattern, text):
                found.append({
                    'type': pii_type,
                    'value': m.group(),
                    'start': m.start(),
                    'end': m.end(),
                })
        return found
    
    def _mask_pii(self, text: str, pii_list: List[Dict]) -> str:
        """脱敏PII信息"""
        result = text
        # 从后向前替换，避免位置偏移
        for pii in sorted(pii_list, key=lambda x: -x['start']):
            v = pii['value']
            if pii['type'] == 'id_card':
                masked = v[:6] + '********' + v[-4:]
            elif pii['type'] == 'phone':
                masked = v[:3] + '****' + v[-4:]
            elif pii['type'] == 'email':
                at_pos = v.index('@')
                masked = v[:2] + '***' + v[at_pos:]
            elif pii['type'] == 'bank_card':
                masked = v[:4] + '****' + v[-4:]
            else:
                masked = '***'
            result = result[:pii['start']] + masked + result[pii['end']:]
        return result
    
    def _sanitize_injection(self, text: str, patterns: List[str]) -> str:
        """净化注入内容"""
        result = text
        for pattern in self.indirect_injection_patterns:
            result = re.sub(pattern, '[FILTERED]', result, flags=re.IGNORECASE)
        return result
    
    def _check_blocked_keywords(self, text: str) -> List[str]:
        """检测禁止关键词"""
        return [k for k in self.blocked_keywords if k in text]
    
    def _check_sensitive_keywords(self, text: str) -> List[str]:
        """检测敏感关键词"""
        return [k for k in self.sensitive_keywords if k in text]
    
    def _check_rate_limit(self, session_id: str) -> bool:
        """检查速率限制"""
        now = time.time()
        if session_id not in self._rate_window:
            self._rate_window[session_id] = []
        
        # 清理过期记录
        window_start = now - 60
        self._rate_window[session_id] = [
            t for t in self._rate_window[session_id] if t > window_start
        ]
        
        if len(self._rate_window[session_id]) >= self.rate_limit_per_min:
            return False
        
        self._rate_window[session_id].append(now)
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {**self.stats}
