#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit Logger - 审计日志
全链路审计记录，满足监管合规要求：
1. 所有预测请求与响应记录
2. 安全事件记录（注入尝试、PII检测、阻断事件）
3. 模型决策轨迹记录
4. 不可篡改的审计轨迹
5. 支持合规审计导出
"""
import json
import os
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from threading import Lock


class AuditLogger:
    """审计日志记录器
    
    遵循合规要求：
    - 每次预测完整记录输入、处理、输出
    - 安全事件详细记录威胁类型和处理动作
    - 日志不可篡改（追加写入）
    - 支持按时间/公司/风险等级检索
    """
    
    def __init__(self, log_dir: str = None, max_memory_entries: int = 10000):
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'logs'
            )
        self.log_dir = log_dir
        self.max_memory_entries = max_memory_entries
        self._lock = Lock()
        
        # 内存日志（用于快速查询）
        self._memory_log: List[Dict] = []
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        self.stats = {
            'total_entries': 0,
            'security_events': 0,
            'predictions_logged': 0,
        }
    
    def log_prediction(self, company_code: str, company_info: Dict,
                       input_data: Dict, prediction_result: Dict,
                       safety_checks: Dict = None) -> str:
        """
        记录一次预测的完整审计轨迹
        
        Returns:
            audit_id: 审计记录ID
        """
        audit_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now()
        
        entry = {
            'audit_id': audit_id,
            'timestamp': timestamp.isoformat(),
            'type': 'prediction',
            'company_code': company_code,
            'company_name': company_info.get('name', ''),
            'input_summary': {
                'has_financial_data': bool(input_data.get('financial_data')),
                'has_announcement': bool(input_data.get('announcement_text')),
                'announcement_length': len(input_data.get('announcement_text', '')),
            },
            'prediction_summary': {
                'risk_level': prediction_result.get('risk_level'),
                'prob_30d': prediction_result.get('inquiry_probability_30d'),
                'prob_60d': prediction_result.get('inquiry_probability_60d'),
                'prob_90d': prediction_result.get('inquiry_probability_90d'),
                'top_risk_factors_count': len(prediction_result.get('top_risk_factors', [])),
                'models_used': prediction_result.get('meta', {}).get('models_used', []),
                'inference_time_ms': prediction_result.get('meta', {}).get('inference_time_ms'),
            },
            'safety': safety_checks or {},
        }
        
        self._add_entry(entry)
        self.stats['predictions_logged'] += 1
        
        # 写入文件
        self._write_to_file('predictions', entry)
        
        return audit_id
    
    def log_security_event(self, event_type: str, severity: str,
                           details: Dict, source: str = 'input_guard') -> str:
        """记录安全事件"""
        event_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now()
        
        entry = {
            'event_id': event_id,
            'timestamp': timestamp.isoformat(),
            'type': 'security_event',
            'event_type': event_type,
            'severity': severity,
            'source': source,
            'details': details,
        }
        
        self._add_entry(entry)
        self.stats['security_events'] += 1
        
        self._write_to_file('security', entry)
        
        return event_id
    
    def log_feedback(self, company_code: str, prediction_id: str,
                     actual_inquiry: bool, feedback_source: str = 'user'):
        """记录反馈事件"""
        entry = {
            'feedback_id': str(uuid.uuid4())[:8],
            'timestamp': datetime.now().isoformat(),
            'type': 'feedback',
            'company_code': company_code,
            'prediction_id': prediction_id,
            'actual_inquiry': actual_inquiry,
            'feedback_source': feedback_source,
        }
        
        self._add_entry(entry)
        self._write_to_file('feedback', entry)
    
    def log_agent_action(self, agent_name: str, action: str,
                         input_summary: str, output_summary: str,
                         duration_ms: float, success: bool):
        """记录Agent动作（用于推理链审计）"""
        entry = {
            'action_id': str(uuid.uuid4())[:8],
            'timestamp': datetime.now().isoformat(),
            'type': 'agent_action',
            'agent_name': agent_name,
            'action': action,
            'input_summary': input_summary[:200],
            'output_summary': output_summary[:500],
            'duration_ms': duration_ms,
            'success': success,
        }
        
        self._add_entry(entry)
    
    def get_recent_entries(self, limit: int = 50, entry_type: str = None) -> List[Dict]:
        """获取最近的日志条目"""
        with self._lock:
            entries = self._memory_log
            if entry_type:
                entries = [e for e in entries if e.get('type') == entry_type]
            return entries[-limit:]
    
    def get_security_events(self, since: str = None, severity: str = None) -> List[Dict]:
        """获取安全事件"""
        events = [e for e in self._memory_log if e.get('type') == 'security_event']
        if severity:
            events = [e for e in events if e.get('severity') == severity]
        return events
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取审计统计"""
        with self._lock:
            total = len(self._memory_log)
            by_type = {}
            by_severity = {}
            for e in self._memory_log:
                t = e.get('type', 'unknown')
                by_type[t] = by_type.get(t, 0) + 1
                if e.get('severity'):
                    s = e['severity']
                    by_severity[s] = by_severity.get(s, 0) + 1
            
            return {
                'total_entries': total,
                'by_type': by_type,
                'by_severity': by_severity,
                'stats': {**self.stats},
            }
    
    def _add_entry(self, entry: Dict):
        """添加条目到内存日志"""
        with self._lock:
            self._memory_log.append(entry)
            self.stats['total_entries'] += 1
            # 防止内存溢出
            if len(self._memory_log) > self.max_memory_entries:
                self._memory_log = self._memory_log[-self.max_memory_entries:]
    
    def _write_to_file(self, category: str, entry: Dict):
        """追加写入日志文件"""
        try:
            date_str = datetime.now().strftime('%Y-%m-%d')
            filename = f'audit_{category}_{date_str}.jsonl'
            filepath = os.path.join(self.log_dir, filename)
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            # 审计失败不应静默，记录到stderr
            import sys
            print(f'[AUDIT ERROR] Failed to write audit log: {e}', file=sys.stderr)
