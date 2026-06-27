#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智鉴风控 v3.0 - 基于Agentic AI + Harness Engineering + 知识图谱 + DL + RL的监管问询预警系统
主应用入口 - Flask托管前端+API

v3.0 全面升级：
- Harness Engineering安全层：输入护栏（注入检测/PII脱敏）、输出护栏（合规校验）、审计日志、工具防火墙
- 监管金融知识图谱：实体关系建模、Graph RAG、风险传播推理、证据链构建、幻觉抑制
- 深度学习：DeepFM（特征交叉）、Temporal Transformer（时序注意力）、GAT（图注意力网络）、RiskTextEncoder（文本编码）
- 强化学习：PPO（自适应阈值优化）、Thompson Sampling（集成权重学习）
- 混合架构：Safety+KG+DL+RL+规则引擎+LLM可解释性
"""

import os
import json
import random
import secrets
import requests
import sys
import time
import numpy as np
from datetime import datetime
from collections import defaultdict
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# 导入混合预测引擎 v3.0（默认启用安全层和知识图谱）
sys.path.insert(0, os.path.dirname(__file__))
from ml_engine import HybridPredictor
from market_data import (
    get_market_stats_summary,
    get_exchange_distribution,
    get_risk_type_distribution,
    MARKET_OVERVIEW,
    INQUIRY_STATS_2025,
    RISK_TYPE_DISTRIBUTION_2025,
)
predictor = HybridPredictor(use_rl=True, use_safety=True, use_kg=True)

app = Flask(__name__, static_folder='static', static_url_path='')

# 安全配置：CORS限制为同源
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5000", "http://127.0.0.1:5000"]}})

# 请求大小限制（1MB）
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

# 智谱AI配置 - 从环境变量读取，不提供默认值（防止密钥泄露）
ZHIPU_API_KEY = os.environ.get('ZHIPU_API_KEY', '')
ZHIPU_API_URL = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'

# 服务端速率限制（基于IP，不依赖客户端session_id）
_rate_limit_store = defaultdict(list)
RATE_LIMIT = 30  # 每分钟30次请求
RATE_WINDOW = 60  # 秒

def check_rate_limit(ip):
    """服务端IP速率限制"""
    now = time.time()
    window_start = now - RATE_WINDOW
    _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if t > window_start]
    if len(_rate_limit_store[ip]) >= RATE_LIMIT:
        return False
    _rate_limit_store[ip].append(now)
    return True

def get_client_ip():
    """获取客户端真实IP"""
    return request.remote_addr or 'unknown'

# 安全响应头中间件
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com cdn.tailwindcss.com; "
        "style-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com fonts.googleapis.com; "
        "font-src 'self' fonts.gstatic.com cdnjs.cloudflare.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'"
    )
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

# HTML转义工具函数（防止XSS）
def html_escape(text):
    """HTML实体转义"""
    if text is None:
        return ''
    if not isinstance(text, str):
        text = str(text)
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#x27;'))

# 真实上市公司数据（基于2026年公开披露）
MOCK_COMPANIES = {
    # 2026年已收到问询函的上市公司
    '600381': {'name': '青海春天', 'industry': '医药', 'market': '上交所', 'market_cap': 23},
    '605081': {'name': '太和水', 'industry': '环保', 'market': '上交所', 'market_cap': 8},
    '603843': {'name': '正平路桥', 'industry': '基建', 'market': '上交所', 'market_cap': 15},
    '601099': {'name': '太平洋证券', 'industry': '证券', 'market': '上交所', 'market_cap': 320},
    '688157': {'name': '松井股份', 'industry': '新材料', 'market': '上交所科创板', 'market_cap': 35},
    '688225': {'name': '亚信安全', 'industry': '网络安全', 'market': '上交所科创板', 'market_cap': 120},
    '688030': {'name': '山石网科', 'industry': '网络安全', 'market': '上交所科创板', 'market_cap': 28},
    '688302': {'name': '海创药业', 'industry': '生物医药', 'market': '上交所科创板', 'market_cap': 45},
    '688005': {'name': '容百科技', 'industry': '锂电池', 'market': '上交所科创板', 'market_cap': 180},
    '600481': {'name': '双良节能', 'industry': '节能环保', 'market': '上交所', 'market_cap': 95},
    '688209': {'name': '英集芯', 'industry': '半导体', 'market': '上交所科创板', 'market_cap': 65},
    # 蓝筹股
    '600000': {'name': '浦发银行', 'industry': '银行', 'market': '上交所', 'market_cap': 2800},
    '601318': {'name': '中国平安', 'industry': '保险', 'market': '上交所', 'market_cap': 8500},
    '000001': {'name': '平安银行', 'industry': '银行', 'market': '深交所', 'market_cap': 2200},
    '000858': {'name': '五粮液', 'industry': '白酒', 'market': '深交所', 'market_cap': 6800},
    '600519': {'name': '贵州茅台', 'industry': '白酒', 'market': '上交所', 'market_cap': 21000},
    '002594': {'name': '比亚迪', 'industry': '新能源汽车', 'market': '深交所', 'market_cap': 5500},
    '300750': {'name': '宁德时代', 'industry': '锂电池', 'market': '深交所', 'market_cap': 7200},
    '601899': {'name': '紫金矿业', 'industry': '有色金属', 'market': '上交所', 'market_cap': 3800},
    '600036': {'name': '招商银行', 'industry': '银行', 'market': '上交所', 'market_cap': 9500},
    '000333': {'name': '美的集团', 'industry': '家电', 'market': '深交所', 'market_cap': 4200},
    '688981': {'name': '中芯国际', 'industry': '半导体', 'market': '上交所科创板', 'market_cap': 4800},
    # 北交所
    '834765': {'name': '美之高', 'industry': '家居用品', 'market': '北交所', 'market_cap': 8},
    '430047': {'name': '诺思兰德', 'industry': '生物医药', 'market': '北交所', 'market_cap': 35},
    '835185': {'name': '贝特瑞', 'industry': '锂电池材料', 'market': '北交所', 'market_cap': 280},
}

RISK_CATEGORIES = [
    '财务异常', '信息披露矛盾', '关联交易', '资金占用', '担保事项',
    '并购重组', '业绩预告偏差', '会计处理争议', '经营波动', '股价异动',
    '股权质押', '违规担保', '商誉减值', '存货异常', '应收账款异常'
]

HISTORICAL_CASES = [
    # 2026年真实公开问询案例
    {
        'case_id': 'CASE2026001', 'company': '正平路桥', 'code': '603843', 'date': '2026-05-11',
        'reason': '2025年年报财务资料缺失及非标意见化解', 'risk_type': '信息披露违规',
        'key_points': ['财务资料缺失', '审计意见非标', '持续经营存疑', '内控有效性存疑'],
        'outcome': '上交所要求补充披露，上证公函【2026】0819号'
    },
    {
        'case_id': 'CASE2026002', 'company': '太平洋证券', 'code': '601099', 'date': '2026-05-27',
        'reason': '2025年年报业绩逆势下滑及减值原因', 'risk_type': '财务异常',
        'key_points': ['经营业绩大幅下滑', '信用业务风险', '债权投资减值异常', '资管业务风险'],
        'outcome': '上交所年报问询函，要求详细说明原因'
    },
    {
        'case_id': 'CASE2026003', 'company': '亚信安全', 'code': '688225', 'date': '2026-05-20',
        'reason': '2025年年报信息披露相关问题', 'risk_type': '信息披露违规',
        'key_points': ['年报信披不充分', '收入确认时点', '研发支出资本化', '客户集中度'],
        'outcome': '上交所科创板年报问询函，上证科创公函【2026】0195号'
    },
    {
        'case_id': 'CASE2026004', 'company': '松井股份', 'code': '688157', 'date': '2026-05-15',
        'reason': '2025年净利润大幅下滑76.66%及投资支出异常', 'risk_type': '业绩变脸',
        'key_points': ['归母净利润下滑76.66%', '单项坏账准备大幅上升', '投资支出大幅波动', '应收账款大额计提'],
        'outcome': '上交所科创板年报监管问询函'
    },
    {
        'case_id': 'CASE2026005', 'company': '山石网科', 'code': '688030', 'date': '2026-05-20',
        'reason': '2025年年报相关事项', 'risk_type': '财务异常',
        'key_points': ['审计意见非标', '收入确认合规性', '减值测试合理性', '内控缺陷'],
        'outcome': '上交所科创板年报问询函，会计师事务所专项核查'
    },
    {
        'case_id': 'CASE2026006', 'company': '海创药业', 'code': '688302', 'date': '2026-05-25',
        'reason': '研发管线与商业化进展披露充分性', 'risk_type': '信息披露违规',
        'key_points': ['研发管线披露不充分', '临床试验进展', '研发投入资本化', '商业化路径'],
        'outcome': '上交所科创板年报问询函'
    },
    {
        'case_id': 'CASE2026007', 'company': '青海春天', 'code': '600381', 'date': '2026-05-19',
        'reason': '*ST春天2025年年报有关事项', 'risk_type': '信息披露违规',
        'key_points': ['年报披露存疑', '持续经营能力', '财务数据真实性', '内控有效性'],
        'outcome': '上交所定期报告信息披露监管问询函'
    },
    {
        'case_id': 'CASE2026008', 'company': '太和水', 'code': '605081', 'date': '2026-04-19',
        'reason': '股票预计触及财务类终止上市情形', 'risk_type': '退市风险',
        'key_points': ['财务类退市指标', '净利润为负+营收不足', '审计意见非标', '持续经营能力存疑'],
        'outcome': '上交所监管工作函，关注退市风险'
    },
    {
        'case_id': 'CASE2026009', 'company': '双良节能', 'code': '600481', 'date': '2026-04-03',
        'reason': '误导性陈述', 'risk_type': '信息披露违规',
        'key_points': ['公告表述不准确', '重大事项误导', '信息披露不充分'],
        'outcome': '证监会行政处罚决定书'
    },
    {
        'case_id': 'CASE2026010', 'company': '英集芯', 'code': '688209', 'date': '2026-04-03',
        'reason': '误导性陈述', 'risk_type': '信息披露违规',
        'key_points': ['公告误导性表述', '重大事项披露不实', '投资者决策误导'],
        'outcome': '证监会行政处罚决定书'
    },
    {
        'case_id': 'CASE2026011', 'company': '容百科技', 'code': '688005', 'date': '2026-05-15',
        'reason': '5个交易日内回复交易所问询', 'risk_type': '信息披露违规',
        'key_points': ['问询回复及时性', '信息披露质量', '关联交易披露'],
        'outcome': '上交所问询函，已及时回复'
    },
    {
        'case_id': 'CASE2026012', 'company': '先歌国际', 'code': 'IPO', 'date': '2026-06-25',
        'reason': '北交所上市委第62次会议问询', 'risk_type': 'IPO审核',
        'key_points': ['经销收入真实性', '经销商分类毛利率', '销售价格公允性', '信用政策一致性'],
        'outcome': '北交所上市委审议通过'
    },
    # 历史经典案例
    {
        'case_id': 'CASE2024001', 'company': '某ST公司', 'date': '2023-08-15',
        'reason': '连续三年财务造假，虚增营收', 'risk_type': '财务异常',
        'key_points': ['营收增长率异常高于行业均值', '应收账款周转天数大幅增加', '经营现金流与净利润背离'],
        'outcome': '被实施退市风险警示'
    },
    {
        'case_id': 'CASE2024002', 'company': '某地产公司', 'date': '2024-03-15',
        'reason': '违规担保，资金占用', 'risk_type': '违规担保',
        'key_points': ['对外担保余额超净资产', '未履行审议程序', '控股股东资金占用'],
        'outcome': '立案调查，相关股东股份冻结'
    },
]


def call_zhipu_ai(messages, temperature=0.3, max_tokens=1500):
    """调用智谱AI，带安全检查"""
    if not ZHIPU_API_KEY:
        # 未配置API Key时返回降级提示
        return "[提示：未配置智谱AI API Key，LLM功能暂不可用。请设置ZHIPU_API_KEY环境变量。]"
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {ZHIPU_API_KEY}'}
    # 限制max_tokens防止滥用
    max_tokens = min(max_tokens, 1500)
    data = {'model': 'glm-4-flash', 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'stream': False}
    try:
        response = requests.post(ZHIPU_API_URL, headers=headers, json=data, timeout=15)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"智谱AI调用失败: {e}")
        return None


def generate_mock_financial_data(company_code):
    """生成模拟财务数据"""
    random.seed(hash(company_code) % 10000)
    has_risk = random.random() < 0.4
    
    # 根据市值和行业调整基准
    info = MOCK_COMPANIES.get(company_code, {'industry': '其他', 'market_cap': 100})
    is_st = 'ST' in info['name'] or '*ST' in info['name']
    
    if has_risk or is_st:
        return {
            'revenue_growth': random.uniform(-30, 80),
            'profit_growth': random.uniform(-50, 120),
            'roe': random.uniform(-10, 5),
            'debt_ratio': random.uniform(65, 90),
            'account_receivable_turnover': random.uniform(1, 4),
            'inventory_turnover': random.uniform(0.5, 2),
            'operating_cashflow_ratio': random.uniform(-0.5, 0.3),
            'goodwill_ratio': random.uniform(20, 50),
            'pledge_ratio': random.uniform(40, 80),
            'guarantee_ratio': random.uniform(30, 70),
        }, True
    else:
        return {
            'revenue_growth': random.uniform(-10, 30),
            'profit_growth': random.uniform(-15, 25),
            'roe': random.uniform(5, 20),
            'debt_ratio': random.uniform(30, 60),
            'account_receivable_turnover': random.uniform(5, 15),
            'inventory_turnover': random.uniform(3, 10),
            'operating_cashflow_ratio': random.uniform(0.8, 1.5),
            'goodwill_ratio': random.uniform(0, 15),
            'pledge_ratio': random.uniform(0, 20),
            'guarantee_ratio': random.uniform(0, 10),
        }, False


def generate_mock_announcements(code, info):
    """生成模拟公告文本"""
    announcements = [
        {'date': '2024-06-10', 'title': f"{info['name']}关于2023年年报的补充公告", 'type': '定期报告'},
        {'date': '2024-05-28', 'title': f"{info['name']}关于对外担保的公告", 'type': '担保公告'},
        {'date': '2024-05-15', 'title': f"{info['name']}2024年一季度报告", 'type': '定期报告'},
    ]
    random.seed(hash(code + 'ann') % 10000)
    if random.random() < 0.3:
        announcements.append({'date': '2024-04-20', 'title': f"{info['name']}关于关联交易的公告", 'type': '关联交易'})
    if random.random() < 0.2:
        announcements.append({'date': '2024-03-10', 'title': f"{info['name']}关于重大资产重组的提示性公告", 'type': '重组公告'})
    return announcements


def analyze_company(code, info):
    """使用DL+RL混合引擎分析公司"""
    logs = []
    def log(agent, action, detail):
        logs.append({'ts': datetime.now().strftime('%H:%M:%S.%f')[:-3], 'agent': agent, 'action': action, 'detail': detail})
    
    # Step 1: 数据获取
    log('公告研读Agent', '检索公告', f"正在获取{info['name']}({code})近期公告...")
    announcements = generate_mock_announcements(code, info)
    announcement_text = ' '.join([a['title'] for a in announcements])
    log('公告研读Agent', '完成', f"获取{len(announcements)}份重要公告")
    
    # Step 2: 财务数据
    log('财务检测Agent', '计算指标', '提取多维财务特征...')
    fin, has_risk = generate_mock_financial_data(code)
    log('财务检测Agent', '完成', f"财务特征提取完成，共24维数值特征+衍生交叉特征")
    
    # Step 3: 深度学习模型预测
    log('DeepFM模型', '推理中', 'DeepFM深度因子分解机进行特征交叉...')
    log('Temporal Transformer', '推理中', '时序Transformer注意力计算中...')
    log('GAT图神经网络', '推理中', '行业关联图注意力传播计算...')
    log('RiskTextEncoder', '推理中', '公告文本语义编码...')
    
    # 核心：调用混合预测引擎
    ml_result = predictor.predict(
        financial_data=fin,
        announcement_text=announcement_text,
        company_info={'code': code, **info},
        all_companies=MOCK_COMPANIES
    )
    
    log('深度学习引擎', '完成', f"DeepFM={ml_result['model_details']['deepfm_score']}%, "
        f"时序60d={ml_result['model_details']['temporal_scores']['60d']}%, "
        f"GAT={ml_result['model_details']['gat_contagion_score']}%, "
        f"文本={ml_result['model_details']['text_risk_score']}%")
    
    # Step 4: Thompson Sampling集成
    log('Thompson Sampling', '集成学习', '汤普森采样动态加权融合多模型输出...')
    weights_str = ', '.join([f"{k}={v:.1%}" for k, v in ml_result['model_details']['ensemble_weights'].items()])
    log('Thompson Sampling', '完成', f"集成权重: {weights_str}")
    
    # Step 5: PPO强化学习阈值优化
    log('PPO强化学习', '策略推理', 'PPO智能体自适应调整风险阈值...')
    rl_delta = ml_result['rl_adjustment']['threshold_delta']
    log('PPO强化学习', '完成', f"阈值调整: 30d{rl_delta[0]:+.3f}, 60d{rl_delta[1]:+.3f}, 90d{rl_delta[2]:+.3f}")
    
    if ml_result['rl_adjustment']['gating_reasons']:
        for reason in ml_result['rl_adjustment']['gating_reasons']:
            log('规则引擎Gating', '硬约束', reason)
    
    # Step 6: LLM可解释报告
    log('归因解释Agent', '归因分析', '生成多维度风险归因链条...')
    log('报告生成Agent', '生成', '输出可解释预警报告...')
    
    # 调用GLM-4.5-Flash生成自然语言解释（增强版prompt）
    ai_enhanced = _enhance_with_llm(code, info, fin, ml_result)
    
    # Step 7: 案例匹配
    log('案例检索Agent', '匹配案例', '检索历史相似问询案例...')
    cases = _match_similar_cases(ml_result)
    log('案例检索Agent', '完成', f"找到{len(cases)}个相似历史案例")
    
    return {
        'company_info': {**info, 'code': code},
        'financial_data': fin,
        'risk_factors': ml_result['top_risk_factors'],
        'ai_analysis': {
            'inquiry_probability_30d': ml_result['inquiry_probability_30d'],
            'inquiry_probability_60d': ml_result['inquiry_probability_60d'],
            'inquiry_probability_90d': ml_result['inquiry_probability_90d'],
            'risk_level': ml_result['risk_level'],
            'main_risk_types': ml_result['main_risk_types'],
            'risk_summary': ai_enhanced.get('summary', ml_result['risk_summary']),
            'key_evidence': ai_enhanced.get('evidence', ml_result['key_evidence']),
            'llm_insights': ai_enhanced.get('insights', ''),
        },
        'matched_cases': cases,
        'announcements': announcements,
        'reasoning_logs': logs,
        'ml_engine_result': ml_result,  # 完整ML结果
    }


def _enhance_with_llm(code, info, fin, ml_result):
    """使用GLM-4.5-Flash增强解释"""
    p30 = ml_result['inquiry_probability_30d']
    p60 = ml_result['inquiry_probability_60d']
    p90 = ml_result['inquiry_probability_90d']
    
    factors_str = '; '.join([f"{f['factor']}({f['source']})" for f in ml_result['top_risk_factors'][:5]])
    
    prompt = f"""你是资深金融风控专家，请基于以下AI模型预测结果，给出专业的风险解读。

公司：{info['name']}({code})，{info['industry']}行业，{info['market']}上市。

深度学习模型预测：
- 30天问询概率：{p30}%
- 60天问询概率：{p60}%（风险等级：{ml_result['risk_level']}）
- 90天问询概率：{p90}%

模型分解：
- DeepFM（特征交叉）：{ml_result['model_details']['deepfm_score']}%
- Temporal Transformer（时序60d）：{ml_result['model_details']['temporal_scores']['60d']}%
- GAT图网络（行业传染）：{ml_result['model_details']['gat_contagion_score']}%
- 文本编码器（公告语义）：{ml_result['model_details']['text_risk_score']}%
- 规则引擎：{ml_result['model_details']['rule_engine_score']}%

主要风险因子：{factors_str}

请用JSON格式输出：
{{"summary": "一段100字以内的专业风险总结", "evidence": ["证据1","证据2","证据3"], "insights": "一段150字以内的投资建议和合规提示"}}"""
    
    try:
        ai_text = call_zhipu_ai([{'role': 'user', 'content': prompt}], temperature=0.3, max_tokens=800)
        if ai_text:
            start = ai_text.find('{')
            end = ai_text.rfind('}')
            if start != -1 and end != -1:
                parsed = json.loads(ai_text[start:end+1])
                return {
                    'summary': parsed.get('summary', ml_result['risk_summary']),
                    'evidence': parsed.get('evidence', ml_result['key_evidence'])[:5],
                    'insights': parsed.get('insights', '')
                }
    except Exception as e:
        print(f"LLM增强失败: {e}")
    
    return {'summary': ml_result['risk_summary'], 'evidence': ml_result['key_evidence'], 'insights': ''}


def _match_similar_cases(ml_result):
    """基于风险类型匹配相似案例"""
    risk_types = set(ml_result['main_risk_types'])
    matched = []
    for case in HISTORICAL_CASES:
        if any(rt in case['risk_type'] or case['risk_type'] in rt for rt in risk_types):
            matched.append(case)
    if not matched:
        matched = random.sample(HISTORICAL_CASES, k=min(3, len(HISTORICAL_CASES)))
    return matched[:3]


# ============ 路由 ============

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/health')
def health():
    model_info = predictor.get_model_info()
    return jsonify({
        'status': 'ok', 
        'message': '智鉴风控API服务正常',
        'ai_model': 'GLM-4.5-Flash + DeepFM + Temporal Transformer + GAT + PPO',
        'version': model_info['version'],
        'architecture': model_info['architecture'],
        'models': model_info['models_used']
    })


@app.route('/api/companies/search')
def search_companies():
    kw = request.args.get('keyword', '').strip()
    return jsonify({'success': True, 'data': [{'code': c, **i} for c, i in MOCK_COMPANIES.items() if kw in c or kw in i['name']]})


@app.route('/api/companies/list')
def list_companies():
    market = request.args.get('market', 'all')
    return jsonify({'success': True, 'data': [{'code': c, **i} for c, i in MOCK_COMPANIES.items() if market == 'all' or i['market'] == market]})


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    code = request.json.get('company_code', '').strip()
    if not code or code not in MOCK_COMPANIES:
        return jsonify({'success': False, 'message': '公司代码无效，请输入正确的6位股票代码'}), 400
    return jsonify({'success': True, 'data': analyze_company(code, MOCK_COMPANIES[code])})


@app.route('/api/analyze/batch', methods=['POST'])
def api_batch():
    codes = request.json.get('company_codes', [])
    results = []
    for c in codes[:20]:
        if c in MOCK_COMPANIES:
            r = analyze_company(c, MOCK_COMPANIES[c])
            results.append({
                'code': c,
                'name': MOCK_COMPANIES[c]['name'],
                'industry': MOCK_COMPANIES[c]['industry'],
                'market': MOCK_COMPANIES[c]['market'],
                'probability_60d': r['ai_analysis']['inquiry_probability_60d'],
                'risk_level': r['ai_analysis']['risk_level'],
                'deepfm_score': r['ml_engine_result']['model_details']['deepfm_score'],
            })
    results.sort(key=lambda x: x['probability_60d'], reverse=True)
    return jsonify({'success': True, 'data': results})


@app.route('/api/risk/hot')
def hot_risks():
    results = []
    for c, i in MOCK_COMPANIES.items():
        r = analyze_company(c, i)
        results.append({
            'code': c, 'name': i['name'], 'industry': i['industry'], 'market': i['market'],
            'market_cap': i['market_cap'],
            'probability_30d': r['ai_analysis']['inquiry_probability_30d'],
            'probability_60d': r['ai_analysis']['inquiry_probability_60d'],
            'probability_90d': r['ai_analysis']['inquiry_probability_90d'],
            'risk_level': r['ai_analysis']['risk_level'],
            'main_risk_types': r['ai_analysis']['main_risk_types'][:3],
            'model_scores': {
                'deepfm': r['ml_engine_result']['model_details']['deepfm_score'],
                'temporal': r['ml_engine_result']['model_details']['temporal_scores']['60d'],
                'gat': r['ml_engine_result']['model_details']['gat_contagion_score'],
            }
        })
    results.sort(key=lambda x: x['probability_60d'], reverse=True)
    return jsonify({'success': True, 'data': results[:10]})


@app.route('/api/chat', methods=['POST'])
def chat():
    # 服务端速率限制
    ip = get_client_ip()
    if not check_rate_limit(ip):
        return jsonify({'success': False, 'message': '请求过于频繁，请稍后再试'}), 429
    
    q = request.json.get('question', '').strip()
    if not q:
        return jsonify({'success': False, 'message': '请输入问题'}), 400
    
    # 输入长度限制（防DoS）
    if len(q) > 500:
        return jsonify({'success': False, 'message': '问题长度不能超过500字符'}), 400
    
    # 使用安全护栏检查用户输入
    if predictor.safety_harness:
        input_check = predictor.safety_harness.check_input(
            financial_data={}, announcement_text=q,
            company_info={'code': 'chat'}, user_role='public_user',
            session_id=ip,
        )
        if input_check.blocked:
            return jsonify({
                'success': False,
                'message': '输入包含不当内容，请重新输入',
                'blocked': True
            }), 400
        # 使用脱敏后的文本
        q = input_check.sanitized_input if input_check.sanitized_input else q
    
    sys_msg = """你是一位专业的上市公司监管风控专家，精通证券法规、财务分析、深度学习风控模型和风险预警。
本系统采用DeepFM（深度因子分解机）、Temporal Transformer（时序Transformer）、GAT（图注意力网络）、PPO（强化学习）等前沿AI算法。
请用中文回答问题，回答专业、准确、有条理，每次回答控制在300字以内。
注意：不得提供投资建议，不得保证盈利，所有分析仅供参考。"""
    ans = call_zhipu_ai([{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': q}], 0.5, 800)
    default_ans = '作为AI风控助手（基于DeepFM+Transformer+GAT+PPO混合架构），我可以帮您分析上市公司的监管问询风险。核心深度学习模型包括：DeepFM捕捉财务特征交叉、Temporal Transformer建模时序依赖、GAT分析行业风险传导、PPO强化学习优化预警阈值。如需具体公司分析，请在"公司扫雷"页面输入股票代码。'
    
    # 对LLM输出进行基本安全过滤
    final_ans = ans or default_ans
    if final_ans:
        # 移除可能的恶意HTML标签
        import re
        final_ans = re.sub(r'<script[^>]*>.*?</script>', '', final_ans, flags=re.DOTALL)
        final_ans = re.sub(r'<[^>]+>', '', final_ans)
    
    return jsonify({'success': True, 'data': {'answer': final_ans}})


@app.route('/api/cases/history')
def history_cases():
    t = request.args.get('risk_type', '')
    cases = [c for c in HISTORICAL_CASES if not t or c['risk_type'] == t]
    return jsonify({'success': True, 'data': cases})


@app.route('/api/rule-engine/info')
def rule_engine_info():
    """返回规则库和ML模型元信息"""
    model_info = predictor.get_model_info()
    return jsonify({
        'success': True,
        'data': {
            'meta': predictor.rule_engine.rules['meta'],
            'rule_count': {
                'trigger_events': len(predictor.rule_engine.categories['trigger_events']['rules']),
                'financial_signals': len(predictor.rule_engine.categories['financial_signals']['rules']),
                'compliance_signals': len(predictor.rule_engine.categories['compliance_signals']['rules']),
            },
            'ml_models': model_info['models'],
            'rl_components': model_info['rl_components'],
            'ensemble_weights': model_info.get('ts_model_performance', {}),
            'online_learning': model_info['online_learning_stats'],
        }
    })


@app.route('/api/rule-engine/categories')
def rule_engine_categories():
    return jsonify({
        'success': True,
        'data': predictor.rule_engine.categories
    })


@app.route('/api/ml/model-info')
def ml_model_info():
    """返回ML模型详细信息"""
    info = predictor.get_model_info()
    return jsonify({'success': True, 'data': info})


@app.route('/api/ml/predict', methods=['POST'])
def ml_predict():
    """直接调用ML引擎预测（高级API）v3.0 - 含安全护栏与知识图谱"""
    # 服务端速率限制
    ip = get_client_ip()
    if not check_rate_limit(ip):
        return jsonify({'success': False, 'message': '请求过于频繁，请稍后再试'}), 429
    
    data = request.json
    financial_data = data.get('financial_data', {})
    announcement_text = data.get('announcement_text', '')
    
    # 输入长度限制
    if len(announcement_text) > 50000:
        return jsonify({'success': False, 'message': '公告文本过长'}), 400
    
    company_info = data.get('company_info', {})
    # user_role由服务端控制，不信任客户端传入
    user_role = 'public_user'
    session_id = ip  # 使用IP作为session_id进行速率限制
    
    result = predictor.predict(
        financial_data, announcement_text, company_info, MOCK_COMPANIES,
        session_id=session_id, user_role=user_role
    )
    return jsonify({'success': True, 'data': result})


@app.route('/api/safety/report')
def safety_report():
    """获取安全层统计报告"""
    if predictor.safety_harness:
        report = predictor.safety_harness.get_safety_report()
        return jsonify({'success': True, 'data': report})
    return jsonify({'success': False, 'message': '安全层未启用'})


@app.route('/api/kg/stats')
def kg_stats():
    """获取知识图谱统计"""
    if predictor.kg:
        stats = predictor.kg.get_stats()
        return jsonify({'success': True, 'data': stats})
    return jsonify({'success': False, 'message': '知识图谱未启用'})


@app.route('/api/ml/feedback', methods=['POST'])
def ml_feedback():
    """提供预测反馈用于在线学习"""
    data = request.json
    code = data.get('company_code', '')
    actual = data.get('actual_inquiry', False)
    risk_type = data.get('risk_type', '')
    
    # 重新获取预测结果
    if code in MOCK_COMPANIES:
        pred_result = analyze_company(code, MOCK_COMPANIES[code])
        predictor.provide_feedback(code, pred_result['ml_engine_result'], actual, risk_type)
        return jsonify({'success': True, 'message': '反馈已记录，模型将在线学习'})
    return jsonify({'success': False, 'message': '公司代码无效'}), 400


@app.route('/api/market/overview')
def market_overview():
    """全市场真实数据（基于中国上市公司协会2026年4月数据）"""
    return jsonify({
        'success': True,
        'data': {
            'market': MARKET_OVERVIEW,
            'inquiry_2025': INQUIRY_STATS_2025,
            'exchange_distribution': get_exchange_distribution(),
            'risk_type_distribution': get_risk_type_distribution(),
            'data_sources': {
                'market': '中国上市公司协会（CASA）2026年4月月报',
                'inquiry': '易董《2025年监管机构对上市公司及相关主体的监管处罚统计》',
                'bse': '北交所官方披露',
            },
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    })


@app.route('/api/dashboard/stats')
def stats():
    all_r = [analyze_company(c, i) for c, i in MOCK_COMPANIES.items()]
    high = sum(1 for r in all_r if r['ai_analysis']['risk_level'] == '高风险')
    mid = sum(1 for r in all_r if r['ai_analysis']['risk_level'] == '中风险')
    low = sum(1 for r in all_r if r['ai_analysis']['risk_level'] == '低风险')
    avg = sum(r['ai_analysis']['inquiry_probability_60d'] for r in all_r) / len(all_r)
    tc = {}
    for r in all_r:
        for t in r['ai_analysis']['main_risk_types']:
            tc[t] = tc.get(t, 0) + 1
    top = sorted(tc.items(), key=lambda x: x[1], reverse=True)[:5]

    # 模型平均得分
    avg_deepfm = sum(r['ml_engine_result']['model_details']['deepfm_score'] for r in all_r) / len(all_r)
    avg_temporal = sum(r['ml_engine_result']['model_details']['temporal_scores']['60d'] for r in all_r) / len(all_r)
    avg_gat = sum(r['ml_engine_result']['model_details']['gat_contagion_score'] for r in all_r) / len(all_r)

    # 合并全市场真实数据
    market_summary = get_market_stats_summary()

    return jsonify({'success': True, 'data': {
        # 系统内（演示用样本公司）风险分布
        'sample_companies': len(MOCK_COMPANIES),
        'high_risk_count': high,
        'medium_risk_count': mid,
        'low_risk_count': low,
        'avg_probability_60d': round(avg, 1),
        'top_risk_types': [{'type': k, 'count': v} for k, v in top],
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_architecture': 'DeepFM + Temporal Transformer + GAT + RiskTextEncoder + PPO + Thompson Sampling',
        'avg_model_scores': {
            'deepfm': round(avg_deepfm, 1),
            'temporal_transformer': round(avg_temporal, 1),
            'gat': round(avg_gat, 1),
        },
        # 全市场真实数据
        'market_real_data': market_summary,
        'risk_type_distribution_2025': RISK_TYPE_DISTRIBUTION_2025,
    }})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 60)
    print("智鉴风控 v3.0 - Safety Harness + Knowledge Graph + DL+RL")
    print("安全层: Harness Engineering (输入护栏/输出护栏/审计日志)")
    print("知识图谱: RegulatoryKG + Graph RAG (风险推理/证据链)")
    print("深度学习: DeepFM, Temporal Transformer, GAT, RiskTextEncoder")
    print("强化学习: PPO(阈值优化), Thompson Sampling(集成权重)")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False)
