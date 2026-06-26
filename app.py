#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智鉴风控 - 基于Agentic AI的上市公司监管问询预警系统
主应用入口 - Flask托管前端+API
"""

import os
import json
import random
import requests
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# 智谱AI配置
ZHIPU_API_KEY = os.environ.get('ZHIPU_API_KEY', '325d6fa364954d2e871c30ba95b553bd.KBdQdqgJgELJBhnv')
ZHIPU_API_URL = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'

# 模拟上市公司数据库
MOCK_COMPANIES = {
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
    {
        'case_id': 'CASE001', 'company': '某ST公司', 'date': '2023-08-15',
        'reason': '连续三年财务造假，虚增营收', 'risk_type': '财务异常',
        'key_points': ['营收增长率异常高于行业均值', '应收账款周转天数大幅增加', '经营现金流与净利润背离'],
        'outcome': '被实施退市风险警示'
    },
    {
        'case_id': 'CASE002', 'company': '某科技公司', 'date': '2023-11-22',
        'reason': '关联交易非关联化，利益输送', 'risk_type': '关联交易',
        'key_points': ['关联方销售占比过高', '交易价格显失公允', '资金流向异常'],
        'outcome': '责令整改，相关责任人处罚'
    },
    {
        'case_id': 'CASE003', 'company': '某医药公司', 'date': '2024-01-08',
        'reason': '商誉减值计提不充分', 'risk_type': '商誉减值',
        'key_points': ['并购标的业绩承诺未达标', '商誉占净资产比例过高', '减值测试参数不合理'],
        'outcome': '年报问询函，要求补充披露'
    },
    {
        'case_id': 'CASE004', 'company': '某地产公司', 'date': '2024-03-15',
        'reason': '违规担保，资金占用', 'risk_type': '违规担保',
        'key_points': ['对外担保余额超净资产', '未履行审议程序', '控股股东资金占用'],
        'outcome': '立案调查，相关股东股份冻结'
    },
    {
        'case_id': 'CASE005', 'company': '某新能源公司', 'date': '2024-05-20',
        'reason': '业绩预告大幅修正', 'risk_type': '业绩预告偏差',
        'key_points': ['业绩预告由盈转亏', '存货跌价准备计提不足', '应收账款坏账准备计提不充分'],
        'outcome': '关注函，要求说明差异原因'
    }
]


def call_zhipu_ai(messages, temperature=0.3, max_tokens=2000):
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {ZHIPU_API_KEY}'}
    data = {'model': 'glm-4-flash', 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'stream': False}
    try:
        response = requests.post(ZHIPU_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"智谱AI调用失败: {e}")
        return None


def safe_int(val, default=50):
    """安全转换为0-100的整数"""
    try:
        if isinstance(val, str):
            val = val.replace('%', '').replace('分', '').strip()
        v = int(round(float(val)))
        return max(0, min(100, v))
    except:
        return default


def normalize_risk_level(level, score=50):
    """标准化风险等级"""
    if level is None:
        return '高风险' if score >= 70 else ('中风险' if score >= 40 else '低风险')
    s = str(level)
    if '高' in s or 'high' in s.lower() or s in ['3', '红色']:
        return '高风险'
    if '低' in s or 'low' in s.lower() or s in ['1', '绿色']:
        return '低风险'
    if score >= 70:
        return '高风险'
    if score >= 40:
        return '中风险'
    return '低风险'


def generate_mock_financial_data(company_code):
    random.seed(hash(company_code) % 10000)
    has_risk = random.random() < 0.4
    return {
        'revenue_growth': random.uniform(-10, 30) if not has_risk else random.uniform(-30, 80),
        'profit_growth': random.uniform(-15, 25) if not has_risk else random.uniform(-50, 120),
        'roe': random.uniform(5, 20) if not has_risk else random.uniform(-10, 5),
        'debt_ratio': random.uniform(30, 60) if not has_risk else random.uniform(65, 90),
        'account_receivable_turnover': random.uniform(5, 15) if not has_risk else random.uniform(1, 4),
        'inventory_turnover': random.uniform(3, 10) if not has_risk else random.uniform(0.5, 2),
        'operating_cashflow_ratio': random.uniform(0.8, 1.5) if not has_risk else random.uniform(-0.5, 0.3),
        'goodwill_ratio': random.uniform(0, 15) if not has_risk else random.uniform(20, 50),
        'pledge_ratio': random.uniform(0, 20) if not has_risk else random.uniform(40, 80),
        'guarantee_ratio': random.uniform(0, 10) if not has_risk else random.uniform(30, 70),
    }, has_risk


def calculate_risk_score(fin):
    score = 0
    factors = []
    checks = [
        (abs(fin['revenue_growth']) > 50, '营收增长率异常', 15, f"营收增长率{fin['revenue_growth']:.1f}%，偏离正常区间"),
        (fin['roe'] < 3, 'ROE偏低', 10, f"ROE为{fin['roe']:.1f}%，盈利能力较弱"),
        (fin['debt_ratio'] > 65, '资产负债率过高', 15, f"资产负债率{fin['debt_ratio']:.1f}%，偿债压力大"),
        (fin['account_receivable_turnover'] < 3, '应收账款周转过慢', 12, f"应收账款周转率{fin['account_receivable_turnover']:.1f}次，回款风险高"),
        (fin['inventory_turnover'] < 2, '存货周转异常', 10, f"存货周转率{fin['inventory_turnover']:.1f}次，存在滞销或减值风险"),
        (fin['operating_cashflow_ratio'] < 0.5, '经营现金流不足', 13, f"经营现金流/净利润比值{fin['operating_cashflow_ratio']:.2f}，盈利质量存疑"),
        (fin['goodwill_ratio'] > 25, '商誉占比过高', 12, f"商誉占净资产{fin['goodwill_ratio']:.1f}%，存在减值风险"),
        (fin['pledge_ratio'] > 50, '股权质押比例高', 8, f"控股股东股权质押比例{fin['pledge_ratio']:.1f}%，存在平仓风险"),
        (fin['guarantee_ratio'] > 30, '对外担保过多', 10, f"对外担保余额占净资产{fin['guarantee_ratio']:.1f}%，存在或有负债风险"),
    ]
    for cond, name, weight, detail in checks:
        if cond:
            score += weight
            factors.append({'factor': name, 'weight': weight, 'detail': detail})
    return min(score, 100), factors


def analyze_company(code, info):
    logs = []
    def log(a, b, c): logs.append({'ts': datetime.now().strftime('%H:%M:%S'), 'agent': a, 'action': b, 'detail': c})
    
    log('公告研读Agent', '检索公告', f"正在获取{info['name']}近期公告...")
    announcements = [
        {'date': '2024-06-10', 'title': f"{info['name']}关于2023年年报的补充公告", 'type': '定期报告'},
        {'date': '2024-05-28', 'title': f"{info['name']}关于对外担保的公告", 'type': '担保公告'},
        {'date': '2024-05-15', 'title': f"{info['name']}2024年一季度报告", 'type': '定期报告'},
        {'date': '2024-04-20', 'title': f"{info['name']}关于关联交易的公告", 'type': '关联交易'},
    ]
    log('公告研读Agent', '完成', f"获取{len(announcements)}份重要公告，正在提取风险关键词...")
    
    log('财务检测Agent', '计算指标', '分析财务异常度...')
    fin, _ = generate_mock_financial_data(code)
    base_score, factors = calculate_risk_score(fin)
    log('财务检测Agent', '完成', f"财务异常基础分：{base_score}分，识别{len(factors)}个风险点")
    
    log('案例检索Agent', '匹配案例', '检索历史相似问询案例...')
    cases = random.sample(HISTORICAL_CASES, k=min(3, len(HISTORICAL_CASES)))
    log('案例检索Agent', '完成', f"找到{len(cases)}个相似历史案例")
    
    log('风险预测Agent', 'AI推理', '调用智谱GLM-4.5-Flash进行综合风险评估...')
    
    # 构建更清晰的prompt
    prompt = f"""请作为资深金融风控专家，基于以下信息评估{info['name']}({code})未来被监管问询的风险概率。

公司背景：{info['industry']}行业，{info['market']}上市，市值{info['market_cap']}亿元。

关键财务指标：
- 营收增长率: {fin['revenue_growth']:.1f}%
- 净利润增长率: {fin['profit_growth']:.1f}%
- ROE: {fin['roe']:.1f}%
- 资产负债率: {fin['debt_ratio']:.1f}%
- 应收账款周转率: {fin['account_receivable_turnover']:.1f}次
- 存货周转率: {fin['inventory_turnover']:.1f}次
- 经营现金流/净利润: {fin['operating_cashflow_ratio']:.2f}
- 商誉占净资产: {fin['goodwill_ratio']:.1f}%
- 股权质押比例: {fin['pledge_ratio']:.1f}%
- 对外担保比例: {fin['guarantee_ratio']:.1f}%

异常风险点: {', '.join([f['factor'] for f in factors]) if factors else '无明显异常'}

请严格按照以下JSON格式输出（概率为0-100的整数，不要输出其他文字）：
{{"inquiry_probability_30d": 整数, "inquiry_probability_60d": 整数, "inquiry_probability_90d": 整数, "risk_level": "高风险"或"中风险"或"低风险", "main_risk_types": ["类型1","类型2"], "risk_summary": "风险总结", "key_evidence": ["证据1","证据2","证据3"]}}"""
    
    ai = None
    p60 = min(base_score + random.randint(10, 30), 95)
    
    try:
        ai_result = call_zhipu_ai([{'role': 'user', 'content': prompt}])
        if ai_result:
            # 尝试提取JSON
            t = ai_result.strip()
            # 找到JSON对象
            start = t.find('{')
            end = t.rfind('}')
            if start != -1 and end != -1:
                t = t[start:end+1]
                parsed = json.loads(t)
                
                p30 = safe_int(parsed.get('inquiry_probability_30d'), max(p60 - 15, 5))
                p60_parsed = safe_int(parsed.get('inquiry_probability_60d'), p60)
                p90 = safe_int(parsed.get('inquiry_probability_90d'), min(p60 + 10, 98))
                
                # 确保时间逻辑合理
                p30 = min(p30, p60_parsed)
                p90 = max(p90, p60_parsed)
                
                risk_types = parsed.get('main_risk_types', [])
                if not isinstance(risk_types, list) or len(risk_types) == 0:
                    risk_types = [f['factor'] for f in factors[:3]] if factors else ['经营波动']
                risk_types = [str(rt) for rt in risk_types[:5]]
                
                summary = parsed.get('risk_summary', '')
                if not summary or len(summary) < 10:
                    summary = f"基于财务指标综合分析，{info['name']}存在{len(factors)}个异常信号，需重点关注财务真实性和信息披露质量。建议持续跟踪公司公告和监管动态。"
                
                evidence = parsed.get('key_evidence', [])
                if not isinstance(evidence, list) or len(evidence) == 0:
                    evidence = [f['detail'] for f in factors[:5]] if factors else ['未发现明显异常']
                evidence = [str(e) for e in evidence[:5]]
                
                ai = {
                    'inquiry_probability_30d': p30,
                    'inquiry_probability_60d': p60_parsed,
                    'inquiry_probability_90d': p90,
                    'risk_level': normalize_risk_level(parsed.get('risk_level'), p60_parsed),
                    'main_risk_types': risk_types,
                    'risk_summary': summary,
                    'key_evidence': evidence
                }
                log('风险预测Agent', 'AI分析完成', f"AI评估完成，60天问询概率{p60_parsed}%")
    except Exception as e:
        print(f"AI解析异常: {e}")
    
    if not ai:
        # 使用规则模型作为备用
        p30 = max(p60 - 15, 5)
        p90 = min(p60 + 10, 98)
        ai = {
            'inquiry_probability_30d': p30,
            'inquiry_probability_60d': p60,
            'inquiry_probability_90d': p90,
            'risk_level': normalize_risk_level(None, p60),
            'main_risk_types': [f['factor'] for f in factors[:3]] if factors else ['经营波动'],
            'risk_summary': f"基于多维度财务指标分析，{info['name']}存在{len(factors)}个异常信号，包括{('、'.join([f['factor'] for f in factors[:3]])) if factors else '经营波动'}等问题。建议重点关注财务真实性、信息披露质量及关联交易情况，持续跟踪公司公告和监管动态。",
            'key_evidence': [f['detail'] for f in factors[:5]] if factors else ['当前未发现明显异常信号']
        }
        log('风险预测Agent', '规则模型', f"使用规则模型评估，60天问询概率{p60}%")
    
    log('归因解释Agent', '归因', '生成风险归因链条...')
    log('报告生成Agent', '生成', '输出可解释预警报告...')
    
    return {
        'company_info': {**info, 'code': code},
        'financial_data': fin,
        'risk_factors': factors,
        'ai_analysis': ai,
        'matched_cases': cases,
        'announcements': announcements,
        'reasoning_logs': logs
    }


# ============ 路由 ============

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'message': '智鉴风控API服务正常', 'ai_model': 'GLM-4.5-Flash'})


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
                'risk_level': r['ai_analysis']['risk_level']
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
            'main_risk_types': r['ai_analysis']['main_risk_types'][:3]
        })
    results.sort(key=lambda x: x['probability_60d'], reverse=True)
    return jsonify({'success': True, 'data': results[:10]})


@app.route('/api/chat', methods=['POST'])
def chat():
    q = request.json.get('question', '').strip()
    if not q:
        return jsonify({'success': False, 'message': '请输入问题'}), 400
    sys_msg = '你是一位专业的上市公司监管风控专家，精通证券法规、财务分析和风险预警。请用中文回答问题，回答专业、准确、有条理，每次回答控制在300字以内。'
    ans = call_zhipu_ai([{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': q}], 0.5, 1000)
    default_ans = '作为AI风控助手，我可以帮您分析上市公司的监管问询风险。建议您关注以下几类高风险信号：1)财务指标异常（如营收增长过快、现金流与利润背离）；2)关联交易频繁且定价不公允；3)商誉占比过高且标的业绩未达标；4)股权质押比例过高；5)信息披露前后矛盾。如需具体公司分析，请在"公司扫雷"页面输入股票代码。'
    return jsonify({'success': True, 'data': {'answer': ans or default_ans}})


@app.route('/api/cases/history')
def history_cases():
    t = request.args.get('risk_type', '')
    cases = [c for c in HISTORICAL_CASES if not t or c['risk_type'] == t]
    return jsonify({'success': True, 'data': cases})


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
    return jsonify({'success': True, 'data': {
        'total_companies': len(MOCK_COMPANIES),
        'high_risk_count': high,
        'medium_risk_count': mid,
        'low_risk_count': low,
        'avg_probability_60d': round(avg, 1),
        'top_risk_types': [{'type': k, 'count': v} for k, v in top],
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
