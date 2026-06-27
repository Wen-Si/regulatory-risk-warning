# 智鉴风控 v3.0 - AI Native 监管问询预警系统

[![AI Native](https://img.shields.io/badge/AI--Native-v3.0-cyan)](https://github.com/)
[![Harness Engineering](https://img.shields.io/badge/Safety-Harness%20Engineering-red)](https://github.com/)
[![Knowledge Graph](https://img.shields.io/badge/Knowledge-Graph%20RAG-purple)](https://github.com/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## 产品简介

智鉴风控是基于 **Agentic AI + Harness Engineering + 知识图谱 + 深度学习 + 强化学习** 的上市公司监管问询预警系统，面向A股市场预测上市公司未来30/60/90天被监管问询的概率。

## 核心技术架构 v3.0

```
┌─────────────────────────────────────────────┐
│        Harness Engineering 安全合规层         │
│   输入护栏 │ 输出护栏 │ 审计日志 │ 工具防火墙    │
├─────────────────────────────────────────────┤
│        知识图谱层 (Graph RAG)                │
│   监管KG │ 风险推理 │ 证据链 │ 幻觉抑制         │
├─────────────────────────────────────────────┤
│        深度学习层                             │
│ DeepFM │ Temporal Transformer │ GAT │ TextEnc │
├─────────────────────────────────────────────┤
│        强化学习层                             │
│ PPO阈值优化 │ Thompson Sampling │ 在线学习     │
├─────────────────────────────────────────────┤
│        规则引擎层                             │
│ 触发事件检测 │ 财务信号 │ 合规信号 │ 硬约束     │
└─────────────────────────────────────────────┘
```

### 核心特性

- **AI Native**: 从底层架构到决策逻辑完全由AI驱动，非AI辅助工具
- **Harness Engineering**: 输入护栏（中英文注入检测、PII脱敏）、输出护栏（合规校验）、审计日志、熔断器
- **知识图谱**: 24+实体、36+关系的监管金融KG，BFS多跳推理，Graph RAG证据链
- **深度学习**: DeepFM + Temporal Transformer + GAT图注意力 + RiskTextEncoder四路并行
- **强化学习**: PPO自适应阈值优化，Thompson Sampling动态集成，Online Learner在线学习
- **安全合规**: 10步AI决策流水线，全链路安全检查和审计记录

## 快速开始

### 环境要求
- Python 3.8+
- pip

### 安装与运行

```bash
# 克隆仓库
git clone <repo-url>
cd regulatory-risk-warning

# 安装依赖
pip install -r requirements.txt

# （可选）配置智谱AI API Key（用于LLM可解释性报告和AI助手）
export ZHIPU_API_KEY="your-api-key"

# 启动服务
python app.py

# 访问 http://localhost:5000
```

### 生产部署

```bash
# 使用gunicorn
gunicorn app:app --bind 0.0.0.0:5000 --workers 4

# 或部署到Render/Railway（已包含render.yaml配置）
```

## 项目结构

```
regulatory-risk-warning/
├── app.py                    # Flask主应用（含安全头/速率限制）
├── requirements.txt          # Python依赖
├── render.yaml               # Render部署配置
├── static/
│   └── index.html            # 前端SPA（含XSS防护）
├── ml_engine/
│   ├── predictor.py          # 混合预测引擎v3.0（10步流水线）
│   ├── features.py           # 特征工程
│   ├── safety/               # Harness Engineering安全层
│   │   ├── guardrails.py     # 安全缰绳主控
│   │   ├── input_guard.py    # 输入护栏
│   │   ├── output_guard.py   # 输出护栏
│   │   └── audit_logger.py   # 审计日志
│   ├── knowledge_graph/      # 知识图谱模块
│   │   ├── knowledge_graph.py # 监管KG主类
│   │   ├── entities.py       # 实体定义
│   │   ├── relations.py      # 关系定义
│   │   └── graph_rag.py      # Graph RAG
│   ├── models/               # 深度学习模型
│   └── rl/                   # 强化学习组件
├── rule_engine/              # 规则引擎
└── logs/                     # 审计日志（.gitignore排除）
```

## API接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端页面 |
| GET | `/api/health` | 健康检查 |
| POST | `/api/ml/predict` | ML预测（含安全护栏+KG） |
| POST | `/api/analyze` | 单公司深度分析 |
| POST | `/api/analyze/batch` | 批量分析 |
| GET | `/api/risk/hot` | 高风险榜单 |
| POST | `/api/chat` | AI风控助手（含安全护栏） |
| GET | `/api/safety/report` | 安全层统计 |
| GET | `/api/kg/stats` | 知识图谱统计 |

## 安全特性

v3.0版本全面强化了安全合规能力：

- 输入安全：中英文提示注入检测（3层防御）、PII自动脱敏、IP速率限制
- 输出安全：置信度门控、禁止表述过滤、强制风险免责声明
- 审计日志：全链路不可篡改记录，支持按时间/公司/风险等级检索
- 访问控制：安全响应头（CSP/HSTS/X-Frame-Options等）、CORS白名单
- 前端安全：XSS防护（escapeHtml/sanitizeHtml）、请求大小限制

## 技术栈

- **后端**: Flask + NumPy
- **前端**: Tailwind CSS + Chart.js
- **深度学习**: NumPy实现（DeepFM/Transformer/GAT）
- **强化学习**: PPO + Thompson Sampling
- **知识图谱**: 属性图 + BFS多跳推理 + Graph RAG
- **安全层**: Harness Engineering四层架构
- **LLM**: 智谱GLM-4-Flash（可选）

## 商业计划书

完整商业计划书见 `智鉴风控v3_商业计划书.pptx`。

## License

MIT License
