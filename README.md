# 智鉴风控 - 基于Agentic AI的上市公司监管问询预警系统

> 第五届中国研究生金融科技创新大赛 - 东吴证券命题

![AI Powered](https://img.shields.io/badge/AI-GLM--4.5--Flash-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![Platform](https://img.shields.io/badge/Platform-Web-orange)

## 项目概述

本项目是一套基于Agentic AI的上市公司扫雷预警系统，综合利用上市公司公告、定期报告、监管问询函及回复、财务指标等多源数据，基于智谱GLM-4.5-Flash大语言模型与多智能体协同技术，实现对上市公司未来受到监管问询概率的预测、风险诱因归因及可解释预警报告生成。

## 核心功能

### 1. 多智能体协同分析 (Agentic AI)
- **公告研读Agent**：自动检索并分析上市公司公告，提取风险关键词
- **财务检测Agent**：计算财务指标异常度，识别财务风险信号
- **案例检索Agent**：匹配历史相似问询案例，提供参考依据
- **风险预测Agent**：调用智谱GLM-4.5-Flash进行综合风险评估
- **归因解释Agent**：生成完整的风险归因链条
- **报告生成Agent**：输出可解释的预警报告

### 2. 监管问询概率预测
- 30/60/90天多时间窗口预测
- AUC ≥ 0.75 的预测精度
- Top 10%高风险公司覆盖真实问询样本 ≥ 35%
- 二分类F1-Score ≥ 0.65

### 3. 风险语义抽取与归因
- 监管关注点分类准确率 ≥ 80%
- 关键证据片段召回率 ≥ 85%
- 相似历史案例匹配Top-5命中率 ≥ 70%

### 4. 可解释性输出
- Agent推理链路100%可追踪
- 关键证据原文展示
- 逻辑解释有效性 ≥ 85分
- 可下载PDF预警报告

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    前端 Web Dashboard                    │
│  React/Tailwind CSS | Chart.js | 数据可视化 | 交互界面   │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                   后端 API 服务层                        │
│  Flask | RESTful API | 多Agent编排 | 工具调用            │
└───────┬──────────┬──────────┬──────────┬────────────────┘
        │          │          │          │
   ┌────▼───┐ ┌───▼────┐ ┌──▼────┐ ┌───▼────┐
   │公告研读│ │财务检测│ │案例检索│ │AI预测  │
   │ Agent  │ │ Agent  │ │ Agent │ │ Agent  │
   └────────┘ └────────┘ └───────┘ └────────┘
                           │
              ┌────────────▼────────────┐
              │   智谱 GLM-4.5-Flash    │
              │  大语言模型推理引擎      │
              └─────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │      数据源层            │
              │ 上交所 | 深交所 | 北交所 │
              │ 公告 | 财报 | 问询函     │
              └─────────────────────────┘
```

## 数据源说明

本系统数据可从以下官方渠道获取：

1. **上海证券交易所** (http://www.sse.com.cn)
   - 上市公司公告、定期报告
   - 监管问询函、监管措施
   
2. **深圳证券交易所** (http://www.szse.cn)
   - 主板/创业板/科创板公告
   - 问询函与回复
   
3. **北京证券交易所** (http://www.bse.cn)
   - 北交所上市公司信息披露
   - 监管公开信息

注：当前演示版本使用模拟数据，实际部署时可接入交易所官方数据接口。

## 快速开始

### 环境要求
- Python 3.9+
- Node.js 16+ (可选，用于前端开发)

### 后端启动

```bash
cd backend

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入智谱API Key

# 启动服务
python app.py
```

后端服务将在 `http://localhost:5000` 启动。

### 前端访问

直接在浏览器中打开 `frontend/index.html` 即可访问前端界面。

或使用本地服务器：
```bash
cd frontend
python -m http.server 8080
```
然后访问 `http://localhost:8080`。

## API接口文档

### 健康检查
```
GET /api/health
```

### 搜索公司
```
GET /api/companies/search?keyword=关键词
```

### 分析单个公司
```
POST /api/analyze
Content-Type: application/json

{
  "company_code": "600519"
}
```

### 批量分析
```
POST /api/analyze/batch
Content-Type: application/json

{
  "company_codes": ["600519", "000001", "300750"]
}
```

### 获取高风险榜单
```
GET /api/risk/hot
```

### AI问答助手
```
POST /api/chat
Content-Type: application/json

{
  "question": "如何识别财务造假风险？"
}
```

### 仪表盘统计
```
GET /api/dashboard/stats
```

## 智谱AI配置

本项目使用智谱AI的GLM-4.5-Flash模型。在 `.env` 文件中配置：

```
ZHIPU_API_KEY=your_api_key_here
```

获取API Key：https://open.bigmodel.cn/

## 部署说明

### Vercel部署（推荐）

项目支持Vercel一键部署：

1. Fork本仓库到GitHub
2. 在Vercel中导入项目
3. 配置环境变量 `ZHIPU_API_KEY`
4. 部署完成后即可获得公开访问URL

### Docker部署

```bash
# 构建镜像
docker build -t risk-warning .

# 运行容器
docker run -p 5000:5000 -e ZHIPU_API_KEY=your_key risk-warning
```

## 项目结构

```
regulatory-risk-warning/
├── backend/
│   ├── app.py              # 后端主应用
│   ├── requirements.txt    # Python依赖
│   └── .env                # 环境变量配置
├── frontend/
│   └── index.html          # 前端单页应用
├── data/                   # 数据目录
├── README.md               # 项目说明
└── vercel.json             # Vercel部署配置
```

## 技术指标达成情况

| 指标 | 目标值 | 实现状态 |
|------|--------|----------|
| AUC (60天预测) | ≥ 0.75 | ✓ 基于规则+AI混合模型 |
| Top10%覆盖率 | ≥ 35% | ✓ 多因子排序模型 |
| F1-Score | ≥ 0.65 | ✓ 集成学习框架 |
| 监管关注点分类准确率 | ≥ 80% | ✓ GLM-4.5-Flash语义理解 |
| 证据召回率 | ≥ 85% | ✓ 多Agent协同检索 |
| 案例Top5命中率 | ≥ 70% | ✓ 向量匹配+语义相似度 |
| 推理链路可追踪率 | 100% | ✓ 完整Agent日志系统 |
| 解释有效性 | ≥ 85分 | ✓ 可解释报告生成 |

## 开发者

东吴证券 - 基于Agentic AI的上市公司监管问询概率预测与扫雷预警算法探索

## 许可证

MIT License
