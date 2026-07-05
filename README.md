# 💊 药盘推演（PharmSandbox）智能体

基于图神经网络（GNN）的药物相互作用沙盘推演系统，为重庆大学医学智能体课程设计项目。

> **v3.0** — 全数据训练（TWOSIDES + Decagon 27万DDI对）+ GNN模型大幅升级（AUC 0.918→0.992）+ 三端UI + 全模块测试

## 核心功能

| 功能 | 说明 | 实现模块 |
|------|------|----------|
| **功能①** GNN DDI沙盘推演 | RGCN图神经网络预测药物相互作用（AUC=0.992），无GNN时回退副作用比较 | `src/decision/gnn/` |
| **功能②** 0-100风险量化评分 | SIDER副作用概率 + MIMIC化验值 + 患者疾病史动态加权 | `src/decision/risk_scorer/` |
| **功能③** 替代药物推荐 | ATC分类树 + 适应症匹配 + 副作用对比推荐同类替代药 | `src/decision/recommender/` |
| **功能④** 慢病续方触发器 | 自动计算剩余药量，断药前触发续方提醒 + 草稿生成 | `src/decision/prescription_trigger.py` |
| **功能⑤** LLM推理与对话 | 14种DDI机制模板 + 多轮用药咨询 + 意图识别 | `src/decision/llm_reasoner.py` |

## GNN 模型性能

基于 TWOSIDES + Decagon 全量数据集训练的 RGCN 药物相互作用预测模型：

| 指标 | v2.5 (旧) | v3.0 (新) | 提升 |
|------|-----------|-----------|------|
| **测试 AUC** | 0.9183 | **0.9921** | +8.0% |
| **测试 F1** | 0.8362 | **0.9531** | +14.0% |
| **Accuracy** | 0.8501 | **0.9532** | +12.1% |
| **Precision** | 0.8275 | **0.9556** | +15.5% |
| **Recall** | 0.8515 | **0.9532** | +11.9% |

### 训练数据规模

| 维度 | v2.5 (旧) | v3.0 (新) |
|------|-----------|-----------|
| DDI正样本 | 22,534 | **274,585** (12.2×) |
| 数据来源 | TWOSIDES部分 | TWOSIDES全量 + Decagon |
| 药物数 | 2,095 | **3,730** |
| 知识图谱 | 6,775节点, 351,410边 | **8,495节点, 352,349边** |
| 模型参数量 | 1,147,266 | **1,367,426** |

训练脚本：`scripts/train_gnn.py`，模型保存：`models/best_model.pt`

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      PharmSandbox v3.0                           │
├─────────────────────────────────────────────────────────────────┤
│ 感知层 (Perception)                                              │
│   └─ NER药物识别: 100+中文药名 + 商品名 + 剂量提取 + 疾病提取    │
├─────────────────────────────────────────────────────────────────┤
│ 决策层 (Decision)                                                │
│   ├─ SandboxEngine (统一入口，协调所有子模块)                     │
│   ├─ GNN DDI推演: RGCN + 知识图谱 (AUC=0.992, 自动检测)         │
│   ├─ 严重度融合: GNN + LLM知识库 + 副作用比较，三者取最高        │
│   ├─ LLM推理器: 14种DDI机制模板 + 多轮对话咨询                   │
│   ├─ 风险评分: SIDER概率 + MIMIC化验值 + 患者约束动态加权         │
│   ├─ 替代推荐: ATC树搜索 + 适应症匹配 + 副作用对比               │
│   └─ 续方触发: 剩余药量计算 + 自动续方草稿                       │
├─────────────────────────────────────────────────────────────────┤
│ 数据层 (Data)                                                    │
│   ├─ DataLoader: 统一加载 + 缓存 (副作用/适应症/靶点/处方)       │
│   ├─ LabEventsLoader: 化验数据 (支持预建索引快速查询)            │
│   └─ 副作用数据统一加载，注入RiskScorer避免重复                  │
├─────────────────────────────────────────────────────────────────┤
│ 行动层 (Action)                                                  │
│   ├─ Flask RESTful API: 9个核心接口                              │
│   ├─ 医生工作台: 患者队列 + 化验监控 + 一键推演                  │
│   ├─ 沙盘推演: 风险仪表盘 + 知识图谱 + 替代推荐                  │
│   └─ 患者端: 用药信息 + 副作用反馈                               │
└─────────────────────────────────────────────────────────────────┘
```

## 三端 UI

| 页面 | 路由 | 功能 |
|------|------|------|
| 🏠 沙盘推演 | `/sandbox` | 药物输入 + 风险仪表盘 + 知识图谱 + 替代推荐 |
| 👨‍⚕️ 医生工作台 | `/doctor` | 患者队列 + 化验监控 + 一键进入推演 |
| 📱 患者端 | `/patient` | 用药信息 + 副作用上报 |

三端通过 `localStorage` 实现跨窗口数据通信。

## 目录结构

```
PharmSandbox/
├── src/                              # 核心源代码 (~8,800 行)
│   ├── config.py                     # 全局配置 (路径/日志/参数)
│   ├── data/
│   │   ├── loader.py                 # 统一数据加载器 (缓存+懒加载)
│   │   └── lab_loader.py             # 化验数据加载器 (支持预建索引)
│   ├── perception/
│   │   └── ner/
│   │       └── drug_ner.py           # 医学NER (中文药名/剂量/疾病)
│   ├── decision/
│   │   ├── sandbox_engine.py         # 综合沙盘推演引擎 (统一入口)
│   │   ├── gnn/
│   │   │   ├── knowledge_graph.py    # 药物知识图谱构建
│   │   │   ├── model.py              # RGCN图神经网络模型
│   │   │   └── trainer.py            # GNN训练器
│   │   ├── llm_reasoner.py           # LLM推理模块 (DDI解释+对话)
│   │   ├── risk_scorer/
│   │   │   └── scorer.py             # 0-100风险量化评分
│   │   ├── recommender/
│   │   │   └── recommender.py        # 替代药物推荐
│   │   └── prescription_trigger.py   # 慢病续方触发器
│   └── action/
│       ├── sandbox_engine.py         # 综合沙盘推演引擎
│       └── api/
│           └── app.py                # Flask RESTful API
│
├── frontend/
│   ├── dashboard.py                  # Streamlit高级仪表板
│   └── templates/
│       ├── index.html                # 沙盘推演主页面
│       ├── doctor.html               # 医生工作台
│       └── patient_h5.html           # 患者端H5页面
│
├── scripts/
│   ├── train_gnn.py                  # GNN模型训练脚本 (v2全数据版)
│   └── build_lab_index.py            # 化验数据预建索引脚本
│
├── models/
│   ├── best_model.pt                 # 训练好的GNN模型 (AUC=0.992)
│   └── training_history.json         # 训练历史
│
├── data/                             # 数据集
│   ├── sider/                        # SIDER副作用数据库
│   ├── drugcentral/                  # DrugCentral药物靶点
│   ├── nsides/                       # TWOSIDES/OFFSIDES/Decagon
│   └── mimic/                        # MIMIC-IV临床数据
│
├── tests/
│   └── test_all.py                   # 全模块单元测试 (11项全部通过)
├── docs/
│   ├── README_项目索引.md
│   ├── 参考项目分析.md
│   └── 课设选题材料.md
│
├── run.py                            # 启动脚本
├── setup.py                          # 一键安装
├── requirements.txt                  # 依赖列表
└── README.md
```

## 快速开始

### 1. 环境配置
```bash
git clone https://github.com/050415/PharmSandbox.git
cd PharmSandbox
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # 或 python setup.py
```

### 2. (可选) 预建化验索引
```bash
# 将 2.5GB labevents 按患者分区，加速查询 (运行一次)
python scripts/build_lab_index.py
```

### 3. (可选) 训练GNN模型
```bash
# 使用TWOSIDES+Decagon全量数据训练RGCN模型 (约45分钟, CPU)
python scripts/train_gnn.py
```

### 4. 启动服务
```bash
python run.py                     # Flask API (端口5000)
# 访问: http://localhost:5000/sandbox  (沙盘推演)
#       http://localhost:5000/doctor   (医生工作台)
#       http://localhost:5000/patient  (患者端)
```

### 5. API 调用示例
```python
import requests

# 综合沙盘推演
resp = requests.post('http://localhost:5000/api/sandbox/simulate', json={
    'drugs': ['aspirin', 'warfarin', 'metformin'],
    'patient': {'age': 70, 'gender': 'M', 'conditions': ['diabetes', 'CKD']}
})
result = resp.json()
print(f"DDI冲突: {result['analysis']['ddi']['count']}")
print(f"风险评分: {result['analysis']['risk']['final_score']}")
print(f"GNN可用: {result['analysis']['ddi'].get('gnn_available')}")
```

## API 接口

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/health` | GET | 健康检查 + 数据统计 |
| `/api/parse` | POST | 自然语言解析 (中文药名/疾病/剂量) |
| `/api/ddi/check` | POST | DDI冲突检测 |
| `/api/risk/score` | POST | 0-100风险评分 |
| `/api/recommend/alternative` | POST | 替代药推荐 |
| `/api/recommend/explain` | POST | 替代推荐因果解释 |
| `/api/prescription/check` | POST | 续方检查 |
| `/api/prescription/refill-draft` | POST | 续方草稿生成 |
| `/api/sandbox/simulate` | POST | **综合沙盘推演** (核心接口) |
| `/api/ehr/patients` | GET | 患者列表 (医生工作台) |
| `/api/ehr/lab_refresh` | POST | 模拟化验刷新 (IoT) |

## 技术栈

| 类别 | 技术 |
|------|------|
| 后端框架 | Flask + Flask-CORS |
| 图神经网络 | PyTorch + PyTorch Geometric (RGCN) |
| 数据处理 | Pandas + NumPy + NetworkX |
| 前端UI | HTML/CSS/JS + Chart.js + vis-network |
| 数据可视化 | Streamlit + Plotly + PyVis |
| 数据集 | SIDER + DrugCentral + TWOSIDES + Decagon + MIMIC-IV |

## 数据集

| 数据 | 来源 | 规模 |
|------|------|------|
| 药物名称 | SIDER | 1,430 种药物 |
| 副作用 | SIDER | 309,849 条记录 |
| 适应症 | SIDER | 30,835 条记录 |
| ATC分类 | SIDER | 1,560 条记录 |
| 药物靶点 | DrugCentral | 药物-靶点相互作用 |
| 分子结构 | DrugCentral | SMILES格式 |
| DDI数据 | TWOSIDES | 42,920,391 条记录 → 211,112 唯一药物对 |
| DDI数据 | Decagon | 4,649,441 条记录 → 63,473 唯一药物对 |
| 患者病历 | MIMIC-IV | 364,627 名患者 |
| 处方数据 | MIMIC-IV | 20,292,611 条处方 |
| 化验数据 | MIMIC-IV | 158,374,765 条化验记录 |

## 测试

```bash
# 运行全模块单元测试 (11项)
python -m tests.test_all
```

测试覆盖模块：
- ✅ 配置模块
- ✅ 数据加载器
- ✅ 药物NER
- ✅ 替代药推荐
- ✅ 风险评分
- ✅ 续方触发器
- ✅ LLM推理器
- ✅ 沙盘引擎
- ✅ GNN模型
- ✅ 知识图谱
- ✅ Flask API

## 参考项目

| 项目 | 星数 | 用途 |
|------|------|------|
| [SumGNN](https://github.com/yueyu1030/SumGNN) | ⭐91 | RGCN DDI预测架构 |
| [EmerGNN](https://github.com/LARS-research/EmerGNN) | ⭐30 | 新兴DDI预测 |
| [TIP](https://github.com/NYXFLOWER/TIP) | ⭐25 | 三图交互传播 |
| [DDI-AltRec](https://github.com/abhayjit07/Drug-Drug-Interaction-and-Alternate-Recommendation-System) | - | 替代药推荐 |
| [RDUKG](https://github.com/zhenjia2017/RDUKG) | - | 医学知识图谱 |

## 团队成员

- 吴汉鹏 - 后端开发、数据处理、GNN模型训练、系统集成
- 队友 - 前端UI设计与开发

## 许可证

本项目仅供学术研究使用。
