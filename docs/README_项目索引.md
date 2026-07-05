# 药盘推演（PharmSandbox）数据与开源参考项目索引

## 📁 数据文件清单

### SIDER 数据库（副作用 + 适应症 + 分类）
| 文件 | 用途 | 对应功能 |
|------|------|----------|
| `drug_names.tsv` | 药物名称→CID标准化映射 | 功能① 感知层输入归一化 |
| `drug_atc.tsv` | ATC分类代码树 | 功能③ 替代药搜索导航树 |
| `meddra_all_se.tsv.gz` | 单药副作用表 | 功能② 临床副作用概率底表 |
| `meddra_all_indications.tsv.gz` | 药物适应症表 | 功能③ 搜索导航树 |
| `meddra_freq.tsv.gz` | 副作用发生频率 | 功能② 风险量化评分 |
| `meddra.tsv.gz` | MedDRA医学字典 | 全局术语标准化 |
| `meddra_all_label_indications.tsv.gz` | 标签适应症 | 功能③ 补充适应症数据 |
| `meddra_all_label_se.tsv.gz` | 标签副作用 | 功能② 补充副作用数据 |

### NSIDES/TwoSIDES 数据库（DDI + 联合用药副作用）
| 文件 | 用途 | 对应功能 |
|------|------|----------|
| `TWOSIDES.csv.gz` | 药物-药物联合副作用（63,000+组合） | 功能① GNN训练正负样本标签 |
| `OFFSIDES.csv.gz` | 标签外副作用信号 | 功能② 补充副作用概率 |
| `NSIDES_README.txt` | 数据字典说明 | 参考文档 |

### DrugCentral 平替（替代DrugBank）
| 文件 | 用途 | 对应功能 |
|------|------|----------|
| `drugcentral_drug_target_interactions.tsv.gz` | 药物-靶点相互作用 | 功能① 特征矩阵输入 |
| `drugcentral_structures_smiles.tsv` | SMILES分子结构 | 功能① GNN分子特征 |
| `drugcentral_structures_3d.sdf.gz` | 3D分子构象 | 功能① 分子图构建 |
| `drugcentral_fda_ema_pmda_approved.csv` | FDA/EMA/PMDA批准药物清单 | 全局药物筛选 |

### Synthea（合成患者数据生成器）
| 文件 | 用途 | 对应功能 |
|------|------|----------|
| `synthea-master.zip` | 生成合成患者EHR数据 | 功能② 动态变权因子（替代MIMIC-IV测试用） |

> ⚠️ MIMIC-IV 真实病历数据正在购买中，Synthea 仅用于开发测试阶段。

---

## 🔬 开源参考项目（按课设功能映射）

### 功能①：GNN隐性DDI沙盘推演

| 项目 | 星数 | 来源 | 核心价值 |
|------|------|------|----------|
| **SumGNN** `SumGNN.zip` | ⭐91 | Bioinformatics 2021 | 多类型DDI预测，使用子图聚合+知识图谱，直接对标你的GNN推演需求 |
| **EmerGNN** `EmerGNN.zip` | ⭐30 | LARS Research | 新兴DDI预测，基于流式GNN，适合预测未知/罕见药物冲突 |
| **TIP** `TIP.zip` | ⭐25 | NeurIPS 2019 | 三图交互传播模型，药物-蛋白质-副作用异构图，多药副作用预测 |
| **GraphDDI** `GraphDDI.zip` | ⭐— | AIiH 2024 | 从SMILES直接构建分子图→GNN预测DDI，端到端流程 |
| **CADGL** `CADGL.zip` | — | CIKM | 上下文感知深度图学习DDI预测 |
| **Polypharmacy-GCN** `Polypharmacy-GCN.zip` | ⭐9 | GCN高效多药副作用预测 | 与TWOSIDES数据直接配套 |

### 功能②：患者特异性风险量化评分

| 项目 | 星数 | 来源 | 核心价值 |
|------|------|------|----------|
| **TIP** `TIP.zip` | ⭐25 | NeurIPS 2019 | 多药副作用概率预测，可作为风险评分的底层模型 |
| **Polypharmacy-GCN** | ⭐9 | GCN | 高效预测多药副作用发生概率 |

### 功能③：替代药物推荐

| 项目 | 星数 | 来源 | 核心价值 |
|------|------|------|----------|
| **DDI-AltRec** `DDI-AltRec.zip` | — | DDI+替代药推荐 | **直接对标**你的功能③：检测DDI后推荐安全替代药 |
| **RDUKG** `RDUKG.zip` | — | MediKS@CIKM 2025 | 医学知识图谱促进合理用药，ATC层级搜索+指南约束推荐 |

### 功能④：慢病续方触发器

| 项目 | 星数 | 来源 | 核心价值 |
|------|------|------|----------|
| **Synthea** `synthea-master.zip` | — | 合成患者数据 | 生成慢病患者用药周期数据，测试续方触发逻辑 |

### 感知层：医学NER/药物名称识别

| 项目 | 星数 | 来源 | 核心价值 |
|------|------|------|----------|
| **MedKG-FDA** `MedKG-FDA.zip` | — | NLP Pipeline | 从FDA数据构建可查询的医学知识图谱 |

### 系统架构参考

| 项目 | 星数 | 来源 | 核心价值 |
|------|------|------|----------|
| **CDSS-FastAPI** `CDSS-FastAPI.zip` | — | FastAPI+React | 临床决策支持系统全栈架构参考 |
| **MedClaimsVerifier** `MedClaimsVerifier.zip` | — | 医学声明验证 | 防止LLM幻觉，验证药物/疾病声明是否符合知识库 |

---

## 🏗️ 推荐技术栈（基于参考项目总结）

```
感知层:  drug_names.tsv + spaCy/BioBERT NER → 药物ID归一化
决策层:  SumGNN/TIP GNN模型 + TWOSIDES训练 → DDI风险概率
         SIDER副作用底表 + MIMIC-IV约束 → 0-100风险评分
         ATC树 + 适应症表 → 同类药搜索 + 指南约束推荐
行动层:  FastAPI后端 + React前端 → 雷达风险图可视化
         HIS系统API → 续方草稿推送
```

---

## 📊 数据覆盖度评估

| 课设需求 | 数据来源 | 覆盖状态 |
|----------|----------|----------|
| 药物分子结构/靶点 | DrugCentral (SMILES + 靶点) | ✅ 已覆盖 |
| DDI训练标签 | TWOSIDES (63,000+药物组合) | ⏳ 下载中 |
| 单药副作用概率 | SIDER + OffSIDES | ✅+⏳ |
| 适应症/ATC分类 | SIDER (meddra_all_indications + drug_atc) | ✅ 已覆盖 |
| 患者EHR数据 | MIMIC-IV | 💰 购买中 |
| GNN模型代码 | SumGNN + EmerGNN + TIP + GraphDDI | ✅ 已获取 |
| 替代药推荐代码 | DDI-AltRec + RDUKG | ✅ 已获取 |
| 系统架构参考 | CDSS-FastAPI + MedClaimsVerifier | ✅ 已获取 |
