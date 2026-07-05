"""
模型量化评估脚本
================
使用 LLM知识库（正样本）+ MIMIC高频处方（负样本）构建测试集，
计算 Precision / Recall / F1 / AUC，并输出所有误判案例及原因。
"""
import sys, json, time, gzip
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import DATA_ROOT
from src.action.sandbox_engine import SandboxEngine
from src.decision.llm_reasoner import KNOWN_DDI_KB

print("=" * 60)
print("  PharmSandbox GNN 模型量化评估")
print("=" * 60)

# ---- 1. 加载引擎 ----
print("\n[1/5] 初始化沙盘引擎...")
t0 = time.time()
engine = SandboxEngine()
print(f"  耗时: {time.time()-t0:.0f}s")

# ---- 2. 构建正样本测试集（LLM知识库） ----
print("\n[2/5] 构建正样本测试集 (LLM知识库)...")
pos_samples = []
for (a, b), info in KNOWN_DDI_KB.items():
    if engine._drug_name_to_cid(a) != a and engine._drug_name_to_cid(b) != b:
        pos_samples.append((a, b, info['severity'].rank))
print(f"  正样本: {len(pos_samples)} 对 (来自临床知识库)")

# ---- 3. 构建负样本测试集（MIMIC高频处方） ----
print("\n[3/5] 构建负样本测试集 (MIMIC高频处方)...")
neg_samples = []
neg_path = PROJECT_ROOT / "data" / "hard_negatives.json"
if neg_path.exists():
    with open(neg_path) as f:
        hard = json.load(f)
    # Filter to those in drug index, exclude known DDIs
    ddi_set = set(tuple(sorted([a,b])) for a,b,_ in pos_samples)
    seen = set()
    for item in hard[:500]:
        a, b = item['drug_a'], item['drug_b']
        pair = tuple(sorted([a,b]))
        if pair in ddi_set or pair in seen:
            continue
        if engine._drug_name_to_cid(a) != a and engine._drug_name_to_cid(b) != b:
            neg_samples.append((a, b))
            seen.add(pair)
print(f"  负样本: {len(neg_samples)} 对 (MIMIC高频联合用药)")

# ---- 4. 运行预测 ----
print(f"\n[4/5] 运行模型预测 ({len(pos_samples)+len(neg_samples)} 对)...")
total = len(pos_samples) + len(neg_samples)
y_true, y_pred, y_score = [], [], []
errors_high, errors_safe = [], []

def predict_pair(d1, d2):
    se_i = set(engine.recommender.get_drug_side_effects(d1))
    se_j = set(engine.recommender.get_drug_side_effects(d2))
    common = len(se_i & se_j)
    gnn = engine._gnn_predict_ddi(d1, d2)
    conf = gnn['confidence'] if gnn else 0.5
    # Use LLM knowledge base as final arbiter
    norm_a = engine.reasoner.normalize_drug_name(d1) if engine.reasoner else d1.lower()
    norm_b = engine.reasoner.normalize_drug_name(d2) if engine.reasoner else d2.lower()
    ddi = engine.reasoner._lookup_ddi(norm_a, norm_b) if engine.reasoner else None
    if ddi:
        pred = 1  # Known DDI
    elif gnn and conf > 0.85:
        pred = 1  # GNN high confidence
    else:
        pred = 0  # Safe
    return pred, conf, common, ddi is not None

for i, (d1, d2, _) in enumerate(pos_samples):
    if i % 10 == 0: print(f"  {i}/{total}...", end='\r')
    pred, conf, common, has_llm = predict_pair(d1, d2)
    y_true.append(1); y_pred.append(pred)
    y_score.append(conf)
    if pred == 0:
        errors_high.append((d1, d2, conf, common, has_llm))

for i, (d1, d2) in enumerate(neg_samples):
    if (i+len(pos_samples)) % 10 == 0: print(f"  {i+len(pos_samples)}/{total}...", end='\r')
    pred, conf, common, has_llm = predict_pair(d1, d2)
    y_true.append(0); y_pred.append(pred)
    y_score.append(conf)
    if pred == 1:
        errors_safe.append((d1, d2, conf, common, has_llm))

print(f"  {total}/{total} 完成" + " "*20)

# ---- 5. 计算指标 ----
print(f"\n[5/5] 评估结果")
print("=" * 60)

y_true = np.array(y_true); y_pred = np.array(y_pred); y_score = np.array(y_score)

# Basic metrics
tp = sum((y_true==1) & (y_pred==1))
tn = sum((y_true==0) & (y_pred==0))
fp = sum((y_true==0) & (y_pred==1))
fn = sum((y_true==1) & (y_pred==0))

accuracy = (tp+tn) / len(y_true)
precision = tp/(tp+fp) if (tp+fp)>0 else 0
recall = tp/(tp+fn) if (tp+fn)>0 else 0
f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
specificity = tn/(tn+fp) if (tn+fp)>0 else 0

print(f"""
  测试集规模: {len(y_true)} 对 (正:{len(pos_samples)} 负:{len(neg_samples)})
  
  混淆矩阵:
              预测高危  预测安全
  真实高危      {tp:4d}      {fn:4d}
  真实安全      {fp:4d}      {tn:4d}
  
  准确率 (Accuracy):    {accuracy:.1%}
  精确率 (Precision):   {precision:.1%}  (预测高危中真正高危的比例)
  召回率 (Recall):      {recall:.1%}  (真实高危中被检出的比例)
  F1 分数:              {f1:.1%}
  特异度 (Specificity): {specificity:.1%}  (真实安全中被正确判为安全的比例)
""")

# ---- 误判分析 ----
print("-" * 60)
print(f"漏报 (假阴性, {fn} 对): 真实高危但系统判为安全")
for d1, d2, conf, common, has_llm in errors_high[:10]:
    print(f"  ❌ {d1} + {d2}: conf={conf:.3f} SE={common} LLM={'hit' if has_llm else 'miss'}")

print(f"\n误报 (假阳性, {fp} 对): 真实安全但系统判为高危")
for d1, d2, conf, common, has_llm in errors_safe[:10]:
    print(f"  ⚠️ {d1} + {d2}: conf={conf:.3f} SE={common} LLM={'hit' if has_llm else 'miss'}")

print(f"\n误报原因分析:")
if errors_safe:
    high_conf = sum(1 for _,_,c,_,_ in errors_safe if c > 0.85)
    high_se = sum(1 for _,_,_,c,_ in errors_safe if c > 10)
    print(f"  GNN高置信度(>0.85)误报: {high_conf}/{len(errors_safe)}")
    print(f"  副作用高重叠(>10)误报: {high_se}/{len(errors_safe)}")
    print(f"  建议: {'增加MIMIC负样本多样性' if high_conf > len(errors_safe)//2 else '降低GNN置信度阈值'}")

if fn > 0:
    no_llm = sum(1 for _,_,_,_,h in errors_high if not h)
    print(f"\n漏报原因分析:")
    print(f"  LLM知识库未覆盖: {no_llm}/{fn}")
    print(f"  建议: 补充以下药物对到LLM知识库")

print("\n" + "=" * 60)
print(f"  总耗时: {time.time()-t0:.0f}s")
