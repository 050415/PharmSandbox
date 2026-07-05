"""
GNN DDI 模型训练脚本 (v2 - 全数据版)
====================================
合并 TWOSIDES全量 + Decagon + hard_negatives 训练 RGCN 药物相互作用预测模型。

数据源:
  - TWOSIDES (42.9M行 → 211,292唯一对, 2,699药物)
  - ChChSe-Decagon (4.6M行 → 63,473唯一对, 645药物)
  - hard_negatives.json (3,000对, MIMIC+SIDER挖掘)
  - SIDER 药物/副作用/适应症 (知识图谱边)
  - DrugCentral 靶点 (知识图谱边)

用法:
    cd D:/PharmSandbox
    python scripts/train_gnn.py
"""
import sys
import os
import time
import json
import pickle
import gzip
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import RGCNConv

from src.config import DATA_ROOT, MODEL_ROOT


# ======================================================================
# 1. 数据准备
# ======================================================================

def load_twosides(data_root):
    """加载 TWOSIDES 全量药物-药物相互作用数据。"""
    path = data_root / "nsides" / "TWOSIDES.csv.gz"
    print(f"[数据] 加载 TWOSIDES 全量: {path}", flush=True)

    if path.stat().st_size < 1000:
        print(f"  TWOSIDES is LFS placeholder ({path.stat().st_size} bytes)", flush=True)
        return [], set()

    pairs = set()
    drugs = set()
    total_rows = 0

    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
        header = f.readline()
        cols = header.strip().split(',')
        d1_idx = cols.index('drug_1_concept_name')
        d2_idx = cols.index('drug_2_concept_name')

        for line in f:
            parts = line.strip().split(',')
            if len(parts) > max(d1_idx, d2_idx):
                d1 = parts[d1_idx].strip().strip('"').lower()
                d2 = parts[d2_idx].strip().strip('"').lower()
                if d1 and d2 and d1 != d2 and d1 != 'nan' and d2 != 'nan':
                    drugs.add(d1)
                    drugs.add(d2)
                    pairs.add(tuple(sorted([d1, d2])))
            total_rows += 1
            if total_rows % 5000000 == 0:
                print(f"  已读取 {total_rows:,} 行, {len(pairs):,} 对, {len(drugs):,} 药物", flush=True)

    pairs = list(pairs)
    print(f"  完成: {len(pairs):,} 个唯一药物对, {len(drugs):,} 种药物 (来自 {total_rows:,} 条记录)", flush=True)
    return pairs, drugs


def load_decagon(data_root):
    """加载 ChChSe-Decagon 多药联用副作用数据。"""
    path = data_root / "nsides" / "ChChSe-Decagon_polypharmacy.csv.gz"
    if not path.exists():
        print("[数据] Decagon 文件不存在，跳过", flush=True)
        return [], set()

    print(f"[数据] 加载 Decagon: {path}", flush=True)
    pairs = set()
    drugs = set()
    total_rows = 0

    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
        header = f.readline()  # STITCH 1,STITCH 2,Polypharmacy Side Effect,Side Effect Name
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                d1 = parts[0].strip().lower()
                d2 = parts[1].strip().lower()
                if d1 and d2 and d1 != d2:
                    drugs.add(d1)
                    drugs.add(d2)
                    pairs.add(tuple(sorted([d1, d2])))
            total_rows += 1
            if total_rows % 1000000 == 0:
                print(f"  已读取 {total_rows:,} 行, {len(pairs):,} 对, {len(drugs):,} 药物", flush=True)

    pairs = list(pairs)
    print(f"  完成: {len(pairs):,} 个唯一药物对, {len(drugs):,} 种药物 (来自 {total_rows:,} 条记录)", flush=True)
    return pairs, drugs


def load_sider_drugs(data_root):
    """加载 SIDER 药物名称列表。"""
    path = data_root / "sider" / "drug_names.tsv"
    df = pd.read_csv(path, sep='\t', header=None,
                     names=['cid', 'drug_name', 'se_id', 'umls_id'])
    drugs = set(df['drug_name'].str.strip().str.lower().unique())
    print(f"[数据] SIDER 药物: {len(drugs)} 种", flush=True)
    return drugs


def build_drug_index(all_drugs):
    """构建药物名→索引映射。"""
    drug_list = sorted(all_drugs)
    drug_to_idx = {name: i for i, name in enumerate(drug_list)}
    print(f"[数据] 药物索引: {len(drug_to_idx)} 种", flush=True)
    return drug_to_idx, drug_list


def build_knowledge_graph_edges(data_root, drug_to_idx):
    """构建知识图谱边列表（SIDER副作用 + DrugCentral靶点 + TWOSIDES交互）。"""
    edges = []
    edge_type_names = ['has_side_effect', 'has_indication', 'targets', 'interacts_with']

    # SIDER 副作用边
    se_path = data_root / "sider" / "meddra_all_se.tsv.gz"
    names_path = data_root / "sider" / "drug_names.tsv"

    print("[数据] 构建 SIDER 副作用边...", flush=True)
    name_df = pd.read_csv(names_path, sep='\t', header=None,
                          names=['cid', 'drug_name', 'se_id', 'umls_id'])
    cid_to_name = dict(zip(name_df['cid'], name_df['drug_name'].str.strip().str.lower()))

    se_nodes = {}
    se_count = 0

    with gzip.open(se_path, 'rt', encoding='utf-8') as f:
        se_df = pd.read_csv(f, sep='\t', header=None, nrows=500000,
                            names=['cid', 'umls_cui_from', 'method',
                                   'side_effect_name', 'umls_cui_to',
                                   'placebo', 'frequency', 'lower', 'upper'])

    for _, row in se_df.iterrows():
        drug_name = cid_to_name.get(row['cid'])
        se_name = str(row['side_effect_name']).strip()
        if drug_name and drug_name in drug_to_idx and se_name and se_name != 'nan':
            if se_name not in se_nodes:
                se_nodes[se_name] = len(drug_to_idx) + len(se_nodes)
            edges.append((drug_to_idx[drug_name], se_nodes[se_name], 0))
            se_count += 1

    print(f"  [OK] 副作用边: {se_count}, 副作用节点: {len(se_nodes)}", flush=True)

    # SIDER 适应症边
    ind_path = data_root / "sider" / "meddra_all_indications.tsv.gz"
    ind_nodes = {}
    ind_count = 0

    with gzip.open(ind_path, 'rt', encoding='utf-8') as f:
        ind_df = pd.read_csv(f, sep='\t', header=None, nrows=300000,
                             names=['cid', 'umls_cui_from', 'method',
                                    'indication_name', 'umls_cui_to',
                                    'mesh_id', 'max_phase', 'evidence_type'])

    for _, row in ind_df.iterrows():
        drug_name = cid_to_name.get(row['cid'])
        ind_name = str(row['indication_name']).strip()
        if drug_name and drug_name in drug_to_idx and ind_name and ind_name != 'nan':
            if ind_name not in ind_nodes:
                ind_nodes[ind_name] = len(drug_to_idx) + len(se_nodes) + len(ind_nodes)
            edges.append((drug_to_idx[drug_name], ind_nodes[ind_name], 1))
            ind_count += 1

    print(f"  [OK] 适应症边: {ind_count}, 适应症节点: {len(ind_nodes)}", flush=True)

    # DrugCentral 靶点边
    dc_path = data_root / "drugcentral" / "drugcentral_drug_target_interactions.tsv.gz"
    target_nodes = {}
    target_count = 0

    try:
        with gzip.open(dc_path, 'rt', encoding='utf-8') as f:
            dc_df = pd.read_csv(f, sep='\t', nrows=50000, low_memory=False)

        drug_col = None
        target_col = None
        for c in dc_df.columns:
            cl = c.lower()
            if 'drug' in cl and 'name' in cl:
                drug_col = c
            if 'target' in cl or 'gene' in cl or 'uniprot' in cl:
                target_col = c

        if drug_col and target_col:
            for _, row in dc_df.iterrows():
                drug_name = str(row[drug_col]).strip().lower()
                target_name = str(row[target_col]).strip()
                if drug_name in drug_to_idx and target_name and target_name != 'nan':
                    if target_name not in target_nodes:
                        target_nodes[target_name] = len(drug_to_idx) + len(se_nodes) + len(ind_nodes) + len(target_nodes)
                    edges.append((drug_to_idx[drug_name], target_nodes[target_name], 2))
                    target_count += 1
            print(f"  [OK] 靶点边: {target_count}, 靶点节点: {len(target_nodes)}", flush=True)
        else:
            print(f"  ⚠ DrugCentral 列名未识别: {list(dc_df.columns)}", flush=True)
    except Exception as e:
        print(f"  ⚠ DrugCentral 加载失败: {e}", flush=True)

    total_nodes = len(drug_to_idx) + len(se_nodes) + len(ind_nodes) + len(target_nodes)
    print(f"[数据] 知识图谱: {total_nodes} 节点, {len(edges)} 边, {len(edge_type_names)} 种边类型", flush=True)

    return edges, edge_type_names, total_nodes, len(drug_to_idx)


def load_hard_negatives(data_root):
    """加载预挖掘的硬负样本数据集。"""
    path = data_root.parent / "data" / "hard_negatives.json"
    if not path.exists():
        print("[数据] 未找到 hard_negatives.json，将使用随机负样本", flush=True)
        return [], set()
    with open(path, 'r') as f:
        raw = json.load(f)
    pairs = []
    seen = set()
    for item in raw:
        d1, d2 = item['drug_a'], item['drug_b']
        pair = tuple(sorted([d1, d2]))
        if pair not in seen:
            pairs.append(pair)
            seen.add(pair)
    print(f"[数据] 加载硬负样本: {len(pairs)} pairs (MIMIC + SIDER)", flush=True)
    return pairs, seen


def prepare_training_data(ddi_pairs, drug_to_idx, num_drugs, neg_ratio=1, hard_negatives=None):
    """
    准备训练数据：正样本=已知DDI对, 负样本=硬负样本+随机负样本。
    """
    print(f"[数据] 准备训练数据 (负采样比例={neg_ratio})...", flush=True)

    # 正样本
    pos_pairs = []
    for d1, d2 in ddi_pairs:
        if d1 in drug_to_idx and d2 in drug_to_idx:
            pos_pairs.append((drug_to_idx[d1], drug_to_idx[d2]))

    print(f"  正样本: {len(pos_pairs):,}", flush=True)

    # 建立已知DDI集合（排除用）
    ddi_set = set()
    for d1, d2 in ddi_pairs:
        if d1 in drug_to_idx and d2 in drug_to_idx:
            i, j = drug_to_idx[d1], drug_to_idx[d2]
            ddi_set.add((min(i, j), max(i, j)))

    # --- 负样本混合策略 ---
    neg_pairs = []

    # 优先使用硬负样本
    if hard_negatives:
        for d1_name, d2_name in hard_negatives:
            if d1_name in drug_to_idx and d2_name in drug_to_idx:
                i, j = drug_to_idx[d1_name], drug_to_idx[d2_name]
                pair = (min(i, j), max(i, j))
                if pair not in ddi_set:
                    neg_pairs.append(pair)
                    ddi_set.add(pair)
        print(f"  硬负样本: {len(neg_pairs):,}", flush=True)

    # 剩余配额用随机负样本补足
    target_neg = len(pos_pairs) * neg_ratio
    remaining = max(0, target_neg - len(neg_pairs))
    rng = np.random.default_rng(42)
    max_attempts = remaining * 10
    attempts = 0
    while len(neg_pairs) < target_neg and attempts < max_attempts:
        i, j = rng.integers(0, num_drugs, size=2)
        if i != j:
            pair = (min(i, j), max(i, j))
            if pair not in ddi_set:
                neg_pairs.append(pair)
                ddi_set.add(pair)
        attempts += 1

    print(f"  总负样本: {len(neg_pairs):,}", flush=True)

    # 合并并打乱
    all_pairs = pos_pairs + neg_pairs
    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

    indices = rng.permutation(len(all_pairs))
    all_pairs = [all_pairs[i] for i in indices]
    labels = [labels[i] for i in indices]

    # 划分 train/val/test (8:1:1)
    n = len(all_pairs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    data = {
        'train': {
            'drug1': torch.tensor([p[0] for p in all_pairs[:n_train]], dtype=torch.long),
            'drug2': torch.tensor([p[1] for p in all_pairs[:n_train]], dtype=torch.long),
            'labels': torch.tensor(labels[:n_train], dtype=torch.long),
        },
        'val': {
            'drug1': torch.tensor([p[0] for p in all_pairs[n_train:n_train+n_val]], dtype=torch.long),
            'drug2': torch.tensor([p[1] for p in all_pairs[n_train:n_train+n_val]], dtype=torch.long),
            'labels': torch.tensor(labels[n_train:n_train+n_val], dtype=torch.long),
        },
        'test': {
            'drug1': torch.tensor([p[0] for p in all_pairs[n_train+n_val:]], dtype=torch.long),
            'drug2': torch.tensor([p[1] for p in all_pairs[n_train+n_val:]], dtype=torch.long),
            'labels': torch.tensor(labels[n_train+n_val:], dtype=torch.long),
        },
    }

    print(f"  训练集: {n_train:,}, 验证集: {n_val:,}, 测试集: {n - n_train - n_val:,}", flush=True)
    return data


# ======================================================================
# 2. 模型定义
# ======================================================================

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced DDI classification."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class DDIPredictor(nn.Module):
    """RGCN 药物相互作用预测器。"""

    def __init__(self, num_nodes, num_relations, hidden_dim=128,
                 num_layers=3, num_classes=2, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim

        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.dropout = dropout
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, edge_index, edge_type):
        """编码整个知识图谱，返回所有节点嵌入。"""
        x = self.node_embedding.weight
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_type)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, edge_index, edge_type, drug1_idx, drug2_idx):
        """前向传播。"""
        x = self.encode(edge_index, edge_type)
        pair = torch.cat([x[drug1_idx], x[drug2_idx]], dim=-1)
        return self.classifier(pair)


# ======================================================================
# 3. 训练器
# ======================================================================

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cpu')
        self.model.to(self.device)

        self.optimizer = Adam(model.parameters(),
                              lr=config['lr'],
                              weight_decay=config['weight_decay'])
        self.scheduler = CosineAnnealingLR(self.optimizer,
                                            T_max=config['epochs'],
                                            eta_min=1e-6)
        self.criterion = FocalLoss(alpha=0.75, gamma=2.0)

        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        self.history = []

    def train_epoch(self, edge_index, edge_type, data):
        self.model.train()
        d1 = data['drug1'].to(self.device)
        d2 = data['drug2'].to(self.device)
        labels = data['labels'].to(self.device)

        ei = edge_index.to(self.device)
        et = edge_type.to(self.device)

        total_loss = 0
        correct = 0
        total = 0
        batch_size = self.config['batch_size']

        indices = torch.randperm(len(d1))
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            idx = indices[start:end]

            self.optimizer.zero_grad()
            logits = self.model(ei, et, d1[idx], d2[idx])
            loss = self.criterion(logits, labels[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(idx)
            pred = logits.argmax(dim=-1)
            correct += (pred == labels[idx]).sum().item()
            total += len(idx)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, edge_index, edge_type, data):
        self.model.eval()
        d1 = data['drug1'].to(self.device)
        d2 = data['drug2'].to(self.device)
        labels = data['labels'].to(self.device)

        ei = edge_index.to(self.device)
        et = edge_type.to(self.device)

        # 优化：评估时只编码一次全图
        node_emb = self.model.encode(ei, et)

        all_logits = []
        batch_size = self.config['batch_size']

        for start in range(0, len(d1), batch_size):
            end = min(start + batch_size, len(d1))
            pair = torch.cat([node_emb[d1[start:end]], node_emb[d2[start:end]]], dim=-1)
            logits = self.model.classifier(pair)
            all_logits.append(logits.cpu())

        all_logits = torch.cat(all_logits)
        probs = torch.softmax(all_logits, dim=-1)[:, 1].numpy()
        preds = all_logits.argmax(dim=-1).numpy()
        labels_np = labels.cpu().numpy()

        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

        metrics = {
            'loss': float(self.criterion(all_logits, labels.cpu()).item()),
            'accuracy': accuracy_score(labels_np, preds),
            'f1': f1_score(labels_np, preds, average='macro'),
            'precision': precision_score(labels_np, preds, average='macro', zero_division=0),
            'recall': recall_score(labels_np, preds, average='macro', zero_division=0),
        }

        try:
            metrics['auc'] = roc_auc_score(labels_np, probs)
        except ValueError:
            metrics['auc'] = 0.5

        return metrics

    def fit(self, edge_index, edge_type, train_data, val_data):
        epochs = self.config['epochs']
        patience = self.config['patience']

        print(f"\n{'='*60}", flush=True)
        print(f"  开始训练", flush=True)
        print(f"  Epochs: {epochs}, Patience: {patience}", flush=True)
        print(f"  LR: {self.config['lr']}, Batch: {self.config['batch_size']}", flush=True)
        print(f"{'='*60}\n", flush=True)

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self.train_epoch(edge_index, edge_type, train_data)
            val_metrics = self.evaluate(edge_index, edge_type, val_data)

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']

            elapsed = time.time() - t0

            record = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                **{f'val_{k}': v for k, v in val_metrics.items()},
                'lr': lr,
                'time': elapsed,
            }
            self.history.append(record)

            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
                  f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics['auc']:.4f} | "
                  f"LR: {lr:.6f} | {elapsed:.1f}s")

            # 早停
            if val_metrics['loss'] < self.best_val_loss - 1e-4:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
                print(f"  → 新的最佳模型 (val_loss={self.best_val_loss:.4f})", flush=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n早停触发于 Epoch {epoch}", flush=True)
                    break

        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n已恢复最佳模型 (best_val_loss={self.best_val_loss:.4f})", flush=True)

        return self.history


# ======================================================================
# 4. 主流程
# ======================================================================

def main():
    print("=" * 60, flush=True)
    print("  PharmSandbox GNN DDI 模型训练 (v2 全数据版)", flush=True)
    print("=" * 60, flush=True)

    data_root = DATA_ROOT
    model_root = MODEL_ROOT
    model_root.mkdir(parents=True, exist_ok=True)

    # ---- 配置（适配大数据量） ----
    config = {
        'hidden_dim': 128,      # 64→128
        'num_layers': 3,        # 2→3
        'dropout': 0.3,
        'lr': 0.002,            # 适中的学习率
        'weight_decay': 1e-4,
        'batch_size': 4096,     # 平衡：足够多的更新次数 + 合理的训练时间
        'epochs': 50,           # 足够收敛
        'patience': 10,
        'neg_ratio': 1,
    }

    print(f"\n配置: {json.dumps(config, indent=2)}", flush=True)
    print(f"数据目录: {data_root}", flush=True)
    print(f"模型目录: {model_root}", flush=True)

    # ---- 1. 加载所有DDI数据源 ----
    print(f"\n{'─'*60}", flush=True)
    print("阶段 1: 加载所有DDI数据源", flush=True)
    print(f"{'─'*60}", flush=True)

    # 1a. TWOSIDES 全量
    twosides_pairs, twosides_drugs = load_twosides(data_root)

    # 1b. Decagon
    decagon_pairs, decagon_drugs = load_decagon(data_root)

    # 1c. 合并所有DDI对（去重）
    all_ddi_set = set()
    for d1, d2 in twosides_pairs:
        all_ddi_set.add(tuple(sorted([d1, d2])))
    for d1, d2 in decagon_pairs:
        all_ddi_set.add(tuple(sorted([d1, d2])))

    ddi_pairs = list(all_ddi_set)
    print(f"\n[数据] 合并DDI数据集:", flush=True)
    print(f"  TWOSIDES: {len(twosides_pairs):,} 对", flush=True)
    print(f"  Decagon:  {len(decagon_pairs):,} 对", flush=True)
    print(f"  合并去重: {len(ddi_pairs):,} 对", flush=True)

    # 1d. 加载SIDER药物名
    sider_drugs = load_sider_drugs(data_root)

    # 1e. 合并所有药物名
    all_drugs = sider_drugs | twosides_drugs | decagon_drugs
    for d1, d2 in ddi_pairs:
        all_drugs.add(d1)
        all_drugs.add(d2)
    print(f"[数据] 药物并集: {len(all_drugs):,} 种", flush=True)

    # 1f. 构建药物索引
    drug_to_idx, drug_list = build_drug_index(all_drugs)

    # ---- 2. 构建知识图谱 ----
    print(f"\n{'─'*60}", flush=True)
    print("阶段 2: 构建知识图谱", flush=True)
    print(f"{'─'*60}", flush=True)

    edges, edge_type_names, total_nodes, num_drugs = build_knowledge_graph_edges(data_root, drug_to_idx)

    edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
    edge_type = torch.tensor([e[2] for e in edges], dtype=torch.long)

    # ---- 3. 准备训练数据 ----
    print(f"\n{'─'*60}", flush=True)
    print("阶段 3: 准备训练数据", flush=True)
    print(f"{'─'*60}", flush=True)

    hard_negatives, _ = load_hard_negatives(data_root)
    data = prepare_training_data(ddi_pairs, drug_to_idx, num_drugs,
                                 neg_ratio=config['neg_ratio'],
                                 hard_negatives=hard_negatives)

    # ---- 4. 训练 ----
    print(f"\n{'─'*60}", flush=True)
    print("阶段 4: 训练模型", flush=True)
    print(f"{'─'*60}", flush=True)

    model = DDIPredictor(
        num_nodes=total_nodes,
        num_relations=len(edge_type_names),
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=2,
        dropout=config['dropout'],
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count:,}", flush=True)

    trainer = Trainer(model, config)
    history = trainer.fit(edge_index, edge_type, data['train'], data['val'])

    # ---- 5. 测试集评估 ----
    print(f"\n{'─'*60}", flush=True)
    print("阶段 5: 测试集评估", flush=True)
    print(f"{'─'*60}", flush=True)

    test_metrics = trainer.evaluate(edge_index, edge_type, data['test'])
    print(f"\n测试集结果:", flush=True)
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}", flush=True)
    print(f"  F1 (macro): {test_metrics['f1']:.4f}", flush=True)
    print(f"  AUC:       {test_metrics['auc']:.4f}", flush=True)
    print(f"  Precision: {test_metrics['precision']:.4f}", flush=True)
    print(f"  Recall:    {test_metrics['recall']:.4f}", flush=True)

    # ---- 6. 保存模型 ----
    print(f"\n{'─'*60}", flush=True)
    print("阶段 6: 保存模型", flush=True)
    print(f"{'─'*60}", flush=True)

    # 备份旧模型
    old_model_path = model_root / "best_model.pt"
    if old_model_path.exists():
        backup_path = model_root / "best_model_v1_backup.pt"
        import shutil
        shutil.copy2(old_model_path, backup_path)
        print(f"旧模型已备份: {backup_path}", flush=True)

    save_path = model_root / "best_model.pt"
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'test_metrics': test_metrics,
        'drug_to_idx': drug_to_idx,
        'edge_type_names': edge_type_names,
        'total_nodes': total_nodes,
        'num_drugs': num_drugs,
        'history': history,
        'training_version': 'v2_full_data',
        'data_sources': {
            'twosides_pairs': len(twosides_pairs),
            'decagon_pairs': len(decagon_pairs),
            'merged_ddi_pairs': len(ddi_pairs),
            'total_drugs': len(all_drugs),
        },
    }
    torch.save(checkpoint, save_path)
    print(f"模型已保存: {save_path}", flush=True)
    print(f"文件大小: {save_path.stat().st_size / 1024 / 1024:.2f} MB", flush=True)

    # 保存训练历史
    history_path = model_root / "training_history.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"训练历史: {history_path}", flush=True)

    # ---- 总结 ----
    print(f"\n{'='*60}", flush=True)
    print(f"  训练完成!", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  药物数: {num_drugs}", flush=True)
    print(f"  知识图谱: {total_nodes} 节点, {len(edges)} 边", flush=True)
    print(f"  DDI对: {len(ddi_pairs):,} (TWOSIDES {len(twosides_pairs):,} + Decagon {len(decagon_pairs):,})", flush=True)
    print(f"  训练样本: {len(data['train']['labels']):,}", flush=True)
    print(f"  测试 AUC: {test_metrics['auc']:.4f}", flush=True)
    print(f"  测试 F1:  {test_metrics['f1']:.4f}", flush=True)
    print(f"  模型路径: {save_path}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
