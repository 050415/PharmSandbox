# -*- coding: utf-8 -*-
"""
GNN DDI模型训练器 - PharmSandbox 药盘推演课设

训练逻辑：知识图谱子图采样 + RGCN训练 + 多类型DDI分类
支持：训练、验证、测试、模型保存/加载
"""

import os
import time
import logging
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np

from .model import DDIPredictor, build_model

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型参数
    num_nodes: int = 1000
    num_relations: int = 5
    num_ddi_types: int = 86
    hidden_dim: int = 128
    num_rgcn_layers: int = 3
    dropout: float = 0.3
    node_emb_dim: int = 128
    
    # 训练参数
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler: str = 'cosine'  # 'cosine' | 'plateau' | 'none'
    warmup_epochs: int = 5
    
    # 子图采样
    subgraph_hop: int = 2          # k跳采样
    max_subgraph_nodes: int = 200  # 子图最大节点数
    neg_samples: int = 5           # 负采样倍数
    
    # 早停
    patience: int = 15
    min_delta: float = 1e-4
    
    # 设备
    device: str = 'cpu'
    
    # 保存
    save_dir: str = 'checkpoints'
    save_every: int = 10


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_acc: float = 0.0
    val_acc: float = 0.0
    val_f1_macro: float = 0.0
    val_f1_micro: float = 0.0
    val_auroc: float = 0.0
    lr: float = 0.0
    time_elapsed: float = 0.0


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class DDITrainer:
    """
    DDI模型训练器
    
    功能：
    - 训练循环（含warmup + 梯度裁剪）
    - 验证与指标计算
    - 早停与学习率调度
    - 模型保存/加载
    - 推理接口
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        train_data: Optional[Dict] = None,
        val_data: Optional[Dict] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        # 构建模型
        self.model = build_model(
            num_nodes=config.num_nodes,
            num_relations=config.num_relations,
            num_ddi_types=config.num_ddi_types,
            hidden_dim=config.hidden_dim,
            num_rgcn_layers=config.num_rgcn_layers,
            dropout=config.dropout,
            node_emb_dim=config.node_emb_dim,
            device=config.device,
        )
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # 学习率调度
        self.scheduler = self._build_scheduler()
        
        # 损失函数（类别加权可选）
        self.criterion = nn.CrossEntropyLoss()
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
        )
        
        # 数据
        self.train_data = train_data
        self.val_data = val_data
        
        # 训练历史
        self.history: List[TrainingMetrics] = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def _build_scheduler(self):
        cfg = self.config
        if cfg.lr_scheduler == 'cosine':
            return CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs, eta_min=1e-6
            )
        elif cfg.lr_scheduler == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        return None
    
    def _compute_accuracy(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> float:
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == labels).sum().item()
        return correct / labels.shape[0]
    
    def _compute_f1(
        self, logits: torch.Tensor, labels: torch.Tensor, num_classes: int
    ) -> Tuple[float, float]:
        """计算 macro-F1 和 micro-F1"""
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        tp = np.zeros(num_classes)
        fp = np.zeros(num_classes)
        fn = np.zeros(num_classes)
        
        for c in range(num_classes):
            tp[c] = np.sum((preds == c) & (labels_np == c))
            fp[c] = np.sum((preds == c) & (labels_np != c))
            fn[c] = np.sum((preds != c) & (labels_np == c))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_per_class = 2 * precision * recall / (precision + recall + 1e-8)
        
        # macro-F1
        macro_f1 = float(np.mean(f1_per_class))
        
        # micro-F1
        total_tp = np.sum(tp)
        total_fp = np.sum(fp)
        total_fn = np.sum(fn)
        micro_p = total_tp / (total_tp + total_fp + 1e-8)
        micro_r = total_tp / (total_tp + total_fn + 1e-8)
        micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r + 1e-8))
        
        return macro_f1, micro_f1
    
    def train_epoch(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        drug1_idx: torch.Tensor,
        drug2_idx: torch.Tensor,
        labels: torch.Tensor,
        num_nodes: int,
        batch_size: int = 256,
    ) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Returns:
            (avg_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        indices = torch.randperm(drug1_idx.shape[0])
        
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_idx = indices[start:end]
            
            d1 = drug1_idx[batch_idx].to(self.device)
            d2 = drug2_idx[batch_idx].to(self.device)
            y = labels[batch_idx].to(self.device)
            
            ei = edge_index.to(self.device)
            et = edge_type.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(ei, et, d1, d2, num_nodes)
            loss = self.criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            acc = self._compute_accuracy(logits, y)
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

        return total_loss / num_batches, total_acc / num_batches

    @torch.no_grad()
    def evaluate(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        drug1_idx: torch.Tensor,
        drug2_idx: torch.Tensor,
        labels: torch.Tensor,
        num_nodes: int,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """
        验证/测试

        Returns:
            dict with loss, accuracy, macro_f1, micro_f1
        """
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        all_logits = []
        all_labels = []

        for start in range(0, drug1_idx.shape[0], batch_size):
            end = min(start + batch_size, drug1_idx.shape[0])

            d1 = drug1_idx[start:end].to(self.device)
            d2 = drug2_idx[start:end].to(self.device)
            y = labels[start:end].to(self.device)

            ei = edge_index.to(self.device)
            et = edge_type.to(self.device)

            logits = self.model(ei, et, d1, d2, num_nodes)
            loss = self.criterion(logits, y)
            
            acc = self._compute_accuracy(logits, y)
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
            
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        macro_f1, micro_f1 = self._compute_f1(
            all_logits, all_labels, self.config.num_ddi_types
        )
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_acc / num_batches,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
        }
    
    def fit(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        train_drug1: torch.Tensor,
        train_drug2: torch.Tensor,
        train_labels: torch.Tensor,
        val_drug1: Optional[torch.Tensor] = None,
        val_drug2: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> List[TrainingMetrics]:
        """
        完整训练流程
        
        Args:
            edge_index: 知识图谱边
            edge_type: 边类型
            train_drug1/2, train_labels: 训练集
            val_drug1/2, val_labels: 验证集
            num_nodes: 节点数
            
        Returns:
            训练历史
        """
        if num_nodes is None:
            num_nodes = edge_index.max().item() + 1
        
        os.makedirs(self.config.save_dir, exist_ok=True)
        logger.info("开始训练...")
        logger.info(f"  Epochs: {self.config.epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Device: {self.device}")
        
        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            
            # Warmup
            if epoch <= self.config.warmup_epochs:
                warmup_lr = self.config.learning_rate * epoch / self.config.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = warmup_lr
            
            # 训练
            train_loss, train_acc = self.train_epoch(
                edge_index, edge_type,
                train_drug1, train_drug2, train_labels,
                num_nodes, self.config.batch_size,
            )
            
            # 验证
            val_metrics = {'loss': 0, 'accuracy': 0, 'macro_f1': 0, 'micro_f1': 0}
            if val_drug1 is not None:
                val_metrics = self.evaluate(
                    edge_index, edge_type,
                    val_drug1, val_drug2, val_labels,
                    num_nodes, self.config.batch_size,
                )
            
            # 学习率调度
            if epoch > self.config.warmup_epochs:
                if self.config.lr_scheduler == 'cosine':
                    self.scheduler.step()
                elif self.config.lr_scheduler == 'plateau':
                    self.scheduler.step(val_metrics['loss'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录指标
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_metrics['loss'],
                train_acc=train_acc,
                val_acc=val_metrics['accuracy'],
                val_f1_macro=val_metrics['macro_f1'],
                val_f1_micro=val_metrics['micro_f1'],
                lr=current_lr,
                time_elapsed=time.time() - t0,
            )
            self.history.append(metrics)
            
            # 日志
            logger.info(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
                f"F1: {val_metrics['macro_f1']:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {metrics.time_elapsed:.1f}s"
            )
            
            # 保存最佳模型
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                self.save_checkpoint(os.path.join(self.config.save_dir, 'best_model.pt'))
                logger.info(f"  -> 新的最佳模型已保存 (val_loss={self.best_val_loss:.4f})")
            
            # 定期保存
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(
                    os.path.join(self.config.save_dir, f'checkpoint_epoch{epoch}.pt')
                )
            
            # 早停
            if val_drug1 is not None and self.early_stopping.step(val_metrics['loss']):
                logger.info(f"早停触发于 Epoch {epoch}")
                break
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"已恢复最佳模型 (best_val_loss={self.best_val_loss:.4f})")
        
        return self.history
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        torch.save(checkpoint, path)
        logger.debug(f"检查点已保存: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', [])
        logger.info(f"检查点已加载: {path}")
    
    @torch.no_grad()
    def predict_batch(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        drug1_idx: torch.Tensor,
        drug2_idx: torch.Tensor,
        num_nodes: int,
    ) -> Dict[str, torch.Tensor]:
        """
        批量预测DDI类型和置信度
        
        Args:
            edge_index: [2, E]
            edge_type: [E]
            drug1_idx: [B]
            drug2_idx: [B]
            num_nodes: 节点数
            
        Returns:
            dict with predictions, probabilities, confidence
        """
        self.model.eval()
        return self.model.predict(
            edge_index.to(self.device),
            edge_type.to(self.device),
            drug1_idx.to(self.device),
            drug2_idx.to(self.device),
            num_nodes,
        )


# --- 辅助函数 ---

def create_synthetic_data(
    num_nodes: int = 500,
    num_relations: int = 5,
    num_ddi_types: int = 86,
    num_train: int = 2000,
    num_val: int = 500,
    num_edges: int = 5000,
) -> Dict:
    """生成合成训练数据（用于测试流程）"""
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))
    
    train_data = {
        'drug1': torch.randint(0, num_nodes, (num_train,)),
        'drug2': torch.randint(0, num_nodes, (num_train,)),
        'labels': torch.randint(0, num_ddi_types, (num_train,)),
    }
    
    val_data = {
        'drug1': torch.randint(0, num_nodes, (num_val,)),
        'drug2': torch.randint(0, num_nodes, (num_val,)),
        'labels': torch.randint(0, num_ddi_types, (num_val,)),
    }
    
    return {
        'edge_index': edge_index,
        'edge_type': edge_type,
        'num_nodes': num_nodes,
        'train': train_data,
        'val': val_data,
    }


# --- 示例用法 ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # 创建合成数据
    data = create_synthetic_data()
    
    # 配置
    config = TrainingConfig(
        num_nodes=data['num_nodes'],
        num_relations=5,
        num_ddi_types=86,
        hidden_dim=64,
        num_rgcn_layers=2,
        epochs=5,
        batch_size=128,
        learning_rate=1e-3,
        device='cpu',
        save_dir='checkpoints',
    )
    
    # 训练
    trainer = DDITrainer(config)
    history = trainer.fit(
        edge_index=data['edge_index'],
        edge_type=data['edge_type'],
        train_drug1=data['train']['drug1'],
        train_drug2=data['train']['drug2'],
        train_labels=data['train']['labels'],
        val_drug1=data['val']['drug1'],
        val_drug2=data['val']['drug2'],
        val_labels=data['val']['labels'],
        num_nodes=data['num_nodes'],
    )
    
    # 预测
    result = trainer.predict_batch(
        edge_index=data['edge_index'],
        edge_type=data['edge_type'],
        drug1_idx=data['val']['drug1'][:10],
        drug2_idx=data['val']['drug2'][:10],
        num_nodes=data['num_nodes'],
    )
    print(f"\n预测结果: {result['predictions']}")
    print(f"置信度:   {result['confidence']}")
