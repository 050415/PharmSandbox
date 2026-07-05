"""
PharmSandbox - GNN药物相互作用预测模型
基于SumGNN的RGCN架构，使用PyTorch Geometric实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import numpy as np


class DDIPredictor(nn.Module):
    """
    药物相互作用预测器
    使用RGCN编码药物知识图谱，预测DDI类型
    """

    def __init__(self, num_nodes, num_relations, hidden_dim=64, num_layers=3,
                 num_classes=2, dropout=0.3):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 节点嵌入层
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)

        # RGCN卷积层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # DDI分类器（与训练脚本一致：2层）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.dropout = dropout
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
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
    
    def forward(self, drug1_idx, drug2_idx, edge_index=None, edge_type=None,
                batch=None, num_nodes=None):
        """
        前向传播，兼容两种调用方式::

            # 方式1（原始）:
            model(drug1_idx, drug2_idx, edge_index, edge_type)
            # 方式2（DDITrainer兼容）:
            model(edge_index, edge_type, drug1_idx, drug2_idx, num_nodes)

        Returns:
            ddi_logits: DDI分类logits [B, num_classes]
        """
        # 自动检测调用方式
        if (edge_index is not None and
            isinstance(drug1_idx, torch.Tensor) and drug1_idx.dim() == 2 and
            isinstance(edge_index, torch.Tensor) and edge_index.dim() == 1):
            drug1_idx, drug2_idx, edge_index, edge_type = (
                edge_index, edge_type, drug1_idx, drug2_idx
            )

        # 编码整个图，获取节点嵌入
        x = self.encode(edge_index, edge_type)

        # 获取药物对的表示并拼接
        pair_repr = torch.cat([x[drug1_idx], x[drug2_idx]], dim=-1)

        # DDI分类
        return self.classifier(pair_repr)
    
    def predict(self, drug1_idx, drug2_idx, edge_index=None, edge_type=None,
                num_nodes=None):
        """
        预测DDI，兼容两种调用方式。

        Returns:
            dict with predictions, probabilities, confidence
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(drug1_idx, drug2_idx, edge_index, edge_type)
            probs = torch.softmax(logits, dim=-1)
            ddi_type = torch.argmax(probs, dim=-1)

        return {
            'predictions': ddi_type,
            'probabilities': probs,
            'confidence': probs.max(dim=-1).values,
        }


class GNNTrainer:
    """GNN训练器"""
    
    def __init__(self, model, lr=0.001, weight_decay=1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train_epoch(self, data_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in data_loader:
            self.optimizer.zero_grad()

            logits = self.model(
                batch['drug1_idx'], batch['drug2_idx'],
                batch['edge_index'], batch['edge_type'], batch.get('batch')
            )

            loss = self.criterion(logits, batch['labels'])
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == batch['labels']).sum().item()
            total += batch['labels'].size(0)

        return total_loss / len(data_loader), correct / total

    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                logits = self.model(
                    batch['drug1_idx'], batch['drug2_idx'],
                    batch['edge_index'], batch['edge_type'], batch.get('batch')
                )

                loss = self.criterion(logits, batch['labels'])
                total_loss += loss.item()
                pred = torch.argmax(logits, dim=-1)
                correct += (pred == batch['labels']).sum().item()
                total += batch['labels'].size(0)

        return total_loss / max(len(data_loader), 1), correct / max(total, 1)
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)


def build_model(
    num_nodes: int,
    num_relations: int,
    num_ddi_types: int = 2,
    hidden_dim: int = 128,
    num_rgcn_layers: int = 3,
    dropout: float = 0.3,
    node_emb_dim: int = 128,
    device: str = 'cpu',
) -> DDIPredictor:
    """
    工厂函数：构建 DDIPredictor 模型。

    供 DDITrainer 调用::

        model = build_model(num_nodes=1000, num_relations=5, num_ddi_types=86)
    """
    # RGCN 要求 node_emb_dim == hidden_dim，强制对齐
    if node_emb_dim != hidden_dim:
        hidden_dim = node_emb_dim
    model = DDIPredictor(
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_dim=hidden_dim,
        num_layers=num_rgcn_layers,
        num_classes=num_ddi_types,
        dropout=dropout,
    )
    return model.to(device)


if __name__ == "__main__":
    # 测试模型创建
    model = DDIPredictor(
        num_nodes=5000,
        num_relations=10,
        hidden_dim=64,
        num_layers=3,
        num_classes=2
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("GNN模型创建成功！")

    # 测试 build_model
    model2 = build_model(num_nodes=1000, num_relations=5, num_ddi_types=86)
    print(f"build_model 参数量: {sum(p.numel() for p in model2.parameters()):,}")
    print("build_model 测试成功！")
