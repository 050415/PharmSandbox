"""
药物知识图谱构建模块
从SIDER / DrugCentral / TWOSIDES数据构建NetworkX异构知识图谱，
并导出GNN训练所需的边列表和特征矩阵。

节点类型: drug, side_effect, indication, target_protein
边类型:   has_side_effect, has_indication, targets, interacts_with
"""
import os
import pickle
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

# 导入项目内DataLoader
import sys
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.data.loader import DataLoader


class DrugKnowledgeGraph:
    """药物异构知识图谱

    Parameters
    ----------
    data_root : str
        数据根目录，默认 ``D:/drug/data``。
    fingerprint_bits : int
        分子指纹位数（Morgan fingerprint），默认 1024。
    """

    NODE_TYPES = ("drug", "side_effect", "indication", "target_protein")
    EDGE_TYPES = ("has_side_effect", "has_indication", "targets", "interacts_with")

    def __init__(self, data_root=None, fingerprint_bits: int = 1024):
        if data_root is None:
            from src.config import DATA_ROOT
            data_root = str(DATA_ROOT)
        self.data_root = Path(data_root)
        self.fingerprint_bits = fingerprint_bits
        self.loader = DataLoader(data_root=str(self.data_root))

        # NetworkX 异构图
        self.G: nx.DiGraph = nx.DiGraph()

        # 节点 ID 映射（类型 -> 名称 -> 内部ID）
        self._node_index: Dict[str, Dict[str, int]] = {t: {} for t in self.NODE_TYPES}
        # 反向映射
        self._index_to_node: Dict[str, Dict[int, str]] = {t: {} for t in self.NODE_TYPES}

        # 药物特征矩阵（分子指纹）
        self.drug_features: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 节点管理
    # ------------------------------------------------------------------

    def _add_node(self, node_type: str, name: str) -> int:
        """添加节点并返回内部 ID。"""
        if name in self._node_index[node_type]:
            return self._node_index[node_type][name]
        idx = len(self._node_index[node_type])
        self._node_index[node_type][name] = idx
        self._index_to_node[node_type][idx] = name
        self.G.add_node(f"{node_type}:{name}", node_type=node_type, name=name)
        return idx

    def _node_id(self, node_type: str, name: str) -> Optional[int]:
        return self._node_index[node_type].get(name)

    # ------------------------------------------------------------------
    # 构建图谱
    # ------------------------------------------------------------------

    def build(self) -> "DrugKnowledgeGraph":
        """从 DataLoader 读取所有数据并构建知识图谱。"""
        print("[KG] 开始构建药物知识图谱 ...")

        # 1) 药物名称 -> drug 节点
        self._load_drug_names()
        # 2) 副作用关系
        self._load_side_effects()
        # 3) 适应症关系
        self._load_indications()
        # 4) 药物-靶点关系
        self._load_targets()
        # 5) 药物相互作用（TWOSIDES）
        self._load_interactions()
        # 6) 分子指纹特征
        self._compute_fingerprints()

        self._print_summary()
        return self

    def _load_drug_names(self):
        """加载药物名称，创建 drug 节点。"""
        try:
            df = self.loader.load_drug_names()
            for _, row in df.iterrows():
                drug_name = str(row["drug_name"]).strip()
                if drug_name:
                    self._add_node("drug", drug_name)
            print(f"  [drug] {len(self._node_index['drug'])} 个药物节点")
        except Exception as e:
            print(f"  [drug] 加载失败: {e}")

    def _load_side_effects(self):
        """加载 SIDER 副作用，建立 has_side_effect 边。"""
        try:
            df = self.loader.load_side_effects()
            for _, row in df.iterrows():
                cid = str(row["cid"]).strip()
                se_name = str(row["side_effect_name"]).strip()
                if not se_name or se_name == "nan":
                    continue
                # 通过 cid 查找对应药物名
                drug_name = self._cid_to_drug(cid)
                if drug_name is None:
                    continue
                drug_idx = self._add_node("drug", drug_name)
                se_idx = self._add_node("side_effect", se_name)
                self.G.add_edge(
                    f"drug:{drug_name}",
                    f"side_effect:{se_name}",
                    edge_type="has_side_effect",
                )
            n_se = len(self._node_index["side_effect"])
            print(f"  [side_effect] {n_se} 个副作用节点")
        except Exception as e:
            print(f"  [side_effect] 加载失败: {e}")

    def _load_indications(self):
        """加载 SIDER 适应症，建立 has_indication 边。"""
        try:
            df = self.loader.load_indications()
            for _, row in df.iterrows():
                cid = str(row["cid"]).strip()
                ind_name = str(row["indication_name"]).strip()
                if not ind_name or ind_name == "nan":
                    continue
                drug_name = self._cid_to_drug(cid)
                if drug_name is None:
                    continue
                self._add_node("drug", drug_name)
                self._add_node("indication", ind_name)
                self.G.add_edge(
                    f"drug:{drug_name}",
                    f"indication:{ind_name}",
                    edge_type="has_indication",
                )
            n_ind = len(self._node_index["indication"])
            print(f"  [indication] {n_ind} 个适应症节点")
        except Exception as e:
            print(f"  [indication] 加载失败: {e}")

    def _load_targets(self):
        """加载 DrugCentral 药物-靶点，建立 targets 边。"""
        try:
            df = self.loader.load_drug_target_interactions()
            # 列名取决于文件；尝试常见字段
            drug_col = self._find_column(df, ["drug_name", "DRUG_NAME", "name", "generic_name1"])
            target_col = self._find_column(df, ["target_name", "TARGET_NAME", "target", "uniprot", "gene"])
            if drug_col is None or target_col is None:
                print(f"  [targets] 无法识别列名: {list(df.columns)}")
                return
            for _, row in df.iterrows():
                drug_name = str(row[drug_col]).strip()
                target_name = str(row[target_col]).strip()
                if not drug_name or drug_name == "nan" or not target_name or target_name == "nan":
                    continue
                self._add_node("drug", drug_name)
                self._add_node("target_protein", target_name)
                self.G.add_edge(
                    f"drug:{drug_name}",
                    f"target_protein:{target_name}",
                    edge_type="targets",
                )
            n_tp = len(self._node_index["target_protein"])
            print(f"  [target_protein] {n_tp} 个靶点节点")
        except Exception as e:
            print(f"  [targets] 加载失败: {e}")

    def _load_interactions(self):
        """加载 TWOSIDES 药物-药物相互作用，建立 interacts_with 边。"""
        try:
            df = self.loader.load_twosides()
            if df.empty:
                print("  [interacts_with] TWOSIDES 数据为空，跳过")
                return
            # TWOSIDES 常见列：drug_1_concept_name, drug_2_concept_name
            d1_col = self._find_column(df, ["drug_1_concept_name", "drug1", "drug_1"])
            d2_col = self._find_column(df, ["drug_2_concept_name", "drug2", "drug_2"])
            if d1_col is None or d2_col is None:
                print(f"  [interacts_with] 无法识别列名: {list(df.columns)}")
                return
            seen = set()
            for _, row in df.iterrows():
                d1 = str(row[d1_col]).strip()
                d2 = str(row[d2_col]).strip()
                if not d1 or d1 == "nan" or not d2 or d2 == "nan":
                    continue
                pair = tuple(sorted([d1, d2]))
                if pair in seen:
                    continue
                seen.add(pair)
                self._add_node("drug", d1)
                self._add_node("drug", d2)
                self.G.add_edge(f"drug:{d1}", f"drug:{d2}", edge_type="interacts_with")
            print(f"  [interacts_with] {len(seen)} 对药物相互作用")
        except Exception as e:
            print(f"  [interacts_with] 加载失败: {e}")

    # ------------------------------------------------------------------
    # 分子指纹
    # ------------------------------------------------------------------

    def _compute_fingerprints(self):
        """使用 RDKit 计算药物 Morgan 指纹，生成特征矩阵。"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            print("  [fingerprint] RDKit 未安装，使用随机特征作为 fallback")
            self._random_fallback_features()
            return

        try:
            smiles_df = self.loader.load_drug_smiles()
            smiles_col = self._find_column(smiles_df, ["SMILES", "smiles", "canonical_smiles"])
            name_col = self._find_column(smiles_df, ["name", "drug_name", "generic_name", "STRUCTURE_NAME"])
            if smiles_col is None:
                print(f"  [fingerprint] 无法识别 SMILES 列: {list(smiles_df.columns)}")
                self._random_fallback_features()
                return

            n_drugs = len(self._node_index["drug"])
            self.drug_features = np.zeros((n_drugs, self.fingerprint_bits), dtype=np.float32)

            matched = 0
            for _, row in smiles_df.iterrows():
                smiles = str(row[smiles_col]).strip()
                drug_name = str(row[name_col]).strip() if name_col else None
                if not smiles or smiles == "nan":
                    continue
                # 匹配药物
                drug_idx = None
                if drug_name and drug_name in self._node_index["drug"]:
                    drug_idx = self._node_index["drug"][drug_name]
                else:
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.fingerprint_bits)
                arr = np.zeros(self.fingerprint_bits, dtype=np.float32)
                fp_on_bits = fp.GetOnBits()
                for bit in fp_on_bits:
                    arr[bit] = 1.0
                self.drug_features[drug_idx] = arr
                matched += 1

            print(f"  [fingerprint] 成功计算 {matched}/{n_drugs} 个药物的 Morgan 指纹")
            # 未匹配的药物用随机特征填充
            zero_mask = np.all(self.drug_features == 0, axis=1)
            if zero_mask.any():
                self.drug_features[zero_mask] = np.random.default_rng(42).standard_normal(
                    (zero_mask.sum(), self.fingerprint_bits)
                ).astype(np.float32) * 0.01
        except Exception as e:
            print(f"  [fingerprint] 计算失败: {e}")
            self._random_fallback_features()

    def _random_fallback_features(self):
        """当 RDKit 不可用时，生成随机特征。"""
        n_drugs = len(self._node_index["drug"])
        rng = np.random.default_rng(42)
        self.drug_features = rng.standard_normal((n_drugs, self.fingerprint_bits)).astype(np.float32) * 0.01
        print(f"  [fingerprint] 生成随机特征: ({n_drugs}, {self.fingerprint_bits})")

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get_neighbors(self, drug_name: str,
                      edge_type: Optional[str] = None) -> Dict[str, List[str]]:
        """查询指定药物的邻居节点。

        Parameters
        ----------
        drug_name : str
            药物名称。
        edge_type : str, optional
            限制边类型；为 None 时返回所有邻居。

        Returns
        -------
        dict
            {node_type: [name, ...]}
        """
        node_key = f"drug:{drug_name}"
        if node_key not in self.G:
            return {}
        neighbors: Dict[str, List[str]] = {}
        for _, tgt, data in self.G.edges(node_key, data=True):
            if edge_type and data.get("edge_type") != edge_type:
                continue
            if ":" in tgt:
                ntype, nname = tgt.split(":", 1)
            else:
                ntype, nname = "unknown", tgt
            neighbors.setdefault(ntype, []).append(nname)
        return neighbors

    def get_drug_subgraph(self, drug_name: str, hops: int = 1) -> nx.DiGraph:
        """获取以某药物为中心的 k-hop 子图。"""
        node_key = f"drug:{drug_name}"
        if node_key not in self.G:
            return nx.DiGraph()
        nodes = {node_key}
        frontier = {node_key}
        for _ in range(hops):
            next_frontier = set()
            for n in frontier:
                for _, tgt in self.G.edges(n):
                    if tgt not in nodes:
                        next_frontier.add(tgt)
                for src, _ in self.G.in_edges(n):
                    if src not in nodes:
                        next_frontier.add(src)
            nodes |= next_frontier
            frontier = next_frontier
        return self.G.subgraph(nodes).copy()

    def search_drugs(self, keyword: str) -> List[str]:
        """按关键词模糊搜索药物名称。"""
        kw = keyword.lower()
        return [d for d in self._node_index["drug"] if kw in d.lower()]

    # ------------------------------------------------------------------
    # GNN 数据导出
    # ------------------------------------------------------------------

    def export_for_gnn(self, output_dir=None):
        if output_dir is None:
            output_dir = str(self.data_root / "gnn")
        """导出 GNN 训练所需的边列表、特征矩阵和节点映射。

        输出文件：
        - ``edge_list.csv``   : src_idx, dst_idx, edge_type
        - ``drug_features.npy``: (n_drugs, fingerprint_bits)
        - ``node_mapping.pkl``: {node_type: {name: idx}}
        - ``graph_stats.json``: 图统计信息
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 边列表（转为数字索引）
        edges = []
        for src, tgt, data in self.G.edges(data=True):
            etype = data.get("edge_type", "unknown")
            src_type, src_name = src.split(":", 1)
            tgt_type, tgt_name = tgt.split(":", 1)
            src_idx = self._node_index[src_type].get(src_name)
            tgt_idx = self._node_index[tgt_type].get(tgt_name)
            if src_idx is not None and tgt_idx is not None:
                edges.append((src_idx, tgt_idx, etype))

        edge_df = pd.DataFrame(edges, columns=["src_idx", "dst_idx", "edge_type"])
        edge_df.to_csv(out / "edge_list.csv", index=False)

        # 药物特征矩阵
        if self.drug_features is not None:
            np.save(out / "drug_features.npy", self.drug_features)

        # 节点映射
        with open(out / "node_mapping.pkl", "wb") as f:
            pickle.dump(self._node_index, f)

        # 统计信息
        import json
        stats = {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "node_counts": {t: len(self._node_index[t]) for t in self.NODE_TYPES},
            "edge_counts": {},
            "fingerprint_bits": self.fingerprint_bits,
        }
        for _, _, d in self.G.edges(data=True):
            et = d.get("edge_type", "unknown")
            stats["edge_counts"][et] = stats["edge_counts"].get(et, 0) + 1
        with open(out / "graph_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"[KG] GNN 数据已导出到 {out}")
        return out

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, path=None):
        if path is None:
            path = str(self.data_root / "gnn" / "knowledge_graph.pkl")
        """序列化整个图谱到 pickle。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({
                "graph": self.G,
                "node_index": self._node_index,
                "index_to_node": self._index_to_node,
                "drug_features": self.drug_features,
                "fingerprint_bits": self.fingerprint_bits,
            }, f)
        print(f"[KG] 图谱已保存到 {p}")

    @classmethod
    def load(cls, path=None) -> "DrugKnowledgeGraph":
        if path is None:
            from src.config import DATA_ROOT
            path = str(DATA_ROOT / "gnn" / "knowledge_graph.pkl")
        """从 pickle 加载图谱。"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        kg = cls(fingerprint_bits=data["fingerprint_bits"])
        kg.G = data["graph"]
        kg._node_index = data["node_index"]
        kg._index_to_node = data["index_to_node"]
        kg.drug_features = data["drug_features"]
        return kg

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _cid_to_drug(self, cid: str) -> Optional[str]:
        """将 SIDER CID 映射回药物名称。"""
        if not hasattr(self, "_cid_map"):
            try:
                dn = self.loader.load_drug_names()
                self._cid_map = {}
                for _, row in dn.iterrows():
                    c = str(row["cid"]).strip()
                    name = str(row["drug_name"]).strip()
                    if c and name and name != "nan":
                        self._cid_map[c] = name
            except Exception:
                self._cid_map = {}
        return self._cid_map.get(cid)

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """在 DataFrame 列中查找第一个匹配项。"""
        cols_lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        return None

    def _print_summary(self):
        """打印图谱摘要。"""
        print("=" * 50)
        print("药物知识图谱摘要")
        print("=" * 50)
        print(f"  总节点数: {self.G.number_of_nodes()}")
        print(f"  总边数:   {self.G.number_of_edges()}")
        for t in self.NODE_TYPES:
            print(f"  {t:20s}: {len(self._node_index[t]):>6d} 个节点")
        edge_counts = {}
        for _, _, d in self.G.edges(data=True):
            et = d.get("edge_type", "unknown")
            edge_counts[et] = edge_counts.get(et, 0) + 1
        for et, cnt in edge_counts.items():
            print(f"  {et:20s}: {cnt:>6d} 条边")
        if self.drug_features is not None:
            print(f"  药物特征矩阵: {self.drug_features.shape}")
        print("=" * 50)


# ======================================================================
# CLI 入口
# ======================================================================

if __name__ == "__main__":
    kg = DrugKnowledgeGraph()
    kg.build()
    kg.export_for_gnn()
    kg.save()

    # 示例查询
    sample_drugs = list(kg._node_index["drug"].keys())[:5]
    for d in sample_drugs:
        neighbors = kg.get_neighbors(d)
        total = sum(len(v) for v in neighbors.values())
        print(f"\n  {d}: {total} 个邻居")
        for ntype, names in neighbors.items():
            print(f"    {ntype}: {names[:3]}{'...' if len(names) > 3 else ''}")
