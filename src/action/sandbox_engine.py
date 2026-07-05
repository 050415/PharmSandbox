"""
PharmSandbox - 综合沙盘推演引擎
整合所有模块：GNN DDI推演 + LLM推理 + 风险评分 + NER + 替代药推荐 + 续方触发
"""
import pickle
from collections import defaultdict
from pathlib import Path
from src.config import get_logger, MODEL_ROOT, DATA_ROOT, ENABLE_GNN_INFERENCE
from src.data.loader import DataLoader
from src.decision.risk_scorer import RiskScorer
from src.decision.recommender import DrugRecommender
from src.decision.prescription_trigger import PrescriptionTrigger

logger = get_logger("sandbox_engine")


class SandboxEngine:
    """
    综合沙盘推演引擎
    整合所有模块，提供一站式推演接口
    """

    def __init__(self, data_root=None):
        print("正在初始化沙盘推演引擎...")
        self.data_root = Path(data_root) if data_root else DATA_ROOT
        self.data_loader = DataLoader(data_root)
        self.recommender = DrugRecommender(data_root)
        self.prescription_trigger = PrescriptionTrigger()

        # 统一加载副作用数据，避免 RiskScorer 重复加载
        drug_se_map = self._build_drug_se_map()

        # 尝试加载化验数据加载器（用于风险评分的化验值增强）
        lab_loader = None
        try:
            from src.data.lab_loader import LabEventsLoader
            lab_loader = LabEventsLoader(data_root)
            if lab_loader._has_index:
                print(f"  [OK] Lab data loader enabled (indexed mode)")
            else:
                print(f"  [INFO] Lab data loader (full scan mode)")
        except Exception as e:
            print(f"  [WARN] Lab data loader: {e}")

        self.risk_scorer = RiskScorer(data_root, drug_se_map=drug_se_map, lab_loader=lab_loader)

        # 尝试加载LLM推理器和NER
        try:
            from src.decision.llm_reasoner import MedicalReasoner
            self.reasoner = MedicalReasoner(data_root)
            print("  [OK] LLM reasoning module loaded")
        except Exception as e:
            self.reasoner = None
            logger.warning(f"LLM推理模块加载失败: {e}")
            print(f"  [WARN] LLM reasoning module: {e}")

        try:
            from src.perception.ner.drug_ner import DrugNER
            self.ner = DrugNER(str(data_root / "sider" / "drug_names.tsv"))
            print("  [OK] NER drug recognition module loaded")
        except Exception as e:
            self.ner = None
            logger.warning(f"NER模块加载失败: {e}")
            print(f"  [WARN] NER module: {e}")

        # 尝试加载 GNN 模型和知识图谱（可选增强层）
        self.gnn_model = None
        self.kg = None
        self._gnn_drug_to_idx = {}
        self._gnn_total_nodes = 0
        self._try_load_gnn()

        # 预热：预加载常用数据缓存，避免首次请求阻塞
        self._warmup()

        print("沙盘推演引擎初始化完成！\n")

    def _warmup(self):
        """预加载关键数据缓存，避免首请求冷启动超时"""
        try:
            _ = self.recommender.get_drug_side_effects("aspirin")
            _ = self.risk_scorer.calculate_combination_risk(["CID100002244"], {"age":65})
            print("  [OK] Warmup complete")
        except Exception as e:
            print(f"  [WARN] Warmup partial: {e}")

    def _try_load_gnn(self):
        """尝试加载预训练 GNN 模型和知识图谱（失败不影响核心功能）。"""
        # 加载知识图谱
        kg_path = self.data_root / "gnn" / "knowledge_graph.pkl"
        if kg_path.exists():
            try:
                from src.decision.gnn.knowledge_graph import DrugKnowledgeGraph
                self.kg = DrugKnowledgeGraph.load(str(kg_path))
                print(f"  [OK] Knowledge graph loaded: {self.kg.G.number_of_nodes()} nodes, {self.kg.G.number_of_edges()} edges")
            except Exception as e:
                print(f"  [WARN] Knowledge graph loading failed: {e}")

        # 加载预训练 GNN 模型
        model_path = MODEL_ROOT / "best_model.pt"
        if not model_path.exists():
            model_path = Path("checkpoints/best_model.pt")
        if model_path.exists():
            try:
                import torch
                from src.decision.gnn.model import DDIPredictor
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                cfg = checkpoint.get('config', {})
                total_nodes = checkpoint.get('total_nodes', cfg.get('total_nodes', 8000))
                num_relations = checkpoint.get('num_relations',
                    len(checkpoint.get('edge_type_names', ['has_side_effect', 'has_indication', 'targets', 'interacts_with'])))

                self.gnn_model = DDIPredictor(
                    num_nodes=total_nodes,
                    num_relations=num_relations,
                    hidden_dim=cfg.get('hidden_dim', 128),
                    num_layers=cfg.get('num_layers', 2),
                    num_classes=2,
                    dropout=cfg.get('dropout', 0.3),
                )
                self.gnn_model.load_state_dict(checkpoint['model_state_dict'])
                self.gnn_model.eval()

                # 保存药物名→索引映射（用于GNN推理）
                self._gnn_drug_to_idx = checkpoint.get('drug_to_idx', {})
                self._gnn_total_nodes = total_nodes

                metrics = checkpoint.get('test_metrics', {})
                auc = metrics.get('auc', 'N/A')
                f1 = metrics.get('f1', 'N/A')
                print(f"  [OK] GNN model loaded: AUC={auc}, F1={f1}, drugs={len(self._gnn_drug_to_idx)}")

                # 预构建全图边索引（避免每次推理都O(|E|)遍历）
                self._prebuilt_edge_index = None
                self._prebuilt_edge_type = None
                if self.kg is not None and self.gnn_model is not None:
                    self._build_global_edge_index(total_nodes, num_relations)
            except Exception as e:
                self.gnn_model = None
                logger.warning(f"GNN模型加载失败: {e}")
                print(f"  [WARN] GNN model loading failed: {e}")
        else:
            print(f"  [INFO] GNN model not found, DDI detection using rule-based mode")

    def _build_global_edge_index(self, total_nodes, num_relations):
        """一次性预构建全图边索引张量（O(|E|)，仅初始化时执行一次）"""
        try:
            import torch
            edges, edge_types = [], []
            edge_type_map = {et: i for i, et in enumerate(self.kg.EDGE_TYPES)}
            for src, tgt, data in self.kg.G.edges(data=True):
                src_type, src_name = src.split(':', 1)
                tgt_type, tgt_name = tgt.split(':', 1)
                src_idx = self.kg._node_index.get(src_type, {}).get(src_name)
                tgt_idx = self.kg._node_index.get(tgt_type, {}).get(tgt_name)
                if src_idx is not None and tgt_idx is not None:
                    edges.append((src_idx, tgt_idx))
                    et = data.get('edge_type', 'unknown')
                    edge_types.append(edge_type_map.get(et, 0))
            if edges:
                self._prebuilt_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                self._prebuilt_edge_type = torch.tensor(edge_types, dtype=torch.long)
                print(f"  [OK] Global edge index pre-built: {len(edges)} edges, {num_relations} relations")
        except Exception as e:
            print(f"  [WARN] Edge index pre-build failed: {e}")

    def _build_drug_se_map(self):
        """通过 DataLoader 统一加载副作用数据，构建 CID → 副作用集合 映射。"""
        try:
            df = self.data_loader.load_side_effects()
            if df.empty:
                return defaultdict(set)
            # 过滤 NaN 副作用名
            df = df.dropna(subset=['side_effect_name'])
            se_map = defaultdict(set)
            for cid, group in df.groupby('cid'):
                names = [str(n) for n in group['side_effect_name'].tolist() if str(n) != 'nan']
                se_map[cid] = set(names)
            print(f"  [OK] Side effect data unified loading: {len(se_map)} drugs")
            return se_map
        except Exception as e:
            print(f"  [WARN] Side effect data loading failed: {e}")
            return defaultdict(set)
    
    def parse_natural_language(self, text):
        """
        解析自然语言输入
        '我吃了阿司匹林和华法林，有高血压和糖尿病' → drugs + conditions
        """
        if self.ner:
            result = self.ner.analyze(text)
            return {
                'drugs': result.get('drugs', []),
                'conditions': result.get('diseases', []),
                'dosages': [result.get('dosage', {})] if result.get('dosage') else [],
            }

        # 简单回退
        return {'drugs': [], 'conditions': [], 'dosages': []}

    def _drug_name_to_cid(self, drug_name: str) -> str:
        """将药物名转换为 SIDER CID（O(1) 字典查表）"""
        self.recommender._load_data()
        cid = self.recommender._name_to_cid.get(drug_name.lower())
        return str(cid) if cid else drug_name

    def _gnn_predict_ddi(self, drug1: str, drug2: str):
        """
        使用 GNN 模型预测两种药物的 DDI（增强层）。
        受 ENABLE_GNN_INFERENCE 全局开关控制。
        """
        if not ENABLE_GNN_INFERENCE:
            return None
        if self.gnn_model is None:
            return None
        if not hasattr(self, '_gnn_drug_to_idx') or not self._gnn_drug_to_idx:
            return None
        try:
            import torch

            d1_idx = self._gnn_drug_to_idx.get(drug1.lower())
            d2_idx = self._gnn_drug_to_idx.get(drug2.lower())
            if d1_idx is None or d2_idx is None:
                return None

            # 优先使用预构建的全图边索引（O(1)），否则回退到最小子图
            if self._prebuilt_edge_index is not None:
                edge_index = self._prebuilt_edge_index
                edge_type = self._prebuilt_edge_type
            elif self.kg is not None:
                # 无预构建索引时回退（首次启动未完成预热或构建失败）
                edges = [(d1_idx, d2_idx), (d2_idx, d1_idx)]
                edge_types_list = [3, 3]  # interacts_with
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_type = torch.tensor(edge_types_list, dtype=torch.long)
            else:
                edges = [(d1_idx, d2_idx), (d2_idx, d1_idx)]
                edge_types_list = [3, 3]
                if not edges:
                    return None
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_type = torch.tensor(edge_types_list, dtype=torch.long)

            result = self.gnn_model.predict(
                torch.tensor([d1_idx], dtype=torch.long),
                torch.tensor([d2_idx], dtype=torch.long),
                edge_index, edge_type
            )
            return {
                'prediction': result['predictions'].item(),
                'confidence': result['confidence'].item(),
                'probabilities': result['probabilities'].squeeze().tolist(),
            }
        except Exception as e:
            logger.warning(f"GNN预测失败: {e}")
            return None
    
    def full_simulation(self, drugs, patient_info=None, prescriptions=None):
        """
        完整沙盘推演
        
        Args:
            drugs: 药物列表
            patient_info: 患者信息 {age, gender, conditions}
            prescriptions: 处方列表
        
        Returns:
            完整推演结果
        """
        result = {
            'drugs': drugs,
            'input': {
                'drugs': drugs,
                'patient': patient_info,
                'prescriptions': prescriptions
            },
            'analysis': {}
        }
        
        # 1. DDI冲突检测（GNN增强 + 副作用比较 fallback）
        interactions = []
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                ddi_info = {
                    'drug1': drugs[i],
                    'drug2': drugs[j],
                }

                # 副作用数据（GNN 和 fallback 都需要）
                se_i = set(self.recommender.get_drug_side_effects(drugs[i]))
                se_j = set(self.recommender.get_drug_side_effects(drugs[j]))
                common = se_i & se_j
                ddi_info['common_se_count'] = len(common)
                ddi_info['common_effects'] = list(common)[:10]
                ddi_info['top_effects'] = list(common)[:8]

                # 副作用比较严重度（基准）
                se_severity = 'none'
                if common:
                    se_severity = 'high' if len(common) > 10 else ('moderate' if len(common) > 5 else 'mild')

                # GNN 预测（增强层）
                gnn_severity = 'none'
                gnn_result = self._gnn_predict_ddi(drugs[i], drugs[j])
                if gnn_result is not None:
                    ddi_info['method'] = 'gnn'
                    ddi_info['gnn_confidence'] = gnn_result['confidence']
                    ddi_info['gnn_prediction'] = gnn_result['prediction']
                    if gnn_result['prediction'] == 1:
                        conf = gnn_result['confidence']
                        gnn_severity = 'high' if conf > 0.92 else ('moderate' if conf > 0.85 else 'mild')
                else:
                    ddi_info['method'] = 'side_effect_comparison'

                # LLM 知识库检查 — 已知临床DDI为金标准
                llm_has_known_ddi = False
                llm_severity = 'none'
                if self.reasoner:
                    try:
                        ddi_record = self.reasoner._lookup_ddi(
                            self.reasoner.normalize_drug_name(drugs[i]),
                            self.reasoner.normalize_drug_name(drugs[j])
                        )
                        if ddi_record:
                            llm_has_known_ddi = True
                            rank = ddi_record['severity'].rank
                            llm_severity = 'high' if rank >= 3 else ('moderate' if rank >= 2 else 'mild')
                    except Exception:
                        pass

                # 综合判定：LLM知识库有记录→用LLM级别；无记录→降低SE误报
                if llm_has_known_ddi:
                    ddi_info['severity'] = llm_severity
                else:
                    # 无已知临床DDI时，SE重叠很可能是误报，降为轻微或忽略
                    if se_severity == 'high':
                        ddi_info['severity'] = 'moderate'
                    elif se_severity == 'moderate':
                        ddi_info['severity'] = 'mild'
                    else:
                        ddi_info['severity'] = 'none'
                if ddi_info.get('severity', 'none') != 'none':
                    # LLM解释
                    if self.reasoner:
                        try:
                            ddi_info['explanation'] = self.reasoner.explain_ddi(drugs[i], drugs[j])
                        except Exception as e:
                            logger.warning(f"DDI解释生成失败 ({drugs[i]}+{drugs[j]}): {e}")
                    interactions.append(ddi_info)

        result['analysis']['ddi'] = {
            'interactions': interactions,
            'count': len(interactions),
            'gnn_available': self.gnn_model is not None and ENABLE_GNN_INFERENCE,
        }

        # 用DDI严重度修正规则风险评分的乘数
        sev_weight = {'high': 1.0, 'moderate': 0.5, 'mild': 0.2, 'none': 0}
        ddi_risk = sum(sev_weight.get(i.get('severity','mild'), 0) * 25 for i in interactions)

        # 2. 风险评分 — 将药物名转换为 CID 后传入 RiskScorer
        drug_cids = []
        for drug in drugs:
            cid = self._drug_name_to_cid(drug)
            drug_cids.append(cid)
        risk_result = self.risk_scorer.calculate_combination_risk(
            drug_cids, patient_info
        )
        # 用实际DDI严重度修正分数：有已知DDI→保持原分；无已知DDI→大幅降权
        raw_score = risk_result.get('final_score', 0)
        if interactions:
            adjusted = min(100, raw_score * 0.3 + ddi_risk * 0.7)
        else:
            adjusted = min(100, raw_score * 0.3)
        risk_result['final_score'] = round(adjusted, 1)
        risk_result['raw_score'] = round(raw_score, 1)
        result['analysis']['risk'] = risk_result
        
        # 3. 替代药推荐 — 找到DDI严重度最高的"祸首"药物
        recommendations = []
        if risk_result.get('final_score', 0) >= 40 and interactions:
            # 从DDI中找出出现频率最高的"罪魁祸首"
            drug_hit = defaultdict(int)
            for ddi in interactions:
                sev_w = {'high': 3, 'moderate': 2, 'mild': 1}.get(ddi.get('severity','mild'), 0)
                drug_hit[ddi['drug1']] += sev_w
                drug_hit[ddi['drug2']] += sev_w
            worst_drug = max(drug_hit, key=drug_hit.get) if drug_hit else drugs[0]

            alts = self.recommender.recommend_alternatives(
                worst_drug, current_meds=None,
                patient_conditions=patient_info.get('conditions', []) if patient_info else [],
                top_k=3
            )
            if alts:
                recommendations.append({
                    'original': worst_drug,
                    'alternatives': alts
                    })
        result['analysis']['alternatives'] = recommendations
        
        # 4. 续方检查
        if prescriptions:
            refill_checks = self.prescription_trigger.batch_check(prescriptions)
            result['analysis']['prescriptions'] = refill_checks
        
        # 5. 综合建议
        try:
            final_score = risk_result.get('final_score', 0)
            result['analysis']['summary'] = {
                'total_interactions': len(interactions),
                'risk_level': risk_result.get('level', '未知'),
                'recommendation': '建议咨询医生' if final_score >= 60 else '可继续使用但需监测' if final_score >= 40 else '用药风险较低'
            }
        except Exception as e:
            logger.warning(f"综合建议生成失败: {e}")
        
        return result


if __name__ == "__main__":
    engine = SandboxEngine()
    
    # 测试自然语言解析
    if engine.ner:
        result = engine.parse_natural_language("我吃了阿司匹林和华法林，有高血压和糖尿病")
        print(f"NER结果: {result}")
    
    # 测试完整推演
    simulation = engine.full_simulation(
        drugs=["aspirin", "warfarin", "metformin"],
        patient_info={"age": 70, "gender": "M", "conditions": ["diabetes", "CKD"]}
    )
    print(f"\n推演结果: {simulation['analysis']['ddi']['count']} 个DDI冲突")
