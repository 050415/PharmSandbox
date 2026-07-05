"""
PharmSandbox - 替代药物推荐算法
基于ATC分类树 + 适应症匹配 + 指南约束的无害化替代药推荐
"""
import pandas as pd
import gzip
from pathlib import Path
from collections import defaultdict


class DrugRecommender:
    """替代药物推荐器"""
    
    def __init__(self, data_root=None):
        if data_root is None:
            from src.config import DATA_ROOT
            data_root = DATA_ROOT
        self.data_root = Path(data_root)
        self._atc_tree = None
        self._indications = None
        self._drug_names = None
        self._side_effects = None
        self._loaded = False
    
    def _load_data(self):
        """加载所有需要的数据"""
        if self._loaded:
            return

        try:
            # 加载ATC分类
            atc_path = self.data_root / "sider" / "drug_atc.tsv"
            if atc_path.exists():
                self._atc_tree = pd.read_csv(atc_path, sep='\t', header=None,
                                              names=['cid', 'drug_name', 'atc_code'])
            else:
                print(f"  [WARN] ATC data not found: {atc_path}")
                self._atc_tree = pd.DataFrame(columns=['cid', 'drug_name', 'atc_code'])

            # 加载适应症
            ind_path = self.data_root / "sider" / "meddra_all_indications.tsv.gz"
            if ind_path.exists():
                with gzip.open(ind_path, 'rt', encoding='utf-8') as f:
                    self._indications = pd.read_csv(f, sep='\t', header=None,
                                                    names=['cid', 'umls_cui_from', 'method',
                                                           'indication_name', 'umls_cui_to',
                                                           'mesh_id', 'max_phase', 'evidence_type'])
            else:
                self._indications = pd.DataFrame(columns=['cid', 'indication_name'])

            # 加载药物名称
            names_path = self.data_root / "sider" / "drug_names.tsv"
            if names_path.exists():
                self._drug_names = pd.read_csv(names_path, sep='\t', header=None,
                                               names=['cid', 'drug_name', 'side_effect_id', 'umls_id'])
            else:
                self._drug_names = pd.DataFrame(columns=['cid', 'drug_name'])

            # 加载副作用
            se_path = self.data_root / "sider" / "meddra_all_se.tsv.gz"
            if se_path.exists():
                with gzip.open(se_path, 'rt', encoding='utf-8') as f:
                    self._side_effects = pd.read_csv(f, sep='\t', header=None,
                                                     names=['cid', 'related_cid', 'umls_cui',
                                                            'method', 'umls_cui_to', 'side_effect_name'])
            else:
                self._side_effects = pd.DataFrame(columns=['cid', 'side_effect_name'])
        except Exception as e:
            print(f"  [WARN] Data loading failed: {e}")
            self._atc_tree = pd.DataFrame() if self._atc_tree is None else self._atc_tree
            self._indications = pd.DataFrame() if self._indications is None else self._indications
            self._drug_names = pd.DataFrame() if self._drug_names is None else self._drug_names
            self._side_effects = pd.DataFrame() if self._side_effects is None else self._side_effects

        self._loaded = True
        self._build_atc_index()
        self._build_side_effect_index()

    def _build_atc_index(self):
        """构建ATC分类索引（向量化），通过CID关联真实药名"""
        self._atc_groups = defaultdict(list)
        # 预建 CID -> drug_name 映射
        cid_to_name = {}
        if not self._drug_names.empty:
            for _, dr in self._drug_names.iterrows():
                cid_to_name[dr['cid']] = dr['drug_name']
        for _, row in self._atc_tree.iterrows():
            # drug_atc.tsv 格式: CID, ATC码 (无药名列, atc_code列为NaN)
            cid = row['cid']
            atc = str(row['atc_code']) if pd.notna(row['atc_code']) else str(row['drug_name'])
            drug_name = cid_to_name.get(cid, str(row['drug_name']))
            for level in range(1, min(6, len(atc) + 1)):
                prefix = atc[:level]
                self._atc_groups[prefix].append({
                    'cid': cid,
                    'drug_name': drug_name,
                    'atc_code': atc
                })

    def _build_side_effect_index(self):
        """预建药物名→副作用索引，避免每次调用都全表扫描。"""
        self._name_to_cid = {}
        self._cid_to_se = defaultdict(list)
        self._name_to_se = {}

        # drug_name → cid 映射
        if not self._drug_names.empty:
            for _, row in self._drug_names.iterrows():
                name_lower = str(row['drug_name']).lower()
                self._name_to_cid[name_lower] = row['cid']

        # cid → side_effects 映射（过滤 NaN）
        if not self._side_effects.empty:
            for cid, group in self._side_effects.groupby('cid'):
                names = group['side_effect_name'].dropna().unique().tolist()
                self._cid_to_se[cid] = [str(n) for n in names if isinstance(n, str)]

        # drug_name → side_effects 映射
        for name_lower, cid in self._name_to_cid.items():
            self._name_to_se[name_lower] = self._cid_to_se.get(cid, [])
    
    def get_drug_atc(self, drug_name):
        """获取药物的ATC分类"""
        self._load_data()
        name_lower = drug_name.lower()
        cid = self._name_to_cid.get(name_lower)
        if cid is None:
            return None
        matches = self._atc_tree[self._atc_tree['cid'] == cid]
        if len(matches) > 0:
            atc = matches.iloc[0]['atc_code']
            # drug_atc.tsv has ATC code in 'drug_name' column, 'atc_code' is NaN
            if pd.isna(atc):
                atc = str(matches.iloc[0]['drug_name'])
            return atc
        return None
    
    def get_same_class_drugs(self, drug_name, level=4):
        """获取同ATC分类的药物（同类药）"""
        self._load_data()
        atc_code = self.get_drug_atc(drug_name)
        if not atc_code:
            return []
        
        prefix = str(atc_code)[:level]
        candidates = self._atc_groups.get(prefix, [])
        
        # 排除自身
        return [d for d in candidates if d['drug_name'].lower() != drug_name.lower()]
    
    def get_drug_indications(self, drug_name):
        """获取药物的适应症列表"""
        self._load_data()
        name_lower = drug_name.lower()
        cid = self._name_to_cid.get(name_lower)
        if cid is None:
            return []
        indications = self._indications[self._indications['cid'] == cid]
        return indications['indication_name'].unique().tolist()
    
    def get_drug_side_effects(self, drug_name):
        """获取药物的副作用列表（使用预建索引）"""
        self._load_data()
        return self._name_to_se.get(drug_name.lower(), [])
    
    def recommend_alternatives(self, drug_name, current_meds=None, patient_conditions=None, top_k=5):
        """
        推荐替代药物
        
        Args:
            drug_name: 需要替换的药物名称
            current_meds: 当前正在使用的其他药物列表（避免DDI）
            patient_conditions: 患者当前诊断/适应症列表
            top_k: 返回前K个推荐
        
        Returns:
            推荐药物列表，每个包含药物名、匹配度评分、原因
        """
        self._load_data()
        
        # 1. 获取原药物的适应症
        drug_indications = self.get_drug_indications(drug_name)
        if patient_conditions:
            drug_indications = list(set(drug_indications + patient_conditions))
        
        # 2. 获取同类药物候选
        candidates = self.get_same_class_drugs(drug_name, level=5)
        if not candidates:
            # 退而求其次，用更粗的分类
            candidates = self.get_same_class_drugs(drug_name, level=4)
        if not candidates:
            candidates = self.get_same_class_drugs(drug_name, level=3)
        
        if not candidates:
            return []
        
        # 3. 对每个候选药物评分
        scored_candidates = []
        current_med_cids = set()
        if current_meds:
            for med in current_meds:
                name_matches = self._drug_names[
                    self._drug_names['drug_name'].str.lower() == med.lower()
                ]
                if len(name_matches) > 0:
                    current_med_cids.add(name_matches.iloc[0]['cid'])
        
        seen_names = set()
        for cand in candidates:
            cand_name = cand['drug_name']
            if cand_name.lower() in seen_names:
                continue
            seen_names.add(cand_name.lower())
            
            # 计算匹配分
            score = 0
            reasons = []
            
            # 适应症匹配
            cand_indications = self.get_drug_indications(cand_name)
            indication_overlap = set(drug_indications) & set(cand_indications)
            if indication_overlap:
                score += len(indication_overlap) * 20
                reasons.append(f"适应症匹配: {', '.join(list(indication_overlap)[:3])}")
            
            # 副作用对比（副作用越少越好）
            cand_se = set(self.get_drug_side_effects(cand_name))
            drug_se = set(self.get_drug_side_effects(drug_name))
            unique_se = cand_se - drug_se
            if len(unique_se) < 5:
                score += 15
                reasons.append("副作用较少")
            
            # 是否与当前用药冲突（简单检查是否有共同副作用）
            if current_med_cids:
                has_overlap_se = False
                for cid in current_med_cids:
                    other_se = set(self._side_effects[
                        self._side_effects['cid'] == cid
                    ]['side_effect_name'].unique())
                    if cand_se & other_se:
                        has_overlap_se = True
                        break
                if not has_overlap_se:
                    score += 25
                    reasons.append("与当前用药无共同副作用")
            
            if score > 0:
                scored_candidates.append({
                    'drug_name': cand_name,
                    'atc_code': cand['atc_code'],
                    'score': min(score, 100),
                    'indications': cand_indications[:5],
                    'reasons': reasons
                })
        
        # 按评分排序
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        return scored_candidates[:top_k]
    
    def explain_recommendation(self, drug_name, alternative_name):
        """生成替代推荐的解释"""
        self._load_data()
        
        drug_se = set(self.get_drug_side_effects(drug_name))
        alt_se = set(self.get_drug_side_effects(alternative_name))
        drug_ind = set(self.get_drug_indications(drug_name))
        alt_ind = set(self.get_drug_indications(alternative_name))
        
        explanation = {
            'original_drug': drug_name,
            'alternative': alternative_name,
            'shared_indications': list(drug_ind & alt_ind),
            'unique_side_effects_original': list(drug_se - alt_se)[:5],
            'unique_side_effects_alternative': list(alt_se - drug_se)[:5],
            'atc_original': self.get_drug_atc(drug_name),
            'atc_alternative': self.get_drug_atc(alternative_name)
        }
        return explanation


if __name__ == "__main__":
    rec = DrugRecommender()
    
    # 测试：查找阿司匹林的同类药
    print("=== 测试替代药推荐 ===")
    alternatives = rec.recommend_alternatives("aspirin", top_k=5)
    for alt in alternatives:
        print(f"\n推荐: {alt['drug_name']} (评分: {alt['score']})")
        print(f"  ATC: {alt['atc_code']}")
        print(f"  原因: {'; '.join(alt['reasons'])}")
