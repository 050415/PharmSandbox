"""
PharmSandbox - 替代药物推荐算法
基于ATC分类树 + 适应症匹配 + 指南约束的无害化替代药推荐
"""
import pandas as pd
import gzip
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class DrugRecommender:
    """替代药物推荐器"""

    SYNONYM_MAP: Dict[str, str] = {
        'levothyroxine': 'thyroxine', 'levothyroxine sodium': 'thyroxine',
        'l-thyroxine': 'thyroxine', 'albuterol': 'salbutamol',
        'adrenaline': 'epinephrine', 'noradrenaline': 'norepinephrine',
        'paracetamol': 'acetaminophen', 'ergocalciferol': 'vitamin d',
        'cholecalciferol': 'vitamin d', 'salbutamol sulfate': 'salbutamol',
        'lignocaine': 'lidocaine', 'pethidine': 'meperidine',
        'frusemide': 'furosemide', 'vitamin b12': 'cyanocobalamin',
        'vitamin b6': 'pyridoxine', 'vitamin b1': 'thiamine',
        'trans': 'streptokinase',  # SIDER 截断数据修复：B01AD01 链激酶
    }

    ATC_OVERRIDES: Dict[str, str] = {
        'aspirin': 'B01AC06', 'ibuprofen': 'M01AE01', 'naproxen': 'M01AE02',
        'diclofenac': 'M01AB05', 'indomethacin': 'M01AB01', 'ketorolac': 'M01AB15',
        'celecoxib': 'M01AH01', 'acetaminophen': 'N02BE01', 'paracetamol': 'N02BE01',
        'metformin': 'A10BA02', 'glibenclamide': 'A10BB01', 'gliclazide': 'A10BB09',
        'insulin': 'A10AD01',
    }

    ATC_CLASS_DDI_RULES: Dict[Tuple[str, str], Tuple[str, str]] = {
        ('B01AA','B01AC'):('high','ATC类DDI:VitK拮抗剂+抗血小板→严重出血'),
        ('B01AB','B01AC'):('high','ATC类DDI:肝素类+抗血小板→出血'),
        ('B01AE','B01AC'):('high','ATC类DDI:凝血酶抑制剂+抗血小板→出血'),
        ('B01AF','B01AC'):('high','ATC类DDI:Xa因子抑制剂+抗血小板→出血'),
        ('B01AA','M01A'):('high','ATC类DDI:VitK拮抗剂+NSAIDs→消化道出血'),
        ('B01AB','M01A'):('high','ATC类DDI:肝素类+NSAIDs→出血'),
        ('B01AE','M01A'):('high','ATC类DDI:凝血酶抑制剂+NSAIDs→出血'),
        ('B01AF','M01A'):('high','ATC类DDI:Xa因子抑制剂+NSAIDs→出血'),
        ('B01AC','M01A'):('moderate','ATC类DDI:抗血小板+NSAIDs→消化道损伤'),
        ('C09AA','M01A'):('moderate','ATC类DDI:ACEI+NSAIDs→肾损伤'),
        ('C09CA','M01A'):('moderate','ATC类DDI:ARB+NSAIDs→肾损伤'),
        ('C09AA','C03DA'):('high','ATC类DDI:ACEI+保钾利尿剂→高钾血症'),
        ('C09CA','C03DA'):('high','ATC类DDI:ARB+保钾利尿剂→高钾血症'),
        ('B01AA','B01AB'):('high','ATC类DDI:双重抗凝→严重出血'),
        ('B01AA','B01AE'):('high','ATC类DDI:双重抗凝→严重出血'),
        ('B01AA','B01AF'):('high','ATC类DDI:双重抗凝→严重出血'),
        ('C10AA','J02AC'):('moderate','ATC类DDI:他汀+唑类抗真菌→横纹肌溶解'),
        ('A10BA','V08A'):('high','ATC类DDI:二甲双胍+碘造影剂→乳酸酸中毒'),
    }

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
        """获取药物的ATC分类（含已知错误修正）"""
        self._load_data()
        name_lower = drug_name.lower()
        override = self.ATC_OVERRIDES.get(name_lower)
        if override:
            return override
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
        """获取药物的副作用列表（含同义词回退）"""
        self._load_data()
        name_lower = drug_name.lower()
        se = self._name_to_se.get(name_lower)
        if se:
            return se
        synonym = self.SYNONYM_MAP.get(name_lower)
        if synonym:
            return self._name_to_se.get(synonym, [])
        return []
    
    def recommend_alternatives(self, drug_name, current_meds=None, patient_conditions=None,
                               top_k=5, lab_values=None, is_unconscious=False,
                               original_ddi_severity=None):
        """推荐替代药物（v2.0 - 安全熔断+净收益评分+自审计）"""
        from src.decision.recommender.safety_rules import SafetyRuleEngine
        safety = SafetyRuleEngine()
        self._load_data()

        # Preload reasoner
        _reasoner = None
        try:
            from src.decision.llm_reasoner import MedicalReasoner
            _reasoner = MedicalReasoner()
        except Exception: pass

        drug_indications = self.get_drug_indications(drug_name)
        if patient_conditions:
            drug_indications = list(set(drug_indications + patient_conditions))
        drug_se = set(self.get_drug_side_effects(drug_name))

        # Merge candidates from level 5+4+3
        l5, l4, l3 = self.get_same_class_drugs(drug_name, level=5), self.get_same_class_drugs(drug_name, level=4), self.get_same_class_drugs(drug_name, level=3)
        seen_n = set()
        candidates = []
        for c in (l5 + l4 + l3):
            k = c['drug_name'].lower()
            if k not in seen_n:
                seen_n.add(k)
                candidates.append(c)
        if not candidates: return []

        current_med_names = set((m or '').lower() for m in (current_meds or []) if m)

        scored = []
        blocked = []
        for cand in candidates:
            cname = cand['drug_name']
            clower = cname.lower()

            # Safety check
            sr = safety.check_drug(cname, patient_conditions or [], lab_values, is_oral=True, is_unconscious=is_unconscious)
            br = [r.reason for r in sr if r.verdict.value == 'blocked']
            warnings = [r.reason for r in sr if r.verdict.value == 'warning']
            if br:
                blocked.append({'drug_name': cname, 'atc_code': cand['atc_code'], 'blocked_reason': br[0], 'tier': 'blocked', 'score': 0, 'indications': []})
                continue

            score = 0; pos = []; neg = []
            ci = self.get_drug_indications(cname)
            cse = set(self.get_drug_side_effects(cname))
            catc = cand.get('atc_code', '')

            # (A) Efficacy
            overlap = set(drug_indications) & set(ci)
            if overlap: e = min(len(overlap)*15, 50); score += e; pos.append(f"适应症匹配({len(overlap)}项): {', '.join(list(overlap)[:3])}")
            elif ci: score += 5; neg.append("适应症不完全匹配")

            # (B) DDI check
            ddi_cnt = 0; ddi_details = []
            if current_med_names:
                for mn in current_med_names:
                    found = False
                    try:
                        if _reasoner is None:
                            from src.decision.llm_reasoner import MedicalReasoner
                            _reasoner = MedicalReasoner()
                        ddi = _reasoner._lookup_ddi(_reasoner.normalize_drug_name(clower), _reasoner.normalize_drug_name(mn))
                        if ddi: found = True; ddi_cnt += 1; rk = ddi['severity'].rank; ddi_details.append(f"与{mn}存在{ddi['severity'].label}级DDI(直接)"); score -= 20*rk
                    except: pass
                    if not found and catc:
                        ma = self.get_drug_atc(mn)
                        cr = self._check_atc_class_ddi(catc, mn, ma or '')
                        if cr: found = True; ddi_cnt += 1; sev, reason = cr; rk = 3 if sev=='high' else (2 if sev=='moderate' else 1); ddi_details.append(f"与{mn}存在{sev}级DDI(ATC类级)"); score -= 20*rk
                    if not found:
                        try:
                            os = set(self.get_drug_side_effects(mn))
                            if cse & os: ddi_cnt += 1; score -= 10; ddi_details.append(f"与{mn}有共同副作用(间接DDI)")
                        except: pass
            if ddi_cnt > 0: neg.append(f"潜在DDI冲突({ddi_cnt}项): {'; '.join(ddi_details)}")

            # (C) New SE penalty
            new_se = cse - drug_se
            severe_kw = ['death','fatal','life threatening','hospitalisation','cardiac','renal','hepatic','arrhythmia','seizure','hemorrhage','respiratory failure']
            severe_new = [s for s in new_se if any(k in s.lower() for k in severe_kw)]
            if severe_new: score -= len(severe_new)*5; neg.append(f"新增高危副作用({len(severe_new)}项): {', '.join(severe_new[:3])}")
            elif len(new_se) > 10: score -= 10; neg.append(f"新副作用较多({len(new_se)}项)")

            # (D) Warning penalty
            if warnings: score -= len(warnings)*10; neg.append("; ".join(warnings))

            # (E) SE improvement
            removed_se = drug_se - cse
            if len(removed_se) > len(new_se): score += 15; pos.append(f"规避原发风险：预期减少{len(removed_se)}项不良反应（新增{len(new_se)}项）")

            tier = self._classify_tier(score, ddi_cnt, severe_new, warnings)
            tl = {'preferred':'优先推荐','safe':'安全备选','caution':'谨慎降级','not_recommended':'不推荐'}[tier]
            scored.append({'drug_name': cname, 'atc_code': catc, 'score': score, 'tier': tier, 'tier_label': tl,
                'indications': ci[:5], 'indication_overlap': list(overlap)[:5],
                'reasons_positive': pos, 'reasons_negative': neg, 'warnings': warnings,
                'new_side_effects': list(new_se)[:8], 'removed_side_effects': list(removed_se)[:5], 'ddi_conflicts': ddi_details})

        # Sort
        to = {'preferred':0,'safe':1,'caution':2,'not_recommended':3}
        scored.sort(key=lambda x: (to.get(x['tier'],4), -x['score']))

        # Self-audit
        try:
            from src.decision.recommender.auditor import RecommendationAuditor
            auditor = RecommendationAuditor()
            orig_atc = self.get_drug_atc(drug_name) or ''
            scored, reports = auditor.audit(scored, drug_name, orig_atc, original_ddi_severity or [],
                current_meds or [], patient_conditions or [], lab_values,
                _preloaded_reasoner=_reasoner, _preloaded_se_getter=self.get_drug_side_effects)
            for report in reports:
                if report.adjusted_tier == 'blocked' and not report.passed:
                    blocked.append({'drug_name': report.drug_name, 'atc_code': '',
                        'blocked_reason': '; '.join([f.message for f in report.findings if f.severity == 'critical']) or '自审计熔断',
                        'all_blocked_reasons': [f.message for f in report.findings],
                        'indications': [], 'tier': 'blocked', 'score': report.adjusted_score,
                        'audit_findings': [{'check': f.check_name, 'severity': f.severity, 'msg': f.message} for f in report.findings]})
        except Exception: pass

        # ---- 7. 候选集清洗流水线（去重 + ROA + 适应症严格过滤） ----
        # IV-only 药物黑名单（这些药只有静脉注射剂型，门诊不可替代口服药）
        IV_ONLY_DRUGS = {
            'tirofiban', 'eptifibatide', 'argatroban', 'bivalirudin',
            'lepirudin', 'urokinase', 'streptokinase', 'alteplase',
            'tenecteplase', 'reteplase', 'abciximab', 'trans',
        }

        # 数据质量过滤：药名异常检查
        def _is_junk_name(name):
            if not name or len(name) <= 3: return True
            if name == 'trans': return True  # 已知数据截断bug
            if name[0].isdigit(): return True  # 纯数字名
            junk = {'test', 'none', 'unknown', 'n/a', 'drug', 'medication'}
            if name.lower() in junk: return True
            return False

        def _pipeline_filter(candidates_list, drug_name, current_meds_list, drug_indications_set):
            """后处理过滤器：在返回前端之前做最后一轮清洗"""
            cleaned = []
            for c in candidates_list:
                cname = c.get('drug_name', '')

                # ① 数据质量：过滤脏数据
                if _is_junk_name(cname):
                    continue

                # ② 去重器：不能推荐已经在用的药
                if current_meds_list and cname.lower() in [m.lower() for m in current_meds_list]:
                    continue

                # ③ 不给推荐自己
                if cname.lower() == drug_name.lower():
                    continue

                # ④ ROA 硬熔断：IV-only 药物不能替代口服药
                if cname.lower() in IV_ONLY_DRUGS:
                    c['tier'] = 'blocked'
                    c['tier_label'] = '已熔断'
                    c['score'] = -100
                    c['blocked_reason'] = f'给药途径硬熔断: {cname}仅静脉注射(IV)，不可替代口服药'
                    cleaned.append(c)  # 保留但标记为blocked（灰显）
                    continue

                # ⑤ 适应症严格过滤
                cinds = set(self.get_drug_indications(cname))
                if drug_indications_set and not (drug_indications_set & cinds):
                    if c.get('tier') == 'preferred':
                        c['tier'] = 'caution'
                        c['tier_label'] = '谨慎降级'
                        if not c.get('reasons_negative'):
                            c['reasons_negative'] = []
                        c['reasons_negative'].append('适应症与目标不完全匹配')

                cleaned.append(c)
            return cleaned

        scored = _pipeline_filter(scored, drug_name, current_meds or [],
                                   set(drug_indications) if drug_indications else set())

        # 方案先行：优先展示可用方案，熔断药物置底
        result = scored[:top_k]
        # 如果可用方案不足，扩大搜索（level 2 ATC）
        if len(result) == 0:
            l2 = self.get_same_class_drugs(drug_name, level=2)
            for c in l2:
                if c['drug_name'].lower() not in [s['drug_name'].lower() for s in result]:
                    result.append({'drug_name': c['drug_name'], 'atc_code': c.get('atc_code',''),
                        'tier': 'caution', 'tier_label': '谨慎降级', 'score': 5,
                        'indications': [], 'reasons_positive': ['跨类候选(扩大搜索)'],
                        'reasons_negative': ['ATC分类较远，需临床确认适用性'], 'warnings': [], 'ddi_conflicts': []})
            result = result[:top_k]
        # 无论如何至少给一个建议
        if len(result) == 0:
            result.append({'drug_name': '需临床综合评估', 'atc_code': '',
                'tier': 'caution', 'tier_label': '需会诊决策', 'score': 0,
                'indications': [], 'reasons_positive': ['无可直接替换的同类药物'],
                'reasons_negative': ['建议多学科会诊，考虑非药物干预或调整联合方案'], 'warnings': [], 'ddi_conflicts': []})
        # 熔断列表追加到末尾
        result.extend(blocked[:3])
        return result

    def _check_atc_class_ddi(self, cand_atc, med_name, med_atc):
        if not cand_atc or not med_atc: return None
        for level in [5, 4, 3]:
            cp = cand_atc[:level] if len(cand_atc) >= level else cand_atc
            mp = med_atc[:level] if len(med_atc) >= level else med_atc
            for (a, b), (severity, reason) in self.ATC_CLASS_DDI_RULES.items():
                if (cp == a and mp == b) or (cp == b and mp == a):
                    return (severity, f"{reason} (药理类: {cp} ↔ {mp})")
        return None

    @staticmethod
    def _classify_tier(score, ddi_conflicts, severe_new_se, warnings):
        if ddi_conflicts >= 3 or len(severe_new_se) >= 4: return 'not_recommended'
        if score >= 40 and ddi_conflicts == 0: return 'preferred'
        if score >= 20 and ddi_conflicts <= 1: return 'safe'
        if score >= 0: return 'caution'
        # 负分但有可用候选 → 仍标注为"谨慎降级"让医生知情决策
        return 'caution'
    
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
