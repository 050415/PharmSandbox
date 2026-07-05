"""
PharmSandbox - 0-100 风险量化评分系统
结合SIDER副作用概率 + MIMIC-IV患者约束的动态风险评分
"""
import pandas as pd
import gzip
from pathlib import Path
from collections import defaultdict
from src.config import DATA_ROOT


class RiskScorer:
    """患者特异性用药风险评分器 (0-100)"""
    
    # 副作用严重度权重
    SEVERITY_WEIGHTS = {
        'death': 100, 'fatal': 100, 'life threatening': 95,
        'hospitalisation': 80, 'hospitalization': 80, 'disability': 75,
        'congenital': 70, 'required intervention': 60,
        'serious': 50, 'severe': 40, 'moderate': 25, 'mild': 10,
    }
    
    # 器官系统风险关键词
    ORGAN_KEYWORDS = {
        'renal': ['renal', 'kidney', 'nephro', 'creatinine', 'bun', 'uria'],
        'hepatic': ['hepatic', 'liver', 'hepat', 'bilirubin', 'alt', 'ast', 'transaminase'],
        'cardiac': ['cardiac', 'heart', 'arrhythmia', 'qt', 'tachycardia', 'bradycardia'],
        'hematologic': ['bleeding', 'hemorrhag', 'thrombo', 'coagul', 'platelet', 'anemia'],
        'respiratory': ['respiratory', 'pulmonary', 'broncho', 'pneumon', 'dyspnea'],
        'neurologic': ['seizure', 'convulsion', 'neuropath', 'tremor', 'dizziness'],
    }
    
    def __init__(self, data_root=None, drug_se_map=None, lab_loader=None):
        self.data_root = Path(data_root) if data_root else DATA_ROOT
        self._loaded = drug_se_map is not None  # 如果注入了数据，标记为已加载
        self._drug_se_map = drug_se_map if drug_se_map is not None else defaultdict(set)
        self._drug_freq_map = {}
        self._lab_loader = lab_loader  # 可选：化验数据加载器
        self._icd_keywords = {}
        self._init_icd_keywords()
    
    def _init_icd_keywords(self):
        """初始化ICD编码到器官系统的映射"""
        self._icd_keywords = {
            'renal': ['585', '586', '593', 'N17', 'N18', 'N19'],
            'hepatic': ['571', '572', 'K70', 'K72', 'K73', 'K74'],
            'cardiac': ['410', '411', '412', '413', '414', 'I20', 'I21', 'I22', 'I25'],
            'diabetes': ['250', 'E10', 'E11'],
            'hypertension': ['401', '402', '403', 'I10', 'I11', 'I12'],
        }
    
    def _load_data(self):
        """加载SIDER副作用数据"""
        if self._loaded:
            return

        # 加载副作用
        se_path = self.data_root / "sider" / "meddra_all_se.tsv.gz"
        try:
            with gzip.open(se_path, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f, sep='\t', header=None, nrows=500000,
                                 names=['cid', 'related_cid', 'umls_cui',
                                        'method', 'umls_cui_to', 'side_effect_name'])
            df = df.dropna(subset=['side_effect_name'])
            for cid, group in df.groupby('cid'):
                names = [str(n) for n in group['side_effect_name'].tolist() if str(n) != 'nan']
                self._drug_se_map[cid] = set(names)
            self._loaded = True  # 仅在成功加载后标记
        except Exception as e:
            print(f"SIDER加载失败: {e}")
            # 加载失败不设置 _loaded，下次调用会重试
    
    def _classify_se_organ(self, se_name):
        """将副作用分类到器官系统"""
        if not isinstance(se_name, str) or not se_name:
            return ['general']
        se_lower = se_name.lower()
        organs = []
        for organ, keywords in self.ORGAN_KEYWORDS.items():
            if any(kw in se_lower for kw in keywords):
                organs.append(organ)
        return organs if organs else ['general']
    
    def _get_severity_score(self, se_name):
        """获取副作用严重度评分"""
        if not isinstance(se_name, str) or not se_name:
            return 15  # 默认中等
        se_lower = se_name.lower()
        for keyword, score in self.SEVERITY_WEIGHTS.items():
            if keyword in se_lower:
                return score
        return 15  # 默认中等
    
    def _get_patient_risk_factors(self, patient_info):
        """
        计算患者特异性风险因子
        返回各器官系统的风险倍率
        """
        risk_factors = {
            'renal': 1.0, 'hepatic': 1.0, 'cardiac': 1.0,
            'hematologic': 1.0, 'respiratory': 1.0, 'neurologic': 1.0
        }
        
        age = patient_info.get('age', 50)
        conditions = [c.lower() for c in patient_info.get('conditions', [])]
        
        # 年龄因子
        if age > 80:
            for k in risk_factors:
                risk_factors[k] *= 1.8
        elif age > 65:
            for k in risk_factors:
                risk_factors[k] *= 1.4
        elif age < 18:
            risk_factors['neurologic'] *= 1.3
        
        # 疾病史因子
        condition_mapping = {
            'ckd': 'renal', 'renal': 'renal', 'kidney': 'renal', '肾病': 'renal', '肾功能不全': 'renal',
            'liver': 'hepatic', 'hepatic': 'hepatic', '肝病': 'hepatic', '肝功能不全': 'hepatic',
            'chf': 'cardiac', 'heart': 'cardiac', '心衰': 'cardiac', '冠心病': 'cardiac',
            'diabetes': 'renal', '糖尿病': 'renal',
            'copd': 'respiratory', 'asthma': 'respiratory', '哮喘': 'respiratory',
        }
        
        for cond in conditions:
            for keyword, organ in condition_mapping.items():
                if keyword in cond:
                    risk_factors[organ] *= 1.8
                    break

        # 如果有化验数据加载器且提供了 subject_id，使用实际化验值调整
        subject_id = patient_info.get('subject_id')
        if self._lab_loader and subject_id is not None:
            lab_factors = self._get_lab_risk_factors(subject_id)
            for organ, multiplier in lab_factors.items():
                risk_factors[organ] *= multiplier

        return risk_factors

    def _get_lab_risk_factors(self, subject_id):
        """根据实际化验值计算器官风险倍率。"""
        if self._lab_loader is None:
            return {}
        try:
            return self._lab_loader.get_risk_factors(subject_id)
        except Exception:
            return {}
    
    def calculate_drug_risk(self, drug_cid, patient_info=None):
        """
        计算单药风险评分
        
        Args:
            drug_cid: 药物CID
            patient_info: 患者信息 {age, gender, conditions}
        
        Returns:
            风险评分 (0-100) 及详细信息
        """
        self._load_data()
        
        ses = self._drug_se_map.get(drug_cid, set())
        if not ses:
            return {'score': 10, 'level': '安全', 'detail': '未找到副作用记录'}
        
        risk_factors = self._get_patient_risk_factors(patient_info or {})
        
        organ_scores = defaultdict(float)
        total_severity = 0
        
        for se in ses:
            severity = self._get_severity_score(se)
            organs = self._classify_se_organ(se)
            
            for organ in organs:
                adjusted = severity * risk_factors.get(organ, 1.0)
                organ_scores[organ] = max(organ_scores[organ], adjusted * 0.5)
                organ_scores[organ] += adjusted * 0.1
            
            total_severity += severity
        
        # 归一化到0-100
        base_score = min(40, total_severity / max(len(ses), 1) * 0.3)
        organ_bonus = sum(organ_scores.values()) * 0.3
        score = min(100, base_score + organ_bonus)
        
        # 风险等级
        if score >= 80: level = "极度高危"
        elif score >= 60: level = "高危"
        elif score >= 40: level = "中等风险"
        elif score >= 20: level = "低风险"
        else: level = "安全"
        
        return {
            'score': round(score, 1),
            'level': level,
            'side_effect_count': len(ses),
            'organ_risks': {k: round(v, 1) for k, v in organ_scores.items()},
            'top_risk_organ': max(organ_scores, key=organ_scores.get) if organ_scores else 'none'
        }
    
    def calculate_combination_risk(self, drug_cids, patient_info=None):
        """
        计算多药联合风险评分
        
        Args:
            drug_cids: 药物CID列表
            patient_info: 患者信息
        
        Returns:
            综合风险评分及各药详细评分
        """
        self._load_data()
        
        individual_risks = []
        all_se = set()
        
        for cid in drug_cids:
            risk = self.calculate_drug_risk(cid, patient_info)
            risk['drug_cid'] = cid
            individual_risks.append(risk)
            all_se.update(self._drug_se_map.get(cid, set()))
        
        # 多药联合效应
        common_se = set()
        for i in range(len(drug_cids)):
            for j in range(i+1, len(drug_cids)):
                se_i = self._drug_se_map.get(drug_cids[i], set())
                se_j = self._drug_se_map.get(drug_cids[j], set())
                common_se.update(se_i & se_j)
        
        # 基础综合分 = 各药平均分
        avg_score = sum(r['score'] for r in individual_risks) / max(len(individual_risks), 1)
        
        # 联合风险加成
        interaction_bonus = min(25, len(common_se) * 2)
        
        # 最终评分
        final_score = min(100, avg_score + interaction_bonus)
        
        if final_score >= 80: level = "极度高危"
        elif final_score >= 60: level = "高危"
        elif final_score >= 40: level = "中等风险"
        elif final_score >= 20: level = "低风险"
        else: level = "安全"
        
        return {
            'final_score': round(final_score, 1),
            'level': level,
            'individual_risks': individual_risks,
            'common_side_effects': [s for s in common_se if isinstance(s, str) and s != 'nan'][:10],
            'interaction_bonus': round(interaction_bonus, 1),
            'total_side_effects': len(all_se)
        }


if __name__ == "__main__":
    scorer = RiskScorer()
    patient = {'age': 70, 'gender': 'M', 'conditions': ['CKD', 'diabetes']}
    
    print("=== 风险评分测试 ===")
    # 测试单药风险
    risk = scorer.calculate_drug_risk('CID0000002787', patient)
    print(f"单药风险: {risk}")
