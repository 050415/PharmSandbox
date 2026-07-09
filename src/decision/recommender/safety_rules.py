# -*- coding: utf-8 -*-
"""
PharmSandbox - 基于医学指南的规则引擎（Rule-based Safety Engine）
双层推理架构：概率模型之上强制执行禁忌症熔断
"""
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SafetyVerdict(Enum):
    BLOCKED = "blocked"
    WARNING = "warning"
    CLEAR = "clear"


class SafetyCheckResult:
    def __init__(self, drug_name, verdict, rule_name, reason, severity="absolute"):
        self.drug_name = drug_name
        self.verdict = verdict
        self.rule_name = rule_name
        self.reason = reason
        self.severity = severity

    def to_dict(self):
        return {"drug": self.drug_name, "verdict": self.verdict.value, "rule": self.rule_name, "reason": self.reason}


# 疾病实体归一化
CONDITION_ALIASES = {
    "高血糖": "diabetes_mellitus", "糖尿病": "diabetes_mellitus", "2型糖尿病": "diabetes_mellitus",
    "肾衰竭": "renal_failure", "肾衰": "renal_failure", "肾功能不全": "renal_impairment",
    "CKD": "ckd", "慢性肾病": "ckd", "ckd 3期": "ckd_stage3", "ckd 4期": "ckd_stage4", "ckd 5期": "ckd_stage5",
    "终末期肾病": "esrd", "ESRD": "esrd", "透析": "dialysis", "血液透析": "dialysis",
    "肝衰竭": "liver_failure", "肝功能不全": "liver_impairment", "肝硬化": "cirrhosis",
    "心衰": "heart_failure", "心力衰竭": "heart_failure",
    "高血压": "hypertension", "心房颤动": "atrial_fibrillation", "房颤": "atrial_fibrillation",
    "昏迷": "unconscious", "妊娠": "pregnancy", "怀孕": "pregnancy",
    "哮喘": "asthma", "慢阻肺": "copd", "COPD": "copd",
    "低血糖": "hypoglycemia", "低钾": "hypokalemia", "高钾": "hyperkalemia",
}


def normalize_condition(condition: str) -> str:
    return CONDITION_ALIASES.get(condition.lower(), condition.lower().replace(" ", "_"))


# 绝对禁忌
ABSOLUTE_CONTRAINDICATIONS = [
    ("renal_failure", "metformin", "肾衰竭患者使用二甲双胍有乳酸酸中毒风险"),
    ("esrd", "metformin", "终末期肾病患者禁用二甲双胍"),
    ("dialysis", "metformin", "透析患者禁用二甲双胍"),
    ("ckd_stage4", "metformin", "CKD 4期禁用二甲双胍(eGFR<30)"),
    ("renal_failure", "nsaids", "肾衰竭患者禁用所有NSAIDs"),
    ("renal_impairment", "nsaids", "肾功能不全患者应避免所有NSAIDs"),
    ("ckd", "nsaids", "CKD患者使用NSAIDs加速肾功能恶化"),
    ("renal_failure", "gentamicin", "肾衰竭患者使用氨基糖苷类抗生素有肾毒性风险"),
    ("renal_failure", "spironolactone", "肾衰竭患者使用螺内酯有高钾血症风险"),
    ("liver_failure", "acetaminophen", "肝衰竭患者禁用对乙酰氨基酚"),
    ("liver_failure", "methotrexate", "肝衰竭患者禁用甲氨蝶呤"),
    ("cirrhosis", "acetaminophen", "肝硬化患者慎用对乙酰氨基酚"),
    ("heart_failure", "nsaids", "心衰患者使用NSAIDs可导致水钠潴留加重心衰"),
    ("pregnancy", "warfarin", "华法林有致畸性，妊娠期禁用"),
    ("pregnancy", "methotrexate", "甲氨蝶呤有致畸性，妊娠期禁用"),
    ("pregnancy", "lisinopril", "ACEI在妊娠期禁用，可导致胎儿肾发育异常"),
    ("pregnancy", "enalapril", "ACEI在妊娠期禁用"),
    ("pregnancy", "valsartan", "ARB在妊娠期禁用"),
    ("pregnancy", "losartan", "ARB在妊娠期禁用"),
    ("asthma", "aspirin", "阿司匹林可诱发哮喘发作"),
    ("asthma", "nsaids", "NSAIDs可诱发哮喘发作"),
    ("asthma", "propranolol", "非选择性β阻滞剂可诱发支气管痉挛"),
    ("hypoglycemia", "insulin", "低血糖状态下使用胰岛素可导致危象"),
    ("hyperkalemia", "spironolactone", "高钾血症患者禁用保钾利尿剂"),
    ("hyperkalemia", "potassium chloride", "高钾血症患者禁用补钾制剂"),
    ("unconscious", "*oral*", "昏迷患者不能口服给药，需改用静脉/鼻饲途径"),
]

RELATIVE_CONTRAINDICATIONS = [
    ("renal_impairment", "metformin", "肾功能不全患者使用二甲双胍需减量"),
    ("ckd_stage3", "metformin", "CKD 3期使用二甲双胍需减量并密切监测"),
    ("liver_impairment", "statins", "肝功能异常患者使用他汀需监测肝酶"),
    ("pregnancy", "aspirin", "妊娠期使用阿司匹林需评估获益风险比"),
    ("hypokalemia", "furosemide", "低钾血症患者使用呋塞米可加重低钾"),
]


class SafetyRuleEngine:
    def __init__(self):
        self._absolute_index: Dict[str, List[Tuple[str, str]]] = {}
        self._relative_index: Dict[str, List[Tuple[str, str]]] = {}
        for c, d, r in ABSOLUTE_CONTRAINDICATIONS:
            self._absolute_index.setdefault(c, []).append((d, r))
        for c, d, r in RELATIVE_CONTRAINDICATIONS:
            self._relative_index.setdefault(c, []).append((d, r))

    def check_drug(self, drug_name, patient_conditions=None, lab_values=None, is_oral=True, is_unconscious=False):
        results = []
        drug_lower = drug_name.lower()
        norm_conditions = set()
        for cond in (patient_conditions or []):
            nc = normalize_condition(cond)
            norm_conditions.add(nc)
            norm_conditions.add(cond.lower().replace(" ", "_"))

        def _lv(key, default):
            v = lab_values.get(key) if lab_values else None
            return v if v is not None else default

        if lab_values:
            if _lv("creatinine", 0) > 200: norm_conditions.add("renal_failure")
            if _lv("creatinine", 0) > 130: norm_conditions.add("renal_impairment")
            if _lv("alt", 0) > 200: norm_conditions.add("liver_failure")
            if _lv("alt", 0) > 80: norm_conditions.add("liver_impairment")
            if _lv("potassium", 4.0) > 5.5: norm_conditions.add("hyperkalemia")
            if _lv("potassium", 4.0) < 3.0: norm_conditions.add("hypokalemia")
            if _lv("inr", 1.0) > 3.5: norm_conditions.add("bleeding_risk")

        if is_unconscious: norm_conditions.add("unconscious")

        for condition in norm_conditions:
            for drug_pattern, reason in self._absolute_index.get(condition, []):
                if self._drug_matches(drug_pattern, drug_lower):
                    results.append(SafetyCheckResult(drug_name, SafetyVerdict.BLOCKED, f"绝对禁忌:{condition}", reason))

        for condition in norm_conditions:
            for drug_pattern, reason in self._relative_index.get(condition, []):
                if self._drug_matches(drug_pattern, drug_lower):
                    results.append(SafetyCheckResult(drug_name, SafetyVerdict.WARNING, f"相对禁忌:{condition}", reason, "relative"))

        if is_unconscious and is_oral:
            results.append(SafetyCheckResult(drug_name, SafetyVerdict.BLOCKED, "给药途径冲突", "昏迷患者不能口服给药"))

        return results

    def get_blocking_reasons(self, drug_name, patient_conditions=None, lab_values=None, is_oral=True, is_unconscious=False):
        return [r.reason for r in self.check_drug(drug_name, patient_conditions, lab_values, is_oral, is_unconscious) if r.verdict == SafetyVerdict.BLOCKED]

    def get_warnings(self, drug_name, patient_conditions=None, lab_values=None, is_oral=True, is_unconscious=False):
        return [r.reason for r in self.check_drug(drug_name, patient_conditions, lab_values, is_oral, is_unconscious) if r.verdict == SafetyVerdict.WARNING]

    def is_contraindicated(self, drug_name, patient_conditions=None, lab_values=None, is_oral=True, is_unconscious=False):
        return len(self.get_blocking_reasons(drug_name, patient_conditions, lab_values, is_oral, is_unconscious)) > 0

    NSAID_LIST = ["ibuprofen","naproxen","diclofenac","indomethacin","ketorolac","celecoxib",
        "meloxicam","piroxicam","etodolac","mefenamic acid","ketoprofen","flurbiprofen",
        "meclofenamate","tolmetin","rofecoxib","sulindac","fenoprofen","oxaprozin","nabumetone"]

    @staticmethod
    def _drug_matches(pattern, drug_lower):
        if pattern == "*oral*": return True
        if pattern in drug_lower: return True
        if pattern == "statins" and any(s in drug_lower for s in ["atorvastatin","simvastatin","rosuvastatin","pravastatin","fluvastatin","lovastatin","pitavastatin"]): return True
        if pattern == "ace_inhibitors" and any(s in drug_lower for s in ["lisinopril","enalapril","captopril","ramipril","quinapril","benazepril","fosinopril","perindopril"]): return True
        if pattern == "nsaids" and any(s in drug_lower for s in SafetyRuleEngine.NSAID_LIST): return True
        if pattern == "arbs" and any(s in drug_lower for s in ["valsartan","losartan","irbesartan","candesartan","telmisartan","olmesartan"]): return True
        return False
