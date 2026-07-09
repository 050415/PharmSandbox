# -*- coding: utf-8 -*-
"""推荐结果自审计模块 — 第二轮校验拦截漏网之鱼"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class AuditFinding:
    check_name: str
    severity: str
    message: str
    action: str

@dataclass
class AuditReport:
    drug_name: str
    original_tier: str
    adjusted_tier: str
    original_score: float
    adjusted_score: float
    findings: List[AuditFinding] = field(default_factory=list)
    passed: bool = True


class RecommendationAuditor:
    def audit(self, candidates, original_drug, original_atc, original_ddi_reasons,
              current_meds, patient_conditions, lab_values=None,
              _preloaded_reasoner=None, _preloaded_se_getter=None):
        reports = []
        audited = []
        for cand in candidates:
            report = self._audit_single(cand, original_drug, original_atc,
                original_ddi_reasons, current_meds, patient_conditions, lab_values,
                _preloaded_reasoner=_preloaded_reasoner, _preloaded_se_getter=_preloaded_se_getter)
            reports.append(report)
            adjusted = dict(cand)
            adjusted['tier'] = report.adjusted_tier
            adjusted['score'] = report.adjusted_score
            adjusted['audit_findings'] = [{'check': f.check_name, 'severity': f.severity, 'msg': f.message} for f in report.findings]
            adjusted['audit_passed'] = report.passed
            if report.adjusted_tier != 'blocked':
                audited.append(adjusted)
        return audited, reports

    def _audit_single(self, cand, original_drug, original_atc, original_ddi_reasons,
                      current_meds, patient_conditions, lab_values,
                      _preloaded_reasoner=None, _preloaded_se_getter=None):
        cand_name = cand.get('drug_name', '?')
        cand_atc = cand.get('atc_code', '')
        original_tier = cand.get('tier', 'unknown')
        original_score = cand.get('score', 0)
        adjusted_tier = original_tier
        adjusted_score = original_score
        findings = []
        passed = True

        # 检查1: 药理同类替换检测
        if cand_atc and original_atc:
            atc_level = self._atc_similarity_level(cand_atc, original_atc)
            if atc_level >= 5:
                findings.append(AuditFinding('药理同类替换检测', 'critical',
                    f'替代药 {cand_name}({cand_atc}) 与原药 {original_drug}({original_atc}) 属于同一ATC化学亚类，DDI风险大概率继承', 'block'))
                adjusted_tier = 'blocked'
                adjusted_score = -100
                passed = False
            elif atc_level >= 4:
                findings.append(AuditFinding('药理同类替换检测', 'warning',
                    f'替代药 {cand_name}({cand_atc}) 与原药 {original_drug}({original_atc}) 属同一ATC药理亚类但不同化学亚类，建议确认DDI风险', 'flag'))
                if adjusted_tier == 'preferred':
                    adjusted_tier = 'safe'
            elif atc_level >= 3:
                findings.append(AuditFinding('药理同类替换检测', 'warning',
                    f'同一ATC治疗大类，建议确认不继承DDI风险', 'flag'))

        # 检查2: DDI重验证
        try:
            if _preloaded_reasoner is None:
                from src.decision.llm_reasoner import MedicalReasoner
                _preloaded_reasoner = MedicalReasoner()
            reasoner = _preloaded_reasoner
            for med_name in current_meds:
                ddi = reasoner._lookup_ddi(
                    reasoner.normalize_drug_name(cand_name.lower()),
                    reasoner.normalize_drug_name(med_name.lower()))
                if ddi and ddi['severity'].rank >= 3:
                    findings.append(AuditFinding('DDI重验证', 'critical',
                        f'自审计发现: {cand_name}+{med_name}存在{ddi["severity"].label}级DDI→自动熔断', 'block'))
                    adjusted_tier = 'blocked'
                    adjusted_score = -100
                    passed = False
                elif ddi and ddi['severity'].rank >= 2:
                    findings.append(AuditFinding('DDI重验证', 'warning',
                        f'自审计: {cand_name}+{med_name}存在{ddi["severity"].label}级DDI', 'downgrade'))
                    if adjusted_tier == 'preferred':
                        adjusted_tier = 'caution'
        except Exception:
            pass

        # 检查3: 禁忌症复检
        try:
            from src.decision.recommender.safety_rules import SafetyRuleEngine
            safety = SafetyRuleEngine()
            blocking = safety.get_blocking_reasons(cand_name, patient_conditions, lab_values)
            if blocking:
                findings.append(AuditFinding('禁忌症复检', 'critical', f'自审计绝对禁忌: {"; ".join(blocking[:2])}→熔断', 'block'))
                adjusted_tier = 'blocked'
                adjusted_score = -100
                passed = False
            warns_list = safety.get_warnings(cand_name, patient_conditions, lab_values)
            if warns_list:
                findings.append(AuditFinding('禁忌症复检', 'warning', f'相对禁忌: {"; ".join(warns_list[:2])}', 'flag'))
        except Exception:
            pass

        # 检查4: 风险一致性
        try:
            if _preloaded_se_getter is None:
                from src.decision.recommender.recommender import DrugRecommender
                _rec = DrugRecommender()
                _rec._load_data()
                _preloaded_se_getter = _rec.get_drug_side_effects
            cand_se = set(_preloaded_se_getter(cand_name.lower()))
            orig_se = set(_preloaded_se_getter(original_drug.lower()))
            severe_kw = ['death','fatal','life threatening','cardiac arrest','respiratory failure','seizure','hemorrhage','arrhythmia','liver failure','renal failure']
            cand_severe = [s for s in cand_se if any(kw in s.lower() for kw in severe_kw)]
            orig_severe = [s for s in orig_se if any(kw in s.lower() for kw in severe_kw)]
            if len(cand_severe) > len(orig_severe) + 3:
                findings.append(AuditFinding('风险一致性检查', 'warning',
                    f'替代药高危副作用({len(cand_severe)}项)显著多于原药({len(orig_severe)}项)→降级', 'downgrade'))
                if adjusted_tier == 'preferred':
                    adjusted_tier = 'caution'
        except Exception:
            pass

        return AuditReport(drug_name=cand_name, original_tier=original_tier, adjusted_tier=adjusted_tier,
                          original_score=original_score, adjusted_score=adjusted_score, findings=findings, passed=passed)

    @staticmethod
    def _atc_similarity_level(atc1, atc2):
        if not atc1 or not atc2: return 0
        level = 0
        for c1, c2 in zip(atc1, atc2):
            if c1 == c2: level += 1
            else: break
        return level
