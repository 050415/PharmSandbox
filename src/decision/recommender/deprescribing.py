# -*- coding: utf-8 -*-
"""处方精简引擎 — 从"找替身"升级为"全局处方优化" """
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    DEPRESCRIBE = "deprescribe"
    REPLACE_SAFE = "replace_safe"
    REPLACE_CAUTIOUS = "replace_cautious"
    MONITOR = "monitor"


@dataclass
class PrescriptionAction:
    action_type: ActionType
    title: str
    description: str
    keep_drugs: List[str]
    stop_drugs: List[str]
    confidence: str = "high"
    evidence: str = ""


DEPRESCRIBING_RULES = [
    {
        "id": "afib_warfarin_aspirin",
        "description": "房颤患者华法林抗凝达标时，阿司匹林无额外获益且增加出血风险",
        "conditions": ["atrial_fibrillation","心房颤动","房颤","AF","afib"],
        "drug_to_keep": ["warfarin"],
        "drug_to_stop": ["aspirin"],
        "requires_good_control": True, "control_param": "inr", "control_range": (2.0, 3.0),
        "evidence": "AHA/ACC/HRS 2019 AFib指南: 无ACS/PCI史的房颤患者，单用口服抗凝药即可",
    },
    {
        "id": "afib_doac_aspirin",
        "description": "房颤患者DOAC抗凝时，阿司匹林增加出血风险",
        "conditions": ["atrial_fibrillation","心房颤动","房颤","AF","afib"],
        "drug_to_keep": ["dabigatran","rivaroxaban","apixaban","edoxaban"],
        "drug_to_stop": ["aspirin","clopidogrel"],
        "requires_good_control": False,
        "evidence": "AHA/ACC/HRS 2019: DOAC单药抗凝优于双抗",
    },
    {
        "id": "afib_warfarin_clopidogrel",
        "description": "房颤患者华法林抗凝达标时，氯吡格雷不增加获益",
        "conditions": ["atrial_fibrillation","心房颤动","房颤"],
        "drug_to_keep": ["warfarin"],
        "drug_to_stop": ["clopidogrel","ticlopidine","prasugrel"],
        "requires_good_control": True, "control_param": "inr", "control_range": (2.0, 3.0),
        "evidence": "ESC 2020 AFib指南: 稳定型CAD+AFib，单用OAC优于OAC+抗血小板",
    },
    {
        "id": "ckd_nsaid",
        "description": "CKD患者应避免NSAIDs",
        "conditions": ["ckd","CKD","renal_failure","renal_impairment","慢性肾病","肾功能不全","肾衰竭"],
        "drug_to_keep": [],
        "drug_to_stop": ["ibuprofen","naproxen","diclofenac","indomethacin","ketorolac","celecoxib","meloxicam","piroxicam"],
        "requires_good_control": False,
        "evidence": "KDIGO 2024 CKD指南: CKD患者避免长期使用NSAIDs",
    },
    {
        "id": "hypertension_nsaid",
        "description": "高血压患者使用NSAIDs可导致血压升高",
        "conditions": ["hypertension","高血压","HTN"],
        "drug_to_keep": [],
        "drug_to_stop": ["ibuprofen","naproxen","diclofenac","indomethacin","ketorolac","celecoxib"],
        "requires_good_control": False,
        "evidence": "AHA 2017高血压指南: NSAIDs可升高血压3-6mmHg",
    },
    {
        "id": "heart_failure_nsaid",
        "description": "心衰患者使用NSAIDs可导致水钠潴留加重心衰",
        "conditions": ["heart_failure","心力衰竭","心衰","CHF","HF"],
        "drug_to_keep": [],
        "drug_to_stop": ["ibuprofen","naproxen","diclofenac","indomethacin","ketorolac","celecoxib"],
        "requires_good_control": False,
        "evidence": "AHA/ACC 2022心衰指南: NSAIDs在心衰患者中应避免",
    },
]


class DeprescribingEngine:
    def analyze(self, drugs, interactions, patient_info=None, lab_values=None):
        if not interactions: return None
        conditions = (patient_info or {}).get('conditions', [])
        labs = lab_values or ((patient_info or {}).get('labs', {}) if patient_info else {})
        for interaction in interactions:
            d1, d2 = interaction.get('drug1',''), interaction.get('drug2','')
            for a, b in [(d1, d2), (d2, d1)]:
                action = self._check_pair(a, b, conditions, labs)
                if action: return action
        return None

    def _check_pair(self, keep_drug, stop_drug, conditions, labs):
        for rule in DEPRESCRIBING_RULES:
            if rule.get('conditions') and not self._match_conditions(rule['conditions'], conditions):
                continue
            keep_list = [d.lower() for d in rule['drug_to_keep']]
            if keep_list and not any(k in keep_drug.lower() for k in keep_list):
                continue
            stop_list = [d.lower() for d in rule['drug_to_stop']]
            if not any(s in stop_drug.lower() for s in stop_list):
                continue
            if rule.get('requires_good_control'):
                param = rule['control_param']
                val = labs.get(param)
                low, high = rule['control_range']
                if val is None or val < low or val > high:
                    continue
            control_info = ""
            if rule.get('requires_good_control'):
                p = rule['control_param']
                v = labs.get(p, '?')
                control_info = f"（当前{p.upper()} {v}，控制良好）"
            return PrescriptionAction(
                action_type=ActionType.DEPRESCRIBE,
                title=f"建议保留 {keep_drug}，停用 {stop_drug}",
                description=f"{rule['description']}。当前{keep_drug}治疗效果良好{control_info}，停用{stop_drug}即可解除高危DDI。",
                keep_drugs=[keep_drug], stop_drugs=[stop_drug],
                evidence=rule['evidence'],
            )
        return None

    @staticmethod
    def _match_conditions(rule_conds, patient_conds):
        pl = [c.lower().replace(' ','_') for c in patient_conds]
        for rc in rule_conds:
            rl = rc.lower().replace(' ','_')
            for pc in pl:
                if rl in pc or pc in rl: return True
        return False

    def generate_global_strategy(self, drugs, interactions, patient_info, lab_values,
                                 alternatives=None, _skip_fallback=False):
        deprescribe = self.analyze(drugs, interactions, patient_info, lab_values)
        if deprescribe:
            return {
                "strategy": "deprescribe",
                "primary_action": {
                    "type": deprescribe.action_type.value,
                    "title": deprescribe.title,
                    "description": deprescribe.description,
                    "keep_drugs": deprescribe.keep_drugs,
                    "stop_drugs": deprescribe.stop_drugs,
                    "confidence": deprescribe.confidence,
                    "evidence": deprescribe.evidence,
                },
                "reasoning": f"分析DDI冲突双方：{deprescribe.stop_drugs[0]}是DDI的问题药而非必需药。保留{', '.join(deprescribe.keep_drugs)}，停用{', '.join(deprescribe.stop_drugs)}可在不引入新风险的前提下化解DDI。",
                "confidence": "high",
                "alternatives": alternatives or [],
            }
        if _skip_fallback:
            return {"strategy": "no_deprescribe", "primary_action": None,
                    "reasoning": "未匹配处方精简规则", "confidence": "low", "alternatives": []}
        if alternatives:
            scored = [a for a in alternatives if a.get('tier') != 'blocked']
            safe = [a for a in scored if a.get('tier') in ('preferred','safe')]
            if safe:
                return {"strategy": "replace_safe", "primary_action": None,
                        "reasoning": "处方精简不适用，推荐跨类安全替换", "confidence": "medium", "alternatives": alternatives}
            return {"strategy": "replace_cautious", "primary_action": None,
                    "reasoning": "无可精简方案且无理想替换药", "confidence": "low", "alternatives": alternatives,
                    "warning": "所有替代药与当前用药均存在不同程度DDI风险"}
        return {"strategy": "no_action", "primary_action": None,
                "reasoning": "未触发DDI或无法确定优化方案", "confidence": "low", "alternatives": []}
