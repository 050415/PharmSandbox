"""
PharmSandbox - 医学LLM推理模块
基于LangGraph医疗智能体模式，使用本地模板+规则引擎实现：
1. DDI风险自然语言解释（代谢竞争、受体拮抗、协同增效等机制）
2. 基于患者病历的个性化用药建议
3. 医学知识库验证推荐合理性
4. 多轮对话式用药咨询

参考:
- SofiaLoukisa/medassist-langgraph-agentic-ai-dissertation (LangGraph医疗助手)
- MrRezaeiUofT/AMG-RAG (医学图谱RAG)

不依赖外部LLM API，全部使用本地模板+规则引擎生成解释。
"""

import re
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from enum import Enum

# 导入项目内模块
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.data.loader import DataLoader
from src.decision.risk_scorer.scorer import RiskScorer
from src.decision.recommender.recommender import DrugRecommender


# ======================================================================
# 枚举与常量
# ======================================================================

class DDIType(Enum):
    """药物-药物相互作用类型"""
    METABOLIC_COMPETITION = "metabolic_competition"       # 代谢竞争
    RECEPTOR_ANTAGONISM = "receptor_antagonism"            # 受体拮抗
    SYNERGISTIC_EFFECT = "synergistic_effect"              # 协同增效
    PHARMACOKINETIC = "pharmacokinetic"                    # 药代动力学
    PHARMACODYNAMIC = "pharmacodynamic"                    # 药效动力学
    ENZYME_INDUCTION = "enzyme_induction"                  # 酶诱导
    ENZYME_INHIBITION = "enzyme_inhibition"                # 酶抑制
    TRANSPORTER_COMPETITION = "transporter_competition"    # 转运体竞争
    QT_PROLONGATION = "qt_prolongation"                    # QT延长
    SEROTONIN_SYNDROME = "serotonin_syndrome"              # 5-HT综合征
    BLEEDING_RISK = "bleeding_risk"                        # 出血风险
    NEPHROTOXICITY = "nephrotoxicity"                      # 肾毒性叠加
    HEPATOTOXICITY = "hepatotoxicity"                      # 肝毒性叠加
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    """严重程度"""
    CONTRAINDICATED = ("禁忌", 4)
    MAJOR = ("重大", 3)
    MODERATE = ("中等", 2)
    MINOR = ("轻微", 1)
    UNKNOWN = ("未知", 0)

    def __init__(self, label: str, rank: int):
        self.label = label
        self.rank = rank


class ConversationRole(Enum):
    """对话角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ======================================================================
# DDI解释模板库
# ======================================================================

DDI_EXPLANATION_TEMPLATES: Dict[DDIType, Dict[str, Any]] = {
    DDIType.METABOLIC_COMPETITION: {
        "mechanism": (
            "{drug_a} 和 {drug_b} 都通过相同的肝药酶（主要是 {enzyme}）代谢。"
            "当两者同时使用时，会竞争同一代谢途径，导致一种或两种药物的血药浓度升高，"
            "增加不良反应风险。"
        ),
        "clinical_effect": (
            "可能出现 {drug_a} 或 {drug_b} 的血药浓度异常升高，"
            "增加剂量相关性毒性的风险，如{toxicity_symptoms}。"
        ),
        "recommendation": (
            "建议：1) 监测两药的血药浓度；2) 考虑调整剂量；"
            "3) 如有可能，替换为不经 {enzyme} 代谢的同类药物。"
        ),
        "severity": SeverityLevel.MODERATE,
    },
    DDIType.RECEPTOR_ANTAGONISM: {
        "mechanism": (
            "{drug_a} 和 {drug_b} 作用于相同的受体或信号通路（{receptor}），"
            "但发挥相反的药理作用。{drug_a} 为 {drug_a_action}，而 {drug_b} 为 {drug_b_action}，"
            "两者相互拮抗，导致治疗效果降低。"
        ),
        "clinical_effect": (
            "两种药物的治疗效果均可能减弱，无法达到预期的临床疗效。"
            "患者可能出现疾病控制不佳的症状。"
        ),
        "recommendation": (
            "建议：1) 避免同时处方两种作用相反的药物；"
            "2) 与主治医师讨论替代方案；3) 如必须合用，需密切监测疗效指标。"
        ),
        "severity": SeverityLevel.MODERATE,
    },
    DDIType.SYNERGISTIC_EFFECT: {
        "mechanism": (
            "{drug_a} 和 {drug_b} 具有相似的药理作用，合用后产生协同增效效应。"
            "虽然这在某些情况下是有意为之，但也可能使药效过强，"
            "超出安全治疗窗口。"
        ),
        "clinical_effect": (
            "合用后可能出现药效过强的反应，如{toxicity_symptoms}。"
            "需要密切监测患者状态。"
        ),
        "recommendation": (
            "建议：1) 如属有意联用，需仔细调整两药剂量；"
            "2) 加强不良反应监测；3) 评估联用的风险-获益比。"
        ),
        "severity": SeverityLevel.MINOR,
    },
    DDIType.ENZYME_INHIBITION: {
        "mechanism": (
            "{drug_a} 是 {enzyme} 的强效抑制剂。当与主要经 {enzyme} 代谢的 {drug_b} 合用时，"
            "{drug_b} 的代谢被显著抑制，导致其血药浓度升高 {fold_increase} 倍。"
        ),
        "clinical_effect": (
            "{drug_b} 血药浓度升高可导致{toxicity_symptoms}等剂量依赖性不良反应的风险显著增加。"
        ),
        "recommendation": (
            "建议：1) 将 {drug_b} 剂量减少 {dose_reduction}%；"
            "2) 或选用不经 {enzyme} 代谢的替代药物；"
            "3) 合用期间密切监测 {drug_b} 相关的不良反应。"
        ),
        "severity": SeverityLevel.MAJOR,
    },
    DDIType.ENZYME_INDUCTION: {
        "mechanism": (
            "{drug_a} 是 {enzyme} 的强效诱导剂，可显著加速 {enzyme} 对 {drug_b} 的代谢清除。"
            "这会导致 {drug_b} 的血药浓度降低，可能达不到治疗浓度。"
        ),
        "clinical_effect": (
            "{drug_b} 的疗效可能显著降低，导致治疗失败或疾病复发。"
        ),
        "recommendation": (
            "建议：1) 增加 {drug_b} 的剂量（通常需要增加 {dose_increase}%）；"
            "2) 或选用不受 {enzyme} 影响的替代药物；"
            "3) 停用 {drug_a} 后需及时将 {drug_b} 剂量回调。"
        ),
        "severity": SeverityLevel.MAJOR,
    },
    DDIType.TRANSPORTER_COMPETITION: {
        "mechanism": (
            "{drug_a} 和 {drug_b} 竞争相同的药物转运蛋白（如 {transporter}），"
            "导致其中一种药物的吸收或排泄受到影响。"
        ),
        "clinical_effect": (
            "可能出现 {drug_a} 或 {drug_b} 的生物利用度改变，"
            "影响治疗效果或增加不良反应风险。"
        ),
        "recommendation": (
            "建议：1) 两药服用时间间隔至少 2 小时；"
            "2) 监测受影响药物的血药浓度或疗效指标。"
        ),
        "severity": SeverityLevel.MODERATE,
    },
    DDIType.QT_PROLONGATION: {
        "mechanism": (
            "{drug_a} 和 {drug_b} 都具有延长心脏 QT 间期的作用。"
            "合用后 QT 延长效应叠加，显著增加致命性心律失常（尖端扭转型室速）的风险。"
        ),
        "clinical_effect": (
            "QT 间期显著延长，可能发生尖端扭转型室性心动过速（TdP），"
            "严重时可导致心源性猝死。"
        ),
        "recommendation": (
            "⚠️ 高危警告：1) 避免同时使用这两种药物；"
            "2) 如必须使用，需进行心电图监测（特别是 QTc 间期）；"
            "3) 纠正低钾血症、低镁血症等危险因素。"
        ),
        "severity": SeverityLevel.CONTRAINDICATED,
    },
    DDIType.SEROTONIN_SYNDROME: {
        "mechanism": (
            "{drug_a} 和 {drug_b} 均可增加中枢神经系统 5-羟色胺（血清素）水平。"
            "合用后可能引发 5-HT 综合征，这是一种潜在致命的药物不良反应。"
        ),
        "clinical_effect": (
            "可能出现 5-HT 综合征三联征：精神状态改变（焦虑、激越）、"
            "自主神经功能亢进（高热、心动过速、出汗）、"
            "神经肌肉异常（肌阵挛、反射亢进、震颤）。"
        ),
        "recommendation": (
            "⚠️ 高危警告：1) 避免同时使用；"
            "2) 如必须合用，从最低剂量开始并密切观察；"
            "3) 患者教育：出现上述症状立即就医。"
        ),
        "severity": SeverityLevel.CONTRAINDICATED,
    },
    DDIType.BLEEDING_RISK: {
        "mechanism": (
            "{drug_a} 具有抗凝/抗血小板作用，而 {drug_b} 可通过{mechanism_detail}增加出血倾向。"
            "两者合用时出血风险显著叠加。"
        ),
        "clinical_effect": (
            "可能出现严重出血事件，包括消化道出血、颅内出血、"
            "皮下瘀斑、牙龈出血等。需密切监测出血征象。"
        ),
        "recommendation": (
            "建议：1) 评估出血风险-获益比；"
            "2) 如必须合用，定期监测血红蛋白和凝血功能；"
            "3) 患者教育：避免同时使用含 NSAIDs 的非处方药。"
        ),
        "severity": SeverityLevel.MAJOR,
    },
    DDIType.NEPHROTOXICITY: {
        "mechanism": (
            "{drug_a} 和 {drug_b} 均具有潜在的肾脏毒性。"
            "合用后肾脏毒性叠加，可能加速肾功能损害。"
        ),
        "clinical_effect": (
            "肾功能可能快速恶化，表现为血肌酐升高、尿量减少、"
            "电解质紊乱等。严重时可导致急性肾损伤。"
        ),
        "recommendation": (
            "建议：1) 合用前评估基线肾功能；"
            "2) 合用期间每 3-7 天监测肾功能；"
            "3) 保证充分水化；4) 考虑使用肾毒性较小的替代药物。"
        ),
        "severity": SeverityLevel.MAJOR,
    },
    DDIType.HEPATOTOXICITY: {
        "mechanism": (
            "{drug_a} 和 {drug_b} 均有肝毒性报告。"
            "合用时肝脏代谢负担加重，肝损伤风险叠加。"
        ),
        "clinical_effect": (
            "可能出现肝功能异常，表现为转氨酶升高、胆红素升高、"
            "黄疸等。严重时可导致药物性肝衰竭。"
        ),
        "recommendation": (
            "建议：1) 合用前检测基线肝功能；"
            "2) 合用期间定期监测 ALT、AST、胆红素；"
            "3) 出现肝功能异常时及时停药。"
        ),
        "severity": SeverityLevel.MAJOR,
    },
    DDIType.PHARMACOKINETIC: {
        "mechanism": (
            "{drug_a} 通过药代动力学途径影响 {drug_b} 的体内过程"
            "（吸收、分布、代谢或排泄），导致 {drug_b} 的血药浓度发生变化。"
        ),
        "clinical_effect": (
            "{drug_b} 的疗效或毒性可能发生改变，需根据具体情况评估。"
        ),
        "recommendation": (
            "建议：1) 监测 {drug_b} 的血药浓度或疗效指标；"
            "2) 根据需要调整剂量。"
        ),
        "severity": SeverityLevel.MODERATE,
    },
    DDIType.PHARMACODYNAMIC: {
        "mechanism": (
            "{drug_a} 和 {drug_b} 在药效学层面存在相互作用，"
            "即一药改变了另一药的药理效应，而不改变其血药浓度。"
        ),
        "clinical_effect": (
            "可能出现预期之外的药效增强或减弱，需要临床密切关注。"
        ),
        "recommendation": (
            "建议：1) 密切监测疗效和不良反应；"
            "2) 根据临床反应调整用药方案。"
        ),
        "severity": SeverityLevel.MODERATE,
    },
    DDIType.UNKNOWN: {
        "mechanism": (
            "{drug_a} 和 {drug_b} 之间存在已记录的相互作用，"
            "但具体机制尚未完全阐明。"
        ),
        "clinical_effect": (
            "临床影响不确定，可能出现不可预测的药效变化或不良反应。"
        ),
        "recommendation": (
            "建议：1) 保持警惕，密切监测不良反应；"
            "2) 如有替代药物可选，优先考虑替换；"
            "3) 记录并报告任何异常反应。"
        ),
        "severity": SeverityLevel.MINOR,
    },
}


# ======================================================================
# 已知DDI知识库（常见药物对 → 交互类型 + 元数据）
# ======================================================================

KNOWN_DDI_KB: Dict[Tuple[str, str], Dict[str, Any]] = {
    # ---- 代谢竞争 (CYP450) ----
    ("warfarin", "fluconazole"): {
        "type": DDIType.ENZYME_INHIBITION,
        "severity": SeverityLevel.MAJOR,
        "params": {"enzyme": "CYP2C9", "fold_increase": "2-4", "dose_reduction": 50,
                    "toxicity_symptoms": "出血、瘀斑、鼻出血"},
    },
    ("warfarin", "amiodarone"): {
        "type": DDIType.ENZYME_INHIBITION,
        "severity": SeverityLevel.MAJOR,
        "params": {"enzyme": "CYP2C9/CYP3A4", "fold_increase": "2-3", "dose_reduction": 30,
                    "toxicity_symptoms": "出血风险显著增加"},
    },
    ("simvastatin", "clarithromycin"): {
        "type": DDIType.ENZYME_INHIBITION,
        "severity": SeverityLevel.MAJOR,
        "params": {"enzyme": "CYP3A4", "fold_increase": "5-10", "dose_reduction": 75,
                    "toxicity_symptoms": "横纹肌溶解、肌痛、肌酸激酶升高"},
    },
    ("simvastatin", "itraconazole"): {
        "type": DDIType.ENZYME_INHIBITION,
        "severity": SeverityLevel.CONTRAINDICATED,
        "params": {"enzyme": "CYP3A4", "fold_increase": "10+", "dose_reduction": 100,
                    "toxicity_symptoms": "横纹肌溶解、急性肾损伤"},
    },
    ("carbamazepine", "erythromycin"): {
        "type": DDIType.ENZYME_INHIBITION,
        "severity": SeverityLevel.MAJOR,
        "params": {"enzyme": "CYP3A4", "fold_increase": "2-3", "dose_reduction": 30,
                    "toxicity_symptoms": "头晕、复视、共济失调、恶心"},
    },
    ("phenytoin", "fluconazole"): {
        "type": DDIType.ENZYME_INHIBITION,
        "severity": SeverityLevel.MAJOR,
        "params": {"enzyme": "CYP2C9", "fold_increase": "2-3", "dose_reduction": 25,
                    "toxicity_symptoms": "眼球震颤、共济失调、意识模糊"},
    },
    ("cyclosporine", "rifampicin"): {
        "type": DDIType.ENZYME_INHIBITION,
        "severity": SeverityLevel.MAJOR,
        "params": {"enzyme": "CYP3A4", "fold_increase": "0.2 (浓度降低)", "dose_reduction": -100,
                    "toxicity_symptoms": "器官排斥反应（浓度降低所致）"},
    },

    # ---- 酶诱导 ----
    ("rifampicin", "oral_contraceptives"): {
        "type": DDIType.ENZYME_INDUCTION,
        "severity": SeverityLevel.MAJOR,
        "params": {"enzyme": "CYP3A4", "dose_increase": 100},
    },
    ("carbamazepine", "warfarin"): {
        "type": DDIType.ENZYME_INDUCTION,
        "severity": SeverityLevel.MAJOR,
        "params": {"enzyme": "CYP3A4/CYP2C9", "dose_increase": 50},
    },
    ("phenobarbital", "digoxin"): {
        "type": DDIType.ENZYME_INDUCTION,
        "severity": SeverityLevel.MODERATE,
        "params": {"enzyme": "CYP3A4", "dose_increase": 30},
    },

    # ---- QT延长 ----
    ("amiodarone", "sotalol"): {
        "type": DDIType.QT_PROLONGATION,
        "severity": SeverityLevel.CONTRAINDICATED,
        "params": {},
    },
    ("erythromycin", "haloperidol"): {
        "type": DDIType.QT_PROLONGATION,
        "severity": SeverityLevel.CONTRAINDICATED,
        "params": {},
    },
    ("ondansetron", "haloperidol"): {
        "type": DDIType.QT_PROLONGATION,
        "severity": SeverityLevel.MAJOR,
        "params": {},
    },
    ("methadone", "erythromycin"): {
        "type": DDIType.QT_PROLONGATION,
        "severity": SeverityLevel.MAJOR,
        "params": {},
    },

    # ---- 5-HT综合征 ----
    ("fluoxetine", "tramadol"): {
        "type": DDIType.SEROTONIN_SYNDROME,
        "severity": SeverityLevel.CONTRAINDICATED,
        "params": {},
    },
    ("sertraline", "tramadol"): {
        "type": DDIType.SEROTONIN_SYNDROME,
        "severity": SeverityLevel.CONTRAINDICATED,
        "params": {},
    },
    ("paroxetine", "linezolid"): {
        "type": DDIType.SEROTONIN_SYNDROME,
        "severity": SeverityLevel.CONTRAINDICATED,
        "params": {},
    },
    ("fluoxetine", "selegiline"): {
        "type": DDIType.SEROTONIN_SYNDROME,
        "severity": SeverityLevel.CONTRAINDICATED,
        "params": {},
    },
    ("venlafaxine", "tramadol"): {
        "type": DDIType.SEROTONIN_SYNDROME,
        "severity": SeverityLevel.CONTRAINDICATED,
        "params": {},
    },

    # ---- 出血风险 ----
    ("warfarin", "aspirin"): {
        "type": DDIType.BLEEDING_RISK,
        "severity": SeverityLevel.MAJOR,
        "params": {"mechanism_detail": "抑制血小板聚集和胃肠黏膜保护"},
    },
    ("heparin", "aspirin"): {
        "type": DDIType.BLEEDING_RISK,
        "severity": SeverityLevel.MAJOR,
        "params": {"mechanism_detail": "抗凝与抗血小板作用叠加"},
    },
    ("clopidogrel", "ibuprofen"): {
        "type": DDIType.BLEEDING_RISK,
        "severity": SeverityLevel.MODERATE,
        "params": {"mechanism_detail": "NSAIDs抑制血小板并损伤胃黏膜"},
    },
    ("rivaroxaban", "aspirin"): {
        "type": DDIType.BLEEDING_RISK,
        "severity": SeverityLevel.MAJOR,
        "params": {"mechanism_detail": "直接抗凝与抗血小板的双重作用"},
    },

    # ---- 肾毒性叠加 ----
    ("lithium", "ibuprofen"): {
        "type": DDIType.NEPHROTOXICITY,
        "severity": SeverityLevel.MAJOR,
        "params": {},
    },
    ("methotrexate", "ibuprofen"): {
        "type": DDIType.NEPHROTOXICITY,
        "severity": SeverityLevel.MAJOR,
        "params": {},
    },
    ("cisplatin", "gentamicin"): {
        "type": DDIType.NEPHROTOXICITY,
        "severity": SeverityLevel.MAJOR,
        "params": {},
    },
    ("vancomycin", "gentamicin"): {
        "type": DDIType.NEPHROTOXICITY,
        "severity": SeverityLevel.MAJOR,
        "params": {},
    },

    # ---- 肝毒性叠加 ----
    ("acetaminophen", "isoniazid"): {
        "type": DDIType.HEPATOTOXICITY,
        "severity": SeverityLevel.MAJOR,
        "params": {},
    },
    ("methotrexate", "leflunomide"): {
        "type": DDIType.HEPATOTOXICITY,
        "severity": SeverityLevel.MAJOR,
        "params": {},
    },

    # ---- 受体拮抗 ----
    ("metoprolol", "albuterol"): {
        "type": DDIType.RECEPTOR_ANTAGONISM,
        "severity": SeverityLevel.MODERATE,
        "params": {"receptor": "β-肾上腺素能受体",
                    "drug_a_action": "β受体阻滞剂（降低心率和支气管舒张）",
                    "drug_b_action": "β2受体激动剂（舒张支气管）"},
    },
    ("atenolol", "dobutamine"): {
        "type": DDIType.RECEPTOR_ANTAGONISM,
        "severity": SeverityLevel.MAJOR,
        "params": {"receptor": "β-肾上腺素能受体",
                    "drug_a_action": "β受体阻滞剂",
                    "drug_b_action": "β1受体激动剂（正性肌力）"},
    },
    ("nifedipine", "dantrolene"): {
        "type": DDIType.SYNERGISTIC_EFFECT,
        "severity": SeverityLevel.MODERATE,
        "params": {"toxicity_symptoms": "严重低血压和高钾血症"},
    },

    # ---- 协同增效 ----
    ("aspirin", "clopidogrel"): {
        "type": DDIType.SYNERGISTIC_EFFECT,
        "severity": SeverityLevel.MINOR,
        "params": {"toxicity_symptoms": "出血时间延长、瘀斑"},
    },
    ("lisinopril", "spironolactone"): {
        "type": DDIType.SYNERGISTIC_EFFECT,
        "severity": SeverityLevel.MODERATE,
        "params": {"toxicity_symptoms": "高钾血症"},
    },
    ("morphine", "diazepam"): {
        "type": DDIType.SYNERGISTIC_EFFECT,
        "severity": SeverityLevel.MAJOR,
        "params": {"toxicity_symptoms": "严重呼吸抑制、过度镇静、昏迷"},
    },
    ("oxycodone", "diazepam"): {
        "type": DDIType.SYNERGISTIC_EFFECT,
        "severity": SeverityLevel.MAJOR,
        "params": {"toxicity_symptoms": "严重呼吸抑制、过度镇静"},
    },

    # ---- 转运体竞争 ----
    ("digoxin", "amiodarone"): {
        "type": DDIType.TRANSPORTER_COMPETITION,
        "severity": SeverityLevel.MAJOR,
        "params": {"transporter": "P-糖蛋白（P-gp）"},
    },
    ("digoxin", "verapamil"): {
        "type": DDIType.TRANSPORTER_COMPETITION,
        "severity": SeverityLevel.MODERATE,
        "params": {"transporter": "P-糖蛋白（P-gp）"},
    },

    # ---- 补充高频临床DDI（压力测试发现缺失）----
    ("clopidogrel", "omeprazole"): {
        "type": DDIType.ENZYME_INHIBITION,
        "severity": SeverityLevel.MAJOR,
        "params": {"enzyme": "CYP2C19", "fold_increase": "0.3（活性代谢物降低70%）",
                    "dose_reduction": 100, "mechanism_detail": "氯吡格雷为前体药，需CYP2C19代谢活化；奥美拉唑抑制该酶导致氯吡格雷失效",
                    "toxicity_symptoms": "支架内血栓、心肌梗死、卒中"},
    },
    ("esomeprazole", "clopidogrel"): {
        "type": DDIType.ENZYME_INHIBITION,
        "severity": SeverityLevel.MODERATE,
        "params": {"enzyme": "CYP2C19", "fold_increase": "0.5（活性代谢物降低50%）",
                    "toxicity_symptoms": "抗血小板疗效降低"},
    },
    ("sildenafil", "nitroglycerin"): {
        "type": DDIType.SYNERGISTIC_EFFECT,
        "severity": SeverityLevel.CONTRAINDICATED,
        "params": {"mechanism_detail": "两者均通过NO-cGMP通路引起血管扩张；叠加导致血压断崖式下降",
                    "receptor": "PDE5/NO-cGMP通路",
                    "toxicity_symptoms": "致死性低血压、休克、心肌缺血、猝死"},
    },
    ("spironolactone", "lisinopril"): {
        "type": DDIType.SYNERGISTIC_EFFECT,
        "severity": SeverityLevel.MAJOR,
        "params": {"mechanism_detail": "螺内酯保钾+ACEI抑制醛固酮减少钾排泄→双重保钾效应",
                    "toxicity_symptoms": "高钾血症、心律失常、心脏骤停",
                    "dose_reduction": 50},
    },
    ("warfarin", "aspirin"): {
        "type": DDIType.RECEPTOR_ANTAGONISM,
        "severity": SeverityLevel.MAJOR,
        "params": {"mechanism_detail": "阿司匹林抑制血小板聚集+华法林抗凝血→双重抗凝",
                    "toxicity_symptoms": "消化道出血、颅内出血、皮下瘀斑、牙龈出血"},
    },
    ("ibuprofen", "lisinopril"): {
        "type": DDIType.RECEPTOR_ANTAGONISM,
        "severity": SeverityLevel.MODERATE,
        "params": {"mechanism_detail": "NSAIDs抑制前列腺素合成可减弱ACEI的降压效果并增加肾毒性风险",
                    "toxicity_symptoms": "血压升高、水肿、肾功能下降"},
    },
    ("metformin", "iodinated_contrast"): {
        "type": DDIType.PHARMACOKINETIC,
        "severity": SeverityLevel.CONTRAINDICATED,
        "params": {"mechanism_detail": "造影剂可致急性肾损伤→二甲双胍蓄积→乳酸性酸中毒",
                    "toxicity_symptoms": "乳酸性酸中毒、急性肾衰竭"},
    },
}

# 药物通用名别名映射（小写）
DRUG_ALIASES: Dict[str, str] = {
    "泰诺": "acetaminophen",
    "扑热息痛": "acetaminophen",
    "对乙酰氨基酚": "acetaminophen",
    "阿司匹林": "aspirin",
    "华法林": "warfarin",
    "法华林": "warfarin",
    "氯吡格雷": "clopidogrel",
    "立普妥": "atorvastatin",
    "阿托伐他汀": "atorvastatin",
    "辛伐他汀": "simvastatin",
    "美托洛尔": "metoprolol",
    "倍他乐克": "metoprolol",
    "氨氯地平": "amlodipine",
    "络活喜": "amlodipine",
    "二甲双胍": "metformin",
    "格华止": "metformin",
    "奥美拉唑": "omeprazole",
    "氟西汀": "fluoxetine",
    "百忧解": "fluoxetine",
    "舍曲林": "sertraline",
    "曲马多": "tramadol",
    "布洛芬": "ibuprofen",
    "芬必得": "ibuprofen",
    "地西泮": "diazepam",
    "安定": "diazepam",
    "阿莫西林": "amoxicillin",
    "头孢": "cephalexin",
    "红霉素": "erythromycin",
    "克拉霉素": "clarithromycin",
    "氟康唑": "fluconazole",
    "胺碘酮": "amiodarone",
    "地高辛": "digoxin",
    "锂盐": "lithium",
    "碳酸锂": "lithium",
    "异烟肼": "isoniazid",
    "利福平": "rifampicin",
    "万古霉素": "vancomycin",
    "庆大霉素": "gentamicin",
    "顺铂": "cisplatin",
    "甲氨蝶呤": "methotrexate",
    "环孢素": "cyclosporine",
    "卡马西平": "carbamazepine",
    "苯妥英": "phenytoin",
    "苯巴比妥": "phenobarbital",
    "吗啡": "morphine",
    "氧可酮": "oxycodone",
    "芬太尼": "fentanyl",
    "沙丁胺醇": "albuterol",
    "螺内酯": "spironolactone",
    "赖诺普利": "lisinopril",
    "索他洛尔": "sotalol",
    "维拉帕米": "verapamil",
    "硝苯地平": "nifedipine",
    "利伐沙班": "rivaroxaban",
    "肝素": "heparin",
    "多巴酚丁胺": "dobutamine",
    "丹曲林": "dantrolene",
    "左乙拉西坦": "levetiracetam",
    "苯乙肼": "selegiline",
    "吗氯贝胺": "moclobemide",
    "文拉法辛": "venlafaxine",
    "帕罗西汀": "paroxetine",
    "利奈唑胺": "linezolid",
    "他汀类": "statins",
}


# ======================================================================
# 患者画像与对话上下文
# ======================================================================

class PatientProfile:
    """患者画像，聚合病历信息"""

    def __init__(self, patient_id: str = "", name: str = "", age: int = 50,
                 gender: str = "", weight: float = 70.0, height: float = 170.0):
        self.patient_id = patient_id
        self.name = name
        self.age = age
        self.gender = gender
        self.weight = weight
        self.height = height
        self.conditions: List[str] = []          # 诊断/疾病史
        self.current_meds: List[str] = []        # 当前用药
        self.allergies: List[str] = []           # 过敏史
        self.lab_results: Dict[str, Any] = {}    # 检验结果
        self.notes: List[str] = []               # 备注

    @property
    def bmi(self) -> float:
        if self.height > 0:
            return round(self.weight / (self.height / 100) ** 2, 1)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "weight": self.weight,
            "height": self.height,
            "bmi": self.bmi,
            "conditions": self.conditions,
            "current_meds": self.current_meds,
            "allergies": self.allergies,
            "lab_results": self.lab_results,
            "notes": self.notes,
        }

    def get_risk_info(self) -> Dict[str, Any]:
        """转换为RiskScorer所需的患者信息格式"""
        return {
            "age": self.age,
            "gender": self.gender,
            "conditions": self.conditions,
        }


class ConversationTurn:
    """对话轮次"""

    def __init__(self, role: ConversationRole, content: str,
                 metadata: Optional[Dict[str, Any]] = None):
        self.role = role
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class ConversationSession:
    """多轮对话会话"""

    def __init__(self, session_id: str, patient: Optional[PatientProfile] = None):
        self.session_id = session_id
        self.patient = patient
        self.turns: List[ConversationTurn] = []
        self.created_at = datetime.now().isoformat()
        self._context_summary: str = ""

    def add_turn(self, role: ConversationRole, content: str,
                 metadata: Optional[Dict[str, Any]] = None):
        self.turns.append(ConversationTurn(role, content, metadata))

    def get_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.turns[-last_n:]]

    def get_context_window(self, last_n: int = 5) -> str:
        """获取最近N轮对话的上下文字符串"""
        recent = self.turns[-last_n * 2:]  # 每轮包含user+assistant
        lines = []
        for t in recent:
            prefix = "患者" if t.role == ConversationRole.USER else "药师"
            lines.append(f"{prefix}：{t.content}")
        return "\n".join(lines)


# ======================================================================
# 查询意图识别
# ======================================================================

class IntentType(Enum):
    """用户查询意图类型"""
    DDI_CHECK = "ddi_check"                      # 查询两种药是否冲突
    MEDICATION_ADVICE = "medication_advice"        # 请求用药建议
    DRUG_INFO = "drug_info"                        # 查询药物信息
    SIDE_EFFECT = "side_effect"                    # 查询副作用
    ALTERNATIVE = "alternative"                    # 请求替代药推荐
    GENERAL_QUESTION = "general_question"          # 一般用药问题
    DOSAGE_QUESTION = "dosage_question"            # 剂量相关问题
    SAFETY_CHECK = "safety_check"                  # 安全性评估
    UNKNOWN = "unknown"


# 意图识别关键词映射
_INTENT_PATTERNS: List[Tuple[IntentType, List[str]]] = [
    (IntentType.DDI_CHECK, [
        r"一起[服用吃用]", r"同时[服用吃用]", r"冲突", r"相互作用",
        r"能不能一起", r"可以一起", r"合用", r"联用", r"搭配",
        r"间隔", r"间隔多久", r"有没有交互",
        r"interact", r"together", r"combine",
    ]),
    (IntentType.ALTERNATIVE, [
        r"替代", r"替换", r"换[一什么]", r"同类药", r"还有什么药",
        r"代替", r"可[以能]换", r"alternative", r"substitute",
    ]),
    (IntentType.SIDE_EFFECT, [
        r"副作用", r"不良反应", r"有什么反应", r"有什么害",
        r"副反应", r"毒性", r"副作用大吗", r"安全吗",
        r"side effect", r"adverse",
    ]),
    (IntentType.MEDICATION_ADVICE, [
        r"怎么[服用吃用]", r"用药[建议方案]", r"该[吃用]什么",
        r"处方", r"建议", r"推荐", r"指导",
        r"advice", r"recommend", r"suggest",
    ]),
    (IntentType.DRUG_INFO, [
        r"是什么药", r"什么药", r"药[物信息]", r"作用",
        r"功效", r"适应[症征]", r"用[途法]", r"说明",
        r"是什么", r"information", r"about",
    ]),
    (IntentType.DOSAGE_QUESTION, [
        r"剂量", r"用量", r"[吃用]多少", r"一天几次",
        r"每次", r"吃几[片粒]", r"dosage", r"dose",
    ]),
    (IntentType.SAFETY_CHECK, [
        r"安全", r"风险", r"危[险害]", r"评估",
        r"有没有问题", r"能[吃用]吗", r"safety", r"risk",
    ]),
]


# ======================================================================
# 核心推理引擎
# ======================================================================

class MedicalReasoner:
    """
    医学LLM推理引擎

    基于模板+规则引擎的本地推理，不依赖外部LLM API。
    整合 DataLoader、RiskScorer、DrugRecommender 三大模块。

    使用方式::

        reasoner = MedicalReasoner()
        result = reasoner.explain_ddi("warfarin", "aspirin")
        print(result["explanation"])

        # 多轮对话
        session = reasoner.start_consultation(patient_profile)
        response = reasoner.chat(session, "华法林和阿司匹林能一起吃吗？")
        print(response)
    """

    def __init__(self, data_root=None):
        if data_root is None:
            from src.config import DATA_ROOT
            data_root = str(DATA_ROOT)
        self.data_root = data_root
        self.loader = DataLoader(data_root=data_root)
        self.risk_scorer = RiskScorer(data_root=data_root)
        self.recommender = DrugRecommender(data_root=data_root)

        # 对话会话存储
        self._sessions: Dict[str, ConversationSession] = {}

        # 拼接通用名正则（用于意图识别中的药物名提取）
        all_names = sorted(DRUG_ALIASES.keys(), key=len, reverse=True)
        self._drug_name_pattern = re.compile(
            "|".join(re.escape(n) for n in all_names), re.IGNORECASE
        )

    # ------------------------------------------------------------------
    # 药物名标准化
    # ------------------------------------------------------------------

    def normalize_drug_name(self, name: str) -> str:
        """将中文药名/别名标准化为英文通用名（小写）"""
        name_lower = name.strip().lower()
        if name_lower in DRUG_ALIASES:
            return DRUG_ALIASES[name_lower]
        # 原名小写作为默认
        return name_lower

    def _extract_drug_names(self, text: str) -> List[str]:
        """从用户输入中提取药物名称"""
        found = []
        for match in self._drug_name_pattern.finditer(text):
            cn_name = match.group()
            en_name = DRUG_ALIASES.get(cn_name.lower(), cn_name)
            if en_name not in found:
                found.append(en_name)
        # 如果中文匹配不到，尝试直接作为英文名
        if not found:
            # 简单的英文药名提取
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
            for w in words:
                wl = w.lower()
                if wl not in found and wl not in {"what", "this", "that", "with", "from",
                                                     "about", "drug", "药", "可以", "一起",
                                                     "能不能", "have", "been", "will"}:
                    found.append(wl)
        return found

    # ------------------------------------------------------------------
    # 意图识别
    # ------------------------------------------------------------------

    def identify_intent(self, text: str) -> Tuple[IntentType, float]:
        """
        识别用户查询意图

        Returns:
            (IntentType, confidence) 元组
        """
        text_lower = text.lower()
        scores: Dict[IntentType, float] = defaultdict(float)

        for intent, patterns in _INTENT_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    scores[intent] += 1.0

        if not scores:
            return IntentType.UNKNOWN, 0.0

        best = max(scores, key=scores.get)
        total_patterns = sum(len(p) for _, p in _INTENT_PATTERNS)
        confidence = min(1.0, scores[best] / 3.0)  # 归一化
        return best, confidence

    # ------------------------------------------------------------------
    # 1. DDI风险解释
    # ------------------------------------------------------------------

    def explain_ddi(self, drug_a: str, drug_b: str,
                    patient_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成两种药物之间相互作用的详细自然语言解释

        Args:
            drug_a: 第一种药物名称
            drug_b: 第二种药物名称
            patient_info: 患者信息（用于个性化解释）

        Returns:
            完整的DDI解释报告
        """
        norm_a = self.normalize_drug_name(drug_a)
        norm_b = self.normalize_drug_name(drug_b)

        # 查询知识库（双向查找）
        ddi_record = self._lookup_ddi(norm_a, norm_b)

        if ddi_record is None:
            return self._generate_no_ddi_explanation(drug_a, drug_b, norm_a, norm_b)

        ddi_type = ddi_record["type"]
        severity = ddi_record["severity"]
        params = ddi_record.get("params", {})

        # 获取模板
        template = DDI_EXPLANATION_TEMPLATES.get(ddi_type, DDI_EXPLANATION_TEMPLATES[DDIType.UNKNOWN])

        # 填充模板参数
        fill_params = {
            "drug_a": drug_a,
            "drug_b": drug_b,
            **params,
        }
        # 补充缺失的模板变量
        for key in ["enzyme", "receptor", "transporter", "toxicity_symptoms",
                     "drug_a_action", "drug_b_action", "fold_increase",
                     "dose_reduction", "dose_increase", "mechanism_detail"]:
            fill_params.setdefault(key, "未知")

        mechanism = self._safe_format(template["mechanism"], fill_params)
        clinical_effect = self._safe_format(template["clinical_effect"], fill_params)
        recommendation = self._safe_format(template["recommendation"], fill_params)

        # 个性化风险评估
        risk_context = ""
        if patient_info:
            risk_context = self._generate_patient_risk_context(
                norm_a, norm_b, patient_info
            )

        # 组装最终解释
        explanation = self._assemble_ddi_explanation(
            drug_a=drug_a,
            drug_b=drug_b,
            ddi_type=ddi_type,
            severity=severity,
            mechanism=mechanism,
            clinical_effect=clinical_effect,
            recommendation=recommendation,
            risk_context=risk_context,
        )

        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "ddi_type": ddi_type.value,
            "severity": severity.label,
            "severity_rank": severity.rank,
            "explanation": explanation,
            "mechanism": mechanism,
            "clinical_effect": clinical_effect,
            "recommendation": recommendation,
            "risk_context": risk_context,
            "has_known_ddi": True,
        }

    def _lookup_ddi(self, norm_a: str, norm_b: str) -> Optional[Dict[str, Any]]:
        """在知识库中双向查找DDI"""
        key_forward = (norm_a, norm_b)
        key_reverse = (norm_b, norm_a)

        if key_forward in KNOWN_DDI_KB:
            return KNOWN_DDI_KB[key_forward]
        if key_reverse in KNOWN_DDI_KB:
            return KNOWN_DDI_KB[key_reverse]
        return None

    def _generate_no_ddi_explanation(self, drug_a: str, drug_b: str,
                                      norm_a: str, norm_b: str) -> Dict[str, Any]:
        """当知识库中没有已知DDI时生成解释"""
        # 尝试从SIDER检查是否有共同副作用
        common_se = self._find_common_side_effects(norm_a, norm_b)

        if common_se:
            se_list = ", ".join(common_se[:5])
            explanation = (
                f"📋 {drug_a} 与 {drug_b} 的相互作用评估\n"
                f"{'=' * 50}\n\n"
                f"🔍 评估结论：目前没有检索到 {drug_a} 和 {drug_b} 之间明确的药物相互作用记录。\n\n"
                f"⚠️ 但需注意：两种药物存在以下共同副作用（{len(common_se)}项），"
                f"合用时可能会增加这些不良反应的发生风险：\n"
                f"  • {se_list}\n\n"
                f"💡 建议：\n"
                f"  1. 即使无已知DDI，合用新药时仍建议密切观察不良反应\n"
                f"  2. 如有任何异常反应，及时咨询医师或药师\n"
                f"  3. 建议记录用药时间，便于追踪不良反应"
            )
            severity_label = "需注意"
        else:
            explanation = (
                f"📋 {drug_a} 与 {drug_b} 的相互作用评估\n"
                f"{'=' * 50}\n\n"
                f"✅ 评估结论：目前没有检索到 {drug_a} 和 {drug_b} 之间已知的药物相互作用。\n"
                f"   两种药物在已知数据中也没有共同副作用记录。\n\n"
                f"💡 建议：\n"
                f"  1. 虽然未发现已知DDI，仍建议按医嘱服药\n"
                f"  2. 开始新的联合用药后，注意观察任何异常反应\n"
                f"  3. 定期复查相关指标以确保安全"
            )
            severity_label = "未发现"

        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "ddi_type": "none",
            "severity": severity_label,
            "severity_rank": 0,
            "explanation": explanation,
            "mechanism": "未发现已知相互作用机制",
            "clinical_effect": "无已知协同或拮抗效应",
            "recommendation": "常规监测即可",
            "risk_context": "",
            "has_known_ddi": False,
            "common_side_effects": common_se[:10],
        }

    def _find_common_side_effects(self, drug_a: str, drug_b: str) -> List[str]:
        """查找两种药物的共同副作用"""
        try:
            se_a = {str(s) for s in self.recommender.get_drug_side_effects(drug_a) if isinstance(s, str)}
            se_b = {str(s) for s in self.recommender.get_drug_side_effects(drug_b) if isinstance(s, str)}
            return list(se_a & se_b)
        except Exception:
            return []

    def _generate_patient_risk_context(self, drug_a: str, drug_b: str,
                                        patient_info: Dict[str, Any]) -> str:
        """基于患者信息生成个性化风险上下文"""
        lines = []
        age = patient_info.get("age", 50)
        conditions = [c.lower() for c in patient_info.get("conditions", [])]
        current_meds = patient_info.get("current_meds", [])

        lines.append(f"\n🏥 个性化风险评估（患者信息）：")

        # 年龄因素
        if age > 65:
            lines.append(f"  • 年龄 {age} 岁（老年患者）：药物代谢能力下降，"
                        f"肝肾清除率降低，DDI风险可能放大。建议从低剂量起始。")
        elif age < 18:
            lines.append(f"  • 年龄 {age} 岁（儿童/青少年）：药物代谢酶系统尚未完全成熟，"
                        f"需特别注意剂量调整。")

        # 疾病因素
        organ_risk_map = {
            "renal": ("肾脏", "肾功能不全可能影响药物排泄，增加蓄积风险"),
            "hepatic": ("肝脏", "肝功能不全可能影响药物代谢，增加毒性风险"),
            "cardiac": ("心脏", "心脏疾病可能使QT延长等心脏毒性风险增加"),
            "diabetes": ("糖尿病", "糖尿病合并症可能影响多器官功能"),
        }
        for cond in conditions:
            for keyword, (organ, risk_desc) in organ_risk_map.items():
                if keyword in cond:
                    lines.append(f"  • {organ}相关疾病（{cond}）：{risk_desc}")

        # 多药因素
        if current_meds and len(current_meds) >= 3:
            lines.append(f"  • 多药联用（当前 {len(current_meds)} 种药物）："
                        f"多药联用本身即增加DDI风险，叠加新药需格外谨慎。")

        # 肝肾功能检查提醒
        if any(kw in " ".join(conditions) for kw in ["ckd", "renal", "kidney", "肾"]):
            lines.append(f"  • 建议检查：血肌酐、BUN、eGFR")
        if any(kw in " ".join(conditions) for kw in ["liver", "hepatic", "肝"]):
            lines.append(f"  • 建议检查：ALT、AST、胆红素、白蛋白")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _assemble_ddi_explanation(self, drug_a: str, drug_b: str,
                                   ddi_type: DDIType, severity: SeverityLevel,
                                   mechanism: str, clinical_effect: str,
                                   recommendation: str,
                                   risk_context: str) -> str:
        """组装最终的DDI解释文本"""
        severity_icon = {
            SeverityLevel.CONTRAINDICATED: "🔴",
            SeverityLevel.MAJOR: "🟠",
            SeverityLevel.MODERATE: "🟡",
            SeverityLevel.MINOR: "🟢",
            SeverityLevel.UNKNOWN: "⚪",
        }
        icon = severity_icon.get(severity, "⚪")

        ddi_type_cn = {
            DDIType.METABOLIC_COMPETITION: "代谢竞争",
            DDIType.RECEPTOR_ANTAGONISM: "受体拮抗",
            DDIType.SYNERGISTIC_EFFECT: "协同增效",
            DDIType.PHARMACOKINETIC: "药代动力学相互作用",
            DDIType.PHARMACODYNAMIC: "药效动力学相互作用",
            DDIType.ENZYME_INDUCTION: "酶诱导",
            DDIType.ENZYME_INHIBITION: "酶抑制",
            DDIType.TRANSPORTER_COMPETITION: "转运体竞争",
            DDIType.QT_PROLONGATION: "QT间期延长",
            DDIType.SEROTONIN_SYNDROME: "5-HT综合征风险",
            DDIType.BLEEDING_RISK: "出血风险增加",
            DDIType.NEPHROTOXICITY: "肾毒性叠加",
            DDIType.HEPATOTOXICITY: "肝毒性叠加",
            DDIType.UNKNOWN: "未分类相互作用",
        }
        type_cn = ddi_type_cn.get(ddi_type, "未分类")

        parts = [
            f"📋 {drug_a} 与 {drug_b} 的药物相互作用报告",
            f"{'=' * 55}",
            f"",
            f"{icon} 严重程度：{severity.label}（级别 {severity.rank}/4）",
            f"🔬 相互作用类型：{type_cn}",
            f"",
            f"📖 作用机制：",
            f"  {mechanism}",
            f"",
            f"⚡ 临床影响：",
            f"  {clinical_effect}",
            f"",
            f"💡 处理建议：",
            f"  {recommendation}",
        ]

        if risk_context:
            parts.append(risk_context)

        # 免责声明
        parts.extend([
            f"",
            f"{'─' * 55}",
            f"⚠️ 免责声明：以上信息基于药物数据库和药理学知识生成，仅供临床参考。",
            f"   请以主治医师的临床判断为准。如有紧急情况，请立即就医。",
        ])

        return "\n".join(parts)

    @staticmethod
    def _safe_format(template: str, params: Dict[str, str]) -> str:
        """安全的字符串格式化，未提供的参数使用默认值"""
        try:
            return template.format(**params)
        except KeyError:
            # 逐步替换已知键
            result = template
            for k, v in params.items():
                result = result.replace("{" + k + "}", str(v))
            return result

    # ------------------------------------------------------------------
    # 2. 个性化用药建议
    # ------------------------------------------------------------------

    def generate_medication_advice(self, patient: PatientProfile,
                                    target_drugs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        根据患者病历生成个性化用药建议

        Args:
            patient: 患者画像
            target_drugs: 目标药物列表（为None则分析当前用药）

        Returns:
            个性化用药建议报告
        """
        drugs = target_drugs or patient.current_meds
        if not drugs:
            return {
                "advice": "当前没有需要评估的药物。请提供用药清单或目标药物。",
                "warnings": [],
                "alternatives": [],
            }

        norm_drugs = [self.normalize_drug_name(d) for d in drugs]
        risk_info = patient.get_risk_info()

        warnings = []
        ddi_findings = []
        alternatives = []

        # 1. 检查所有药物对的DDI
        for i in range(len(norm_drugs)):
            for j in range(i + 1, len(norm_drugs)):
                ddi_result = self.explain_ddi(norm_drugs[i], norm_drugs[j], risk_info)
                if ddi_result["has_known_ddi"]:
                    ddi_findings.append(ddi_result)
                    if ddi_result["severity_rank"] >= 3:
                        warnings.append(
                            f"⚠️ {drugs[i]} + {drugs[j]}: "
                            f"{ddi_result['severity']}级{ddi_result['ddi_type']}风险"
                        )

        # 2. 评估每种药物对患者的风险
        drug_risks = []
        for drug in norm_drugs:
            try:
                risk = self.risk_scorer.calculate_combination_risk(
                    [drug], risk_info
                )
                drug_risks.append({
                    "drug": drug,
                    "risk": risk.get("final_score", 0),
                    "level": risk.get("level", "未知"),
                })
            except Exception:
                pass

        # 3. 针对高风险DDI推荐替代药
        for finding in ddi_findings:
            if finding["severity_rank"] >= 2:
                try:
                    alts = self.recommender.recommend_alternatives(
                        finding["drug_b"],
                        current_meds=[d for d in norm_drugs if d != finding["drug_b"]],
                        patient_conditions=patient.conditions,
                        top_k=3,
                    )
                    if alts:
                        alternatives.append({
                            "replacing": finding["drug_b"],
                            "reason": f"避免与 {finding['drug_a']} 的 {finding['severity']}级DDI",
                            "candidates": alts,
                        })
                except Exception:
                    pass

        # 4. 生成个性化建议文本
        advice_text = self._format_medication_advice(
            patient, drugs, ddi_findings, drug_risks, warnings, alternatives
        )

        return {
            "advice": advice_text,
            "ddi_findings": [
                {"drug_a": d["drug_a"], "drug_b": d["drug_b"],
                 "severity": d["severity"], "type": d["ddi_type"]}
                for d in ddi_findings
            ],
            "drug_risks": drug_risks,
            "warnings": warnings,
            "alternatives": alternatives,
        }

    def _format_medication_advice(self, patient: PatientProfile,
                                    drugs: List[str],
                                    ddi_findings: List[Dict],
                                    drug_risks: List[Dict],
                                    warnings: List[str],
                                    alternatives: List[Dict]) -> str:
        """格式化个性化用药建议"""
        parts = [
            f"🏥 个性化用药评估报告",
            f"{'=' * 50}",
            f"",
            f"👤 患者：{patient.name or '未填写'}"
            f"  | 年龄：{patient.age}岁"
            f"  | 性别：{patient.gender or '未填写'}",
        ]

        if patient.conditions:
            parts.append(f"📋 诊断/病史：{', '.join(patient.conditions)}")
        parts.append(f"💊 当前用药：{', '.join(drugs)}")
        parts.append("")

        # DDI 评估
        if ddi_findings:
            parts.append(f"⚡ 药物相互作用评估（发现 {len(ddi_findings)} 项）：")
            for i, finding in enumerate(ddi_findings, 1):
                parts.append(f"  {i}. {finding['drug_a']} ⟷ {finding['drug_b']}")
                parts.append(f"     严重程度：{finding['severity']} | 类型：{finding['ddi_type']}")
            parts.append("")
        else:
            parts.append("✅ 药物相互作用评估：未发现已知的药物间相互作用\n")

        # 警告
        if warnings:
            parts.append("🚨 重要警告：")
            for w in warnings:
                parts.append(f"  {w}")
            parts.append("")

        # 个体风险
        if drug_risks:
            parts.append("📊 各药物个体风险评分（0-100）：")
            for dr in drug_risks:
                bar_len = int(dr["risk"] / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                parts.append(f"  {dr['drug']:20s} [{bar}] {dr['risk']:5.1f} ({dr['level']})")
            parts.append("")

        # 替代建议
        if alternatives:
            parts.append("🔄 替代药物建议：")
            for alt in alternatives:
                parts.append(f"  • 替换 {alt['replacing']}（原因：{alt['reason']}）：")
                for cand in alt["candidates"][:3]:
                    parts.append(f"    → {cand['drug_name']} (匹配度: {cand['score']})")
                    if cand.get("reasons"):
                        parts.append(f"      理由: {'; '.join(cand['reasons'])}")
            parts.append("")

        # 通用建议
        parts.extend([
            f"💡 通用建议：",
            f"  1. 严格按照医嘱时间和剂量服药",
            f"  2. 如出现任何不适，及时记录并咨询医师",
            f"  3. 定期复查相关指标（肝肾功能、血常规等）",
            f"  4. 未经医师同意，不要自行停药或调整剂量",
            f"",
            f"{'─' * 50}",
            f"⚠️ 本报告基于药物数据库自动生成，仅供参考，不替代医师诊断。",
        ])

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 3. 知识库验证
    # ------------------------------------------------------------------

    def validate_recommendation(self, recommended_drug: str,
                                 patient: PatientProfile,
                                 reason: str = "") -> Dict[str, Any]:
        """
        验证推荐药物的合理性

        Args:
            recommended_drug: 被推荐的药物
            patient: 患者画像
            reason: 推荐原因

        Returns:
            验证结果
        """
        norm_drug = self.normalize_drug_name(recommended_drug)
        issues = []
        checks = []

        # 检查1：与当前用药的DDI
        for med in patient.current_meds:
            norm_med = self.normalize_drug_name(med)
            if norm_med == norm_drug:
                continue
            ddi = self._lookup_ddi(norm_drug, norm_med)
            if ddi:
                sev = ddi["severity"]
                if sev.rank >= 3:
                    issues.append(
                        f"🔴 与当前用药 {med} 存在{sev.label}级DDI（{ddi['type'].value}）"
                    )
                elif sev.rank >= 2:
                    issues.append(
                        f"🟡 与当前用药 {med} 存在{sev.label}级DDI（{ddi['type'].value}）"
                    )
                checks.append(f"DDI检查 {recommended_drug} × {med}: {sev.label}")
            else:
                checks.append(f"DDI检查 {recommended_drug} × {med}: 未发现")

        # 检查2：过敏史
        for allergy in patient.allergies:
            allergy_norm = self.normalize_drug_name(allergy)
            if allergy_norm == norm_drug or allergy.lower() in norm_drug:
                issues.append(f"🔴 患者有 {allergy} 过敏史，使用 {recommended_drug} 存在过敏风险！")

        # 检查3：疾病禁忌
        disease_contraindications = {
            "renal": {
                "drugs": ["gentamicin", "vancomycin", "cisplatin", "methotrexate", "lithium"],
                "warning": "肾功能不全患者应慎用或避免使用肾毒性药物",
            },
            "hepatic": {
                "drugs": ["acetaminophen", "isoniazid", "methotrexate", "statins"],
                "warning": "肝功能不全患者应慎用肝毒性药物",
            },
            "asthma": {
                "drugs": ["atenolol", "metoprolol", "propranolol"],
                "warning": "哮喘患者应避免使用非选择性β受体阻滞剂",
            },
        }
        for cond in patient.conditions:
            cond_lower = cond.lower()
            for keyword, info in disease_contraindications.items():
                if keyword in cond_lower and norm_drug in info["drugs"]:
                    issues.append(f"🟠 {info['warning']}（患者诊断：{cond}）")

        # 检查4：年龄相关
        if patient.age > 75:
            checks.append("老年患者提示：建议从低剂量起始，缓慢滴定")
        if patient.age < 18:
            checks.append("儿童/青少年提示：确认该药物在该年龄段的安全性数据")

        # 检查5：通过知识图谱查询药物信息
        drug_info = {}
        try:
            indications = self.recommender.get_drug_indications(norm_drug)
            side_effects = self.recommender.get_drug_side_effects(norm_drug)
            atc_code = self.recommender.get_drug_atc(norm_drug)
            drug_info = {
                "indications": indications[:5],
                "side_effect_count": len(side_effects),
                "atc_code": atc_code,
            }
            checks.append(f"药物信息查询：ATC={atc_code}, "
                         f"适应症={len(indications)}项, 副作用={len(side_effects)}项")
        except Exception as e:
            checks.append(f"药物信息查询：暂不可用 ({e})")

        # 综合判定
        critical_count = sum(1 for i in issues if "🔴" in i)
        warning_count = sum(1 for i in issues if "🟠" in i or "🟡" in i)

        if critical_count > 0:
            verdict = "❌ 不推荐使用"
            verdict_detail = "存在禁忌或严重DDI，强烈建议更换方案"
        elif warning_count > 0:
            verdict = "⚠️ 谨慎使用"
            verdict_detail = "存在中等风险，需加强监测或调整剂量"
        else:
            verdict = "✅ 可以使用"
            verdict_detail = "未发现明显的禁忌或高风险DDI"

        # 生成验证报告文本
        report = self._format_validation_report(
            recommended_drug, patient, verdict, verdict_detail,
            issues, checks, drug_info, reason
        )

        return {
            "drug": recommended_drug,
            "verdict": verdict,
            "verdict_detail": verdict_detail,
            "issues": issues,
            "checks": checks,
            "drug_info": drug_info,
            "report": report,
            "safe_to_use": critical_count == 0,
        }

    def _format_validation_report(self, drug: str, patient: PatientProfile,
                                    verdict: str, verdict_detail: str,
                                    issues: List[str], checks: List[str],
                                    drug_info: Dict, reason: str) -> str:
        """格式化验证报告"""
        parts = [
            f"🔍 药物推荐合理性验证报告",
            f"{'=' * 50}",
            f"",
            f"💊 验证药物：{drug}",
            f"👤 患者：{patient.name or '未填写'}（{patient.age}岁）",
        ]
        if reason:
            parts.append(f"📝 推荐原因：{reason}")
        parts.append(f"{'─' * 50}")

        # 综合判定
        parts.extend([f"", f"📋 验证结论：{verdict}", f"   {verdict_detail}", f""])

        # 发现的问题
        if issues:
            parts.append("🚨 发现的问题：")
            for issue in issues:
                parts.append(f"  {issue}")
            parts.append("")

        # 验证过程
        parts.append("🔬 验证过程：")
        for check in checks:
            parts.append(f"  ✓ {check}")
        parts.append("")

        # 药物信息
        if drug_info:
            parts.append("📊 药物知识库信息：")
            if drug_info.get("atc_code"):
                parts.append(f"  ATC分类: {drug_info['atc_code']}")
            if drug_info.get("indications"):
                parts.append(f"  适应症: {', '.join(drug_info['indications'][:5])}")
            if drug_info.get("side_effect_count"):
                parts.append(f"  已知副作用: {drug_info['side_effect_count']} 项")
            parts.append("")

        parts.extend([
            f"{'─' * 50}",
            f"⚠️ 本验证基于数据库和规则引擎自动完成，仅供参考。最终用药决定请遵医嘱。",
        ])
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 4. 多轮对话式用药咨询
    # ------------------------------------------------------------------

    def start_consultation(self, patient: Optional[PatientProfile] = None,
                           session_id: Optional[str] = None) -> str:
        """
        开始一个新的用药咨询会话

        Args:
            patient: 患者画像（可选）
            session_id: 会话ID（自动生成如果不提供）

        Returns:
            会话ID
        """
        if session_id is None:
            session_id = hashlib.md5(
                datetime.now().isoformat().encode()
            ).hexdigest()[:12]

        session = ConversationSession(session_id, patient)

        # 系统提示
        system_msg = self._build_system_prompt(patient)
        session.add_turn(ConversationRole.SYSTEM, system_msg)

        # 欢迎消息
        welcome = self._build_welcome_message(patient)
        session.add_turn(ConversationRole.ASSISTANT, welcome)

        self._sessions[session_id] = session
        return session_id

    def chat(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """
        处理用户的对话消息

        Args:
            session_id: 会话ID
            user_message: 用户消息

        Returns:
            回复结果
        """
        session = self._sessions.get(session_id)
        if session is None:
            return {"reply": "会话不存在或已过期，请重新开始咨询。", "session_id": session_id}

        # 记录用户消息
        session.add_turn(ConversationRole.USER, user_message)

        # 意图识别
        intent, confidence = self.identify_intent(user_message)

        # 提取药物名
        drug_names = self._extract_drug_names(user_message)

        # 根据意图路由处理
        reply, metadata = self._route_and_respond(
            session, user_message, intent, confidence, drug_names
        )

        # 记录回复
        session.add_turn(ConversationRole.ASSISTANT, reply, metadata)

        return {
            "reply": reply,
            "session_id": session_id,
            "intent": intent.value,
            "confidence": confidence,
            "extracted_drugs": drug_names,
            "metadata": metadata,
        }

    def _route_and_respond(self, session: ConversationSession,
                            user_message: str, intent: IntentType,
                            confidence: float,
                            drug_names: List[str]) -> Tuple[str, Dict[str, Any]]:
        """根据意图路由到对应的处理逻辑"""
        patient = session.patient
        metadata = {"intent": intent.value, "confidence": confidence}

        if intent == IntentType.DDI_CHECK and len(drug_names) >= 2:
            # DDI查询
            result = self.explain_ddi(drug_names[0], drug_names[1],
                                       patient.get_risk_info() if patient else None)
            reply = result["explanation"]
            metadata["ddi_result"] = {
                "severity": result["severity"],
                "type": result["ddi_type"],
                "has_ddi": result["has_known_ddi"],
            }

        elif intent == IntentType.SAFETY_CHECK and drug_names:
            # 安全性评估
            if patient:
                result = self.validate_recommendation(drug_names[0], patient)
                reply = result["report"]
            else:
                reply = (
                    f"要评估 {drug_names[0]} 的安全性，我需要了解您的基本情况。\n"
                    f"请提供：\n"
                    f"  1. 年龄\n"
                    f"  2. 是否有肝肾疾病等基础病\n"
                    f"  3. 当前正在使用的其他药物\n"
                    f"  4. 是否有过敏史"
                )

        elif intent == IntentType.ALTERNATIVE and drug_names:
            # 替代药推荐
            try:
                current_meds = [d for d in (patient.current_meds if patient else [])
                               if self.normalize_drug_name(d) != drug_names[0]]
                alts = self.recommender.recommend_alternatives(
                    drug_names[0],
                    current_meds=current_meds if current_meds else None,
                    patient_conditions=patient.conditions if patient else None,
                    top_k=5,
                )
                if alts:
                    lines = [
                        f"🔄 {drug_names[0]} 的替代药物推荐：\n"
                    ]
                    for i, alt in enumerate(alts, 1):
                        lines.append(f"  {i}. {alt['drug_name']} (ATC: {alt['atc_code']})")
                        lines.append(f"     匹配度: {alt['score']}/100")
                        if alt.get("reasons"):
                            lines.append(f"     理由: {'; '.join(alt['reasons'])}")
                        if alt.get("indications"):
                            lines.append(f"     适应症: {', '.join(alt['indications'][:3])}")
                        lines.append("")
                    lines.append("💡 以上推荐基于ATC分类和适应症匹配，具体选择请咨询医师。")
                    reply = "\n".join(lines)
                else:
                    reply = f"抱歉，暂未找到 {drug_names[0]} 的合适替代药物。建议咨询医师。"
            except Exception as e:
                reply = f"查询替代药物时出现错误：{e}"

        elif intent == IntentType.DRUG_INFO and drug_names:
            # 药物信息查询
            reply = self._query_drug_info(drug_names[0])

        elif intent == IntentType.SIDE_EFFECT and drug_names:
            # 副作用查询
            reply = self._query_side_effects(drug_names[0])

        elif intent == IntentType.MEDICATION_ADVICE:
            # 用药建议
            if patient and drug_names:
                advice = self.generate_medication_advice(patient, drug_names)
                reply = advice["advice"]
                metadata["advice_result"] = {
                    "warning_count": len(advice["warnings"]),
                    "ddi_count": len(advice["ddi_findings"]),
                }
            elif patient and patient.current_meds:
                advice = self.generate_medication_advice(patient)
                reply = advice["advice"]
            else:
                reply = (
                    "我可以为您提供个性化用药建议。请告诉我：\n"
                    "  1. 您目前在用哪些药物？\n"
                    "  2. 您的年龄和诊断/疾病史？\n"
                    "  3. 是否有过敏史？\n"
                    "  或者您可以先注册患者信息，我会记住这些信息。"
                )

        elif intent == IntentType.DOSAGE_QUESTION and drug_names:
            reply = (
                f"关于 {drug_names[0]} 的剂量问题：\n"
                f"  具体剂量应由医师根据您的病情、体重、肝肾功能等因素综合决定。\n"
                f"  一般情况下，请严格遵照处方标注的剂量和频次服药。\n"
                f"  如有疑问，请咨询您的主治医师或药师。"
            )

        else:
            # 通用回复
            reply = self._generate_general_response(user_message, drug_names, patient)

        return reply, metadata

    def _query_drug_info(self, drug_name: str) -> str:
        """查询药物信息"""
        norm = self.normalize_drug_name(drug_name)
        lines = [f"💊 药物信息：{drug_name}\n{'─' * 40}\n"]

        try:
            atc = self.recommender.get_drug_atc(norm)
            if atc:
                lines.append(f"  ATC分类: {atc}")
        except Exception:
            pass

        try:
            indications = self.recommender.get_drug_indications(norm)
            if indications:
                lines.append(f"\n  📋 适应症（前10项）：")
                for ind in indications[:10]:
                    lines.append(f"    • {ind}")
        except Exception:
            lines.append("  适应症信息：暂不可用")

        try:
            ses = self.recommender.get_drug_side_effects(norm)
            if ses:
                lines.append(f"\n  ⚠️ 已知副作用（共{len(ses)}项，列出前10项）：")
                for se in ses[:10]:
                    lines.append(f"    • {se}")
        except Exception:
            lines.append("  副作用信息：暂不可用")

        lines.append(f"\n💡 以上信息基于SIDER/DrugCentral数据库，仅供参考。")
        return "\n".join(lines)

    def _query_side_effects(self, drug_name: str) -> str:
        """查询副作用"""
        norm = self.normalize_drug_name(drug_name)

        try:
            ses = self.recommender.get_drug_side_effects(norm)
            if ses:
                lines = [
                    f"⚠️ {drug_name} 的副作用报告\n{'─' * 40}\n",
                    f"  共有 {len(ses)} 项已记录的副作用，以下列出前15项：\n",
                ]
                for se in ses[:15]:
                    lines.append(f"  • {se}")
                lines.extend([
                    f"\n  💡 注意：",
                    f"  • 并非所有人都会出现以上副作用",
                    f"  • 副作用的发生频率因人而异",
                    f"  • 如出现严重不良反应，请立即就医",
                ])
                return "\n".join(lines)
            else:
                return f"未找到 {drug_name} 的副作用记录。"
        except Exception as e:
            return f"查询 {drug_name} 副作用时出现错误：{e}"

    def _generate_general_response(self, user_message: str,
                                    drug_names: List[str],
                                    patient: Optional[PatientProfile]) -> str:
        """生成通用回复"""
        if drug_names:
            return (
                f"我注意到您提到了 {', '.join(drug_names)}。\n"
                f"您可以问我：\n"
                f"  • 两种药能不能一起吃（药物相互作用查询）\n"
                f"  • 某药的副作用有哪些\n"
                f"  • 某药有没有替代药\n"
                f"  • 用药安全评估\n"
                f"  • 个性化用药建议\n"
                f"请具体描述您的问题。"
            )

        return (
            "您好！我是药盘推演系统的智能用药顾问。我可以帮您：\n\n"
            "  1️⃣ 查询两种药物是否存在相互作用\n"
            "  2️⃣ 获取药物的详细信息和副作用\n"
            "  3️⃣ 寻找替代药物推荐\n"
            "  4️⃣ 根据您的病历提供个性化用药建议\n"
            "  5️⃣ 评估新药与现有用药的安全性\n\n"
            "请告诉我您需要什么帮助？"
        )

    def _build_system_prompt(self, patient: Optional[PatientProfile]) -> str:
        """构建系统提示"""
        base = (
            "你是一个专业的智能用药顾问系统，基于药物数据库和药理学知识提供用药指导。\n"
            "你的回答应该：\n"
            "  - 准确、专业、有据可依\n"
            "  - 使用中文回复\n"
            "  - 在必要时提醒患者咨询医师\n"
            "  - 永远不替代医师的临床判断"
        )
        if patient:
            base += f"\n\n当前患者信息：\n{json.dumps(patient.to_dict(), ensure_ascii=False, indent=2)}"
        return base

    def _build_welcome_message(self, patient: Optional[PatientProfile]) -> str:
        """构建欢迎消息"""
        if patient and patient.name:
            meds = ", ".join(patient.current_meds) if patient.current_meds else "暂无记录"
            return (
                f"您好 {patient.name}！欢迎使用智能用药咨询服务。\n\n"
                f"我已了解到您的基本信息：\n"
                f"  • 年龄：{patient.age}岁\n"
                f"  • 当前用药：{meds}\n"
                f"  • 诊断/病史：{', '.join(patient.conditions) if patient.conditions else '暂无记录'}\n\n"
                f"请随时告诉我您的用药疑问，我会基于药物数据库为您提供专业建议。"
            )
        return (
            "您好！欢迎使用智能用药咨询服务。\n\n"
            "我是一个基于药物数据库的智能用药顾问，可以帮您：\n"
            "  • 查询药物相互作用\n"
            "  • 了解药物副作用\n"
            "  • 获取替代药推荐\n"
            "  • 评估用药安全性\n\n"
            "请随时提问，或先告诉我您的基本情况以便获得更个性化的建议。"
        )

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """获取会话"""
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        结束会话并生成咨询摘要

        Returns:
            会话摘要
        """
        session = self._sessions.pop(session_id, None)
        if session is None:
            return None

        user_turns = [t for t in session.turns if t.role == ConversationRole.USER]
        drug_mentions = set()
        for t in user_turns:
            drugs = self._extract_drug_names(t.content)
            drug_mentions.update(drugs)

        return {
            "session_id": session_id,
            "started_at": session.created_at,
            "ended_at": datetime.now().isoformat(),
            "total_turns": len(session.turns),
            "user_messages": len(user_turns),
            "drugs_discussed": list(drug_mentions),
            "patient_id": session.patient.patient_id if session.patient else None,
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有活跃会话"""
        return [
            {
                "session_id": sid,
                "patient": s.patient.name if s.patient else None,
                "turns": len(s.turns),
                "created_at": s.created_at,
            }
            for sid, s in self._sessions.items()
        ]

    # ------------------------------------------------------------------
    # 批量分析
    # ------------------------------------------------------------------

    def batch_ddi_check(self, drug_list: List[str],
                        patient_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        批量检查药物列表中所有药物对的DDI

        Args:
            drug_list: 药物名称列表
            patient_info: 患者信息

        Returns:
            批量DDI检查结果
        """
        norm_drugs = [self.normalize_drug_name(d) for d in drug_list]
        findings = []
        safe_pairs = []

        for i in range(len(norm_drugs)):
            for j in range(i + 1, len(norm_drugs)):
                result = self.explain_ddi(norm_drugs[i], norm_drugs[j], patient_info)
                if result["has_known_ddi"]:
                    findings.append({
                        "drug_a": drug_list[i],
                        "drug_b": drug_list[j],
                        "severity": result["severity"],
                        "type": result["ddi_type"],
                        "severity_rank": result["severity_rank"],
                    })
                else:
                    safe_pairs.append((drug_list[i], drug_list[j]))

        # 按严重程度排序
        findings.sort(key=lambda x: x["severity_rank"], reverse=True)

        return {
            "total_drugs": len(drug_list),
            "total_pairs": len(norm_drugs) * (len(norm_drugs) - 1) // 2,
            "ddi_found": len(findings),
            "safe_pairs": len(safe_pairs),
            "findings": findings,
            "safe": len(findings) == 0,
        }


# ======================================================================
# CLI 入口 / Demo
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PharmSandbox 医学LLM推理模块 测试")
    print("=" * 60)

    reasoner = MedicalReasoner()

    # ---- 测试1: DDI解释 ----
    print("\n" + "=" * 60)
    print("测试1: DDI风险解释 - 华法林 vs 阿司匹林")
    print("=" * 60)
    result = reasoner.explain_ddi("华法林", "阿司匹林", patient_info={
        "age": 72, "conditions": ["CKD", "hypertension"],
    })
    print(result["explanation"])

    # ---- 测试2: 无已知DDI ----
    print("\n" + "=" * 60)
    print("测试2: 无已知DDI - 阿莫西林 vs 二甲双胍")
    print("=" * 60)
    result2 = reasoner.explain_ddi("阿莫西林", "二甲双胍")
    print(result2["explanation"])

    # ---- 测试3: 个性化用药建议 ----
    print("\n" + "=" * 60)
    print("测试3: 个性化用药建议")
    print("=" * 60)
    patient = PatientProfile(
        patient_id="P001", name="张三", age=70, gender="男",
    )
    patient.conditions = ["高血压", "糖尿病", "CKD"]
    patient.current_meds = ["华法林", "阿司匹林", "二甲双胍"]
    advice = reasoner.generate_medication_advice(patient)
    print(advice["advice"])

    # ---- 测试4: 推荐验证 ----
    print("\n" + "=" * 60)
    print("测试4: 推荐合理性验证")
    print("=" * 60)
    validation = reasoner.validate_recommendation("布洛芬", patient, reason="止痛需要")
    print(validation["report"])

    # ---- 测试5: 多轮对话 ----
    print("\n" + "=" * 60)
    print("测试5: 多轮对话式用药咨询")
    print("=" * 60)
    session_id = reasoner.start_consultation(patient)
    print(f"会话已创建: {session_id}\n")

    test_messages = [
        "华法林和氟康唑一起吃安全吗？",
        "那有什么替代氟康唑的药吗？",
        "华法林有哪些副作用？",
        "我想了解一下二甲双胍这个药",
    ]

    for msg in test_messages:
        print(f"👤 患者：{msg}")
        resp = reasoner.chat(session_id, msg)
        print(f"\n{resp['reply']}")
        print(f"\n[意图: {resp['intent']}, 置信度: {resp['confidence']:.2f}, "
              f"提取药物: {resp['extracted_drugs']}]")
        print("-" * 60)

    # 结束会话
    summary = reasoner.end_session(session_id)
    print(f"\n📋 会话摘要: {json.dumps(summary, ensure_ascii=False, indent=2)}")

    # ---- 测试6: 批量DDI检查 ----
    print("\n" + "=" * 60)
    print("测试6: 批量DDI检查")
    print("=" * 60)
    batch = reasoner.batch_ddi_check(
        ["华法林", "阿司匹林", "氟康唑", "二甲双胍", "胺碘酮"],
        patient_info={"age": 70, "conditions": ["CKD"]},
    )
    print(f"检查药物数: {batch['total_drugs']}")
    print(f"总药物对数: {batch['total_pairs']}")
    print(f"发现DDI数: {batch['ddi_found']}")
    print(f"安全配对数: {batch['safe_pairs']}")
    for f in batch["findings"]:
        print(f"  {f['drug_a']} ⟷ {f['drug_b']}: {f['severity']} ({f['type']})")
