# -*- coding: utf-8 -*-
"""
药物名称NER模块 - 纯Python实现
功能：药物识别、名称标准化、剂量提取、症状/疾病提取
"""

import re
import os
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 中文药物名称字典（常见药物：中文名 -> 英文名）
# ---------------------------------------------------------------------------
_ZH_TO_EN: Dict[str, str] = {
    # 常见西药
    '阿司匹林': 'aspirin',
    '华法林': 'warfarin',
    '二甲双胍': 'metformin',
    '甲硝唑': 'metronidazole',
    '头孢': 'cephalosporin',
    '青霉素': 'penicillin',
    '布洛芬': 'ibuprofen',
    '对乙酰氨基酚': 'acetaminophen',
    '扑热息痛': 'acetaminophen',
    '阿莫西林': 'amoxicillin',
    '奥美拉唑': 'omeprazole',
    '氯吡格雷': 'clopidogrel',
    '阿托伐他汀': 'atorvastatin',
    '辛伐他汀': 'simvastatin',
    '美托洛尔': 'metoprolol',
    '氨氯地平': 'amlodipine',
    '硝苯地平': 'nifedipine',
    '缬沙坦': 'valsartan',
    '厄贝沙坦': 'irbesartan',
    '氯沙坦': 'losartan',
    '氢氯噻嗪': 'hydrochlorothiazide',
    '呋塞米': 'furosemide',
    '螺内酯': 'spironolactone',
    '地高辛': 'digoxin',
    '胺碘酮': 'amiodarone',
    '利多卡因': 'lidocaine',
    '普萘洛尔': 'propranolol',
    '格列本脲': 'glibenclamide',
    '格列齐特': 'gliclazide',
    '胰岛素': 'insulin',
    '左旋甲状腺素': 'levothyroxine',
    '泼尼松': 'prednisone',
    '地塞米松': 'dexamethasone',
    '氢化可的松': 'hydrocortisone',
    '甲氨蝶呤': 'methotrexate',
    '环磷酰胺': 'cyclophosphamide',
    '硫唑嘌呤': 'azathioprine',
    '他克莫司': 'tacrolimus',
    '环孢素': 'cyclosporine',
    '氟康唑': 'fluconazole',
    '伊曲康唑': 'itraconazole',
    '阿昔洛韦': 'aciclovir',
    '利巴韦林': 'ribavirin',
    '奥司他韦': 'oseltamivir',
    '阿奇霉素': 'azithromycin',
    '红霉素': 'erythromycin',
    '左氧氟沙星': 'levofloxacin',
    '诺氟沙星': 'norfloxacin',
    '克林霉素': 'clindamycin',
    '万古霉素': 'vancomycin',
    '庆大霉素': 'gentamicin',
    '多西环素': 'doxycycline',
    '氯霉素': 'chloramphenicol',
    '卡马西平': 'carbamazepine',
    '丙戊酸钠': 'sodium_valproate',
    '苯妥英钠': 'phenytoin',
    '加巴喷丁': 'gabapentin',
    '普瑞巴林': 'pregabalin',
    '氟西汀': 'fluoxetine',
    '舍曲林': 'sertraline',
    '帕罗西汀': 'paroxetine',
    '文拉法辛': 'venlafaxine',
    '米氮平': 'mirtazapine',
    '阿米替林': 'amitriptyline',
    '氯硝西泮': 'clonazepam',
    '阿普唑仑': 'alprazolam',
    '地西泮': 'diazepam',
    '唑吡坦': 'zolpidem',
    '奥氮平': 'olanzapine',
    '利培酮': 'risperidone',
    '喹硫平': 'quetiapine',
    '氯氮平': 'clozapine',
    '多奈哌齐': 'donepezil',
    '美金刚': 'memantine',
    '多巴胺': 'dopamine',
    '肾上腺素': 'epinephrine',
    '去甲肾上腺素': 'norepinephrine',
    '硝酸甘油': 'nitroglycerin',
    '肝素': 'heparin',
    '依诺肝素': 'enoxaparin',
    '达比加群': 'dabigatran',
    '利伐沙班': 'rivaroxaban',
    '奥利司他': 'orlistat',
    '西地那非': 'sildenafil',
    '他达拉非': 'tadalafil',
    '枸橼酸': 'citric_acid',
    '叶酸': 'folic_acid',
    '维生素C': 'vitamin_c',
    '维生素B': 'vitamin_b',
    '钙': 'calcium',
    '铁剂': 'iron',
    '维生素D': 'vitamin_d',
}

# 商品名映射（商品名 -> 通用名英文）
_BRAND_TO_GENERIC: Dict[str, str] = {
    '拜阿司匹灵': 'aspirin',
    '泰诺': 'acetaminophen',
    '芬必得': 'ibuprofen',
    '阿乐': 'atorvastatin',
    '立普妥': 'atorvastatin',
    '可定': 'rosuvastatin',
    '波立维': 'clopidogrel',
    '络活喜': 'amlodipine',
    '代文': 'valsartan',
    '科素亚': 'losartan',
    '美迪康': 'metformin',
    '格华止': 'metformin',
    '拜唐苹': 'acarbose',
    '诺和龙': 'repaglinide',
    '诺和灵': 'insulin',
    '来得时': 'insulin_glargine',
    '百忧解': 'fluoxetine',
    '左洛复': 'sertraline',
    '思瑞康': 'quetiapine',
    '再普乐': 'olanzapine',
    '维思通': 'risperidone',
}

# ---------------------------------------------------------------------------
# 中文疾病/症状 -> 英文标准化名称
# ---------------------------------------------------------------------------
_ZH_DISEASE_MAP: Dict[str, str] = {
    '高血压': 'hypertension',
    '糖尿病': 'diabetes',
    '高血脂': 'hyperlipidemia',
    '高脂血症': 'hyperlipidemia',
    '冠心病': 'coronary_heart_disease',
    '心绞痛': 'angina',
    '心肌梗死': 'myocardial_infarction',
    '心力衰竭': 'heart_failure',
    '心房颤动': 'atrial_fibrillation',
    '房颤': 'atrial_fibrillation',
    '中风': 'stroke',
    '脑卒中': 'stroke',
    '脑梗': 'cerebral_infarction',
    '脑梗塞': 'cerebral_infarction',
    '哮喘': 'asthma',
    '慢阻肺': 'copd',
    '肺炎': 'pneumonia',
    '支气管炎': 'bronchitis',
    '胃炎': 'gastritis',
    '胃溃疡': 'gastric_ulcer',
    '消化性溃疡': 'peptic_ulcer',
    '肝炎': 'hepatitis',
    '脂肪肝': 'fatty_liver',
    '肾炎': 'nephritis',
    '肾结石': 'kidney_stone',
    '尿路感染': 'urinary_tract_infection',
    '甲状腺功能亢进': 'hyperthyroidism',
    '甲亢': 'hyperthyroidism',
    '甲状腺功能减退': 'hypothyroidism',
    '甲减': 'hypothyroidism',
    '贫血': 'anemia',
    '白血病': 'leukemia',
    '淋巴瘤': 'lymphoma',
    '关节炎': 'arthritis',
    '类风湿': 'rheumatoid_arthritis',
    '痛风': 'gout',
    '骨质疏松': 'osteoporosis',
    '抑郁症': 'depression',
    '焦虑症': 'anxiety',
    '失眠': 'insomnia',
    '癫痫': 'epilepsy',
    '帕金森': 'parkinson',
    '阿尔茨海默': 'alzheimer',
    '老年痴呆': 'alzheimer',
    '偏头痛': 'migraine',
    '头痛': 'headache',
    '过敏': 'allergy',
    '湿疹': 'eczema',
    '荨麻疹': 'urticaria',
    '痤疮': 'acne',
    '肿瘤': 'tumor',
    '癌症': 'cancer',
    '乳腺癌': 'breast_cancer',
    '肺癌': 'lung_cancer',
    '肝癌': 'liver_cancer',
    '胃癌': 'gastric_cancer',
    '结肠癌': 'colon_cancer',
}

# ---------------------------------------------------------------------------
# 单位正则
# ---------------------------------------------------------------------------
_UNIT_PATTERN = r'(?:mg|g|μg|ug|mcg|ml|mL|IU|万单位|单位|片|粒|丸|袋|支|滴)'
_NUM_PATTERN = r'(?:\d+(?:\.\d+)?)'

# 频率词映射
_FREQ_MAP = {
    '一天一次': 1, '每天一次': 1, '每日一次': 1, 'qd': 1, '一天一回': 1,
    '一天两次': 2, '每天两次': 2, '每日两次': 2, 'bid': 2, '一天两回': 2,
    '一天三次': 3, '每天三次': 3, '每日三次': 3, 'tid': 3, '一天三回': 3,
    '一天四次': 4, '每天四次': 4, '每日四次': 4, 'qid': 4, '一天四回': 4,
    '每12小时': 2, '每八小时': 3, '每六小时': 4,
    '隔天一次': 0.5, '隔日一次': 0.5, 'qod': 0.5,
    '每周一次': 1/7, 'qw': 1/7,
}


class DrugNER:
    """药物命名实体识别器"""

    def __init__(self, drug_names_path: Optional[str] = None):
        """
        初始化NER，加载药物名称映射表。

        Args:
            drug_names_path: drug_names.tsv文件路径，默认 D:/drug/data/sider/drug_names.tsv
        """
        if drug_names_path is None:
            from src.config import DATA_ROOT
            drug_names_path = str(DATA_ROOT / "sider" / "drug_names.tsv")

        # 英文名 -> CID 映射
        self._en_to_cid: Dict[str, str] = {}
        # 小写英文名 -> 标准英文名
        self._en_lower: Dict[str, str] = {}
        # 中文名 -> 英文名（合并内置 + 从文件加载的）
        self._zh_to_en: Dict[str, str] = dict(_ZH_TO_EN)
        # 商品名 -> 英文名
        self._brand_to_generic: Dict[str, str] = dict(_BRAND_TO_GENERIC)

        self._load_drug_names(drug_names_path)

        # 构建用于匹配的英文药物名集合（按长度降序，优先匹配长词）
        self._en_names_sorted = sorted(self._en_lower.keys(), key=len, reverse=True)

        # 构建中文药物名匹配正则（按长度降序）
        all_zh = sorted(self._zh_to_en.keys(), key=len, reverse=True)
        all_brand = sorted(self._brand_to_generic.keys(), key=len, reverse=True)
        self._zh_drug_pattern = re.compile(
            '|'.join(re.escape(n) for n in all_zh + all_brand)
        ) if (all_zh or all_brand) else None

        # 疾病正则
        all_diseases = sorted(_ZH_DISEASE_MAP.keys(), key=len, reverse=True)
        self._disease_pattern = re.compile(
            '|'.join(re.escape(d) for d in all_diseases)
        )

        # 剂量正则
        self._dose_pattern = re.compile(
            rf'({_NUM_PATTERN})\s*({_UNIT_PATTERN})'
        )
        # 频率正则 - 直接匹配频率词
        self._freq_patterns = [(re.compile(re.escape(k)), v) for k, v in
                               sorted(_FREQ_MAP.items(), key=lambda x: len(x[0]), reverse=True)]

    def _load_drug_names(self, path: str):
        """加载drug_names.tsv，格式：CID100000085\tcarnitine"""
        if not os.path.isfile(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '\t' not in line:
                    continue
                parts = line.split('\t', 1)
                if len(parts) < 2:
                    continue
                cid = parts[0].strip()
                name = parts[1].strip().lower()
                self._en_to_cid[name] = cid
                self._en_lower[name] = name

    def recognize_drugs(self, text: str) -> List[Dict[str, str]]:
        """
        从文本中识别药物名称。

        Returns:
            [{'name': '阿司匹林', 'en_name': 'aspirin', 'type': 'chinese'}, ...]
        """
        results = []
        seen = set()

        # 1) 匹配中文药物名 + 商品名
        if self._zh_drug_pattern:
            for m in self._zh_drug_pattern.finditer(text):
                zh_name = m.group()
                if zh_name in seen:
                    continue
                seen.add(zh_name)
                en_name = self._zh_to_en.get(zh_name) or self._brand_to_generic.get(zh_name, '')
                cid = self._en_to_cid.get(en_name.lower(), '')
                results.append({
                    'name': zh_name,
                    'en_name': en_name,
                    'cid': cid,
                    'type': 'brand' if zh_name in self._brand_to_generic else 'chinese',
                })

        # 2) 匹配英文药物名（在文本中查找英文名）
        # 先对文本小写化后搜索
        text_lower = text.lower()
        for en_name in self._en_names_sorted:
            # 避免太短的通用词误匹配（如 'bile', 'calcium' 等基础物质）
            if len(en_name) < 4:
                continue
            # 使用 word boundary 匹配
            pattern = re.compile(r'\b' + re.escape(en_name) + r'\b', re.IGNORECASE)
            for m in pattern.finditer(text):
                matched = m.group()
                if matched.lower() in seen:
                    continue
                seen.add(matched.lower())
                cid = self._en_to_cid.get(en_name, '')
                results.append({
                    'name': matched,
                    'en_name': en_name,
                    'cid': cid,
                    'type': 'english',
                })

        return results

    def extract_drug_names(self, text: str) -> List[str]:
        """简化接口：返回识别到的药物名称列表（原始文本形式）"""
        return [r['name'] for r in self.recognize_drugs(text)]

    def normalize_drug(self, name: str) -> Dict[str, str]:
        """
        药物名称标准化。

        Args:
            name: 中文名、英文名或商品名

        Returns:
            {'input': ..., 'en_name': ..., 'cid': ...}
        """
        name_stripped = name.strip()

        # 尝试中文名
        en = self._zh_to_en.get(name_stripped)
        if en:
            cid = self._en_to_cid.get(en.lower(), '')
            return {'input': name_stripped, 'en_name': en, 'cid': cid}

        # 尝试商品名
        en = self._brand_to_generic.get(name_stripped)
        if en:
            cid = self._en_to_cid.get(en.lower(), '')
            return {'input': name_stripped, 'en_name': en, 'cid': cid}

        # 尝试英文名
        lower = name_stripped.lower()
        if lower in self._en_to_cid:
            return {'input': name_stripped, 'en_name': lower, 'cid': self._en_to_cid[lower]}

        return {'input': name_stripped, 'en_name': '', 'cid': ''}

    def extract_dosage(self, text: str) -> Dict[str, object]:
        """
        从文本中提取剂量和频率信息。

        Example:
            '每天吃两次，每次500mg' -> {'times_per_day': 2, 'dose': 500.0, 'unit': 'mg'}
        """
        result: Dict[str, object] = {}

        # 提取剂量
        dose_match = self._dose_pattern.search(text)
        if dose_match:
            result['dose'] = float(dose_match.group(1))
            result['unit'] = dose_match.group(2)

        # 提取频率
        for pattern, freq_val in self._freq_patterns:
            if pattern.search(text):
                result['times_per_day'] = freq_val
                break

        # 兜底：如果只出现"每天X次"的数字形式（允许中间有其他字，支持中文数字）
        if 'times_per_day' not in result:
            _cn_num = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
                       '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
            m = re.search(r'每[天日].{0,4}?(\d+|[一二两三四五六七八九十])\s*次', text)
            if m:
                val = m.group(1)
                result['times_per_day'] = _cn_num.get(val, int(val) if val.isdigit() else val)

        return result

    def extract_diseases(self, text: str) -> List[str]:
        """
        从文本中提取疾病/症状名称，返回标准化英文名。

        Example:
            '我有高血压和糖尿病' -> ['hypertension', 'diabetes']
        """
        results = []
        seen = set()
        for m in self._disease_pattern.finditer(text):
            zh = m.group()
            en = _ZH_DISEASE_MAP[zh]
            if en not in seen:
                seen.add(en)
                results.append(en)
        return results

    def analyze(self, text: str) -> Dict[str, object]:
        """
        综合分析：一次性提取药物、剂量、疾病。

        Returns:
            {
                'drugs': [...],
                'dosage': {...},
                'diseases': [...],
            }
        """
        return {
            'drugs': self.recognize_drugs(text),
            'dosage': self.extract_dosage(text),
            'diseases': self.extract_diseases(text),
        }


# ---------------------------------------------------------------------------
# 快捷函数
# ---------------------------------------------------------------------------
_default_ner: Optional[DrugNER] = None

def _get_ner() -> DrugNER:
    global _default_ner
    if _default_ner is None:
        _default_ner = DrugNER()
    return _default_ner

def recognize_drugs(text: str) -> List[str]:
    """快捷函数：识别药物名称"""
    return _get_ner().extract_drug_names(text)

def normalize_drug(name: str) -> Dict[str, str]:
    """快捷函数：药物名称标准化"""
    return _get_ner().normalize_drug(name)

def extract_dosage(text: str) -> Dict[str, object]:
    """快捷函数：提取剂量"""
    return _get_ner().extract_dosage(text)

def extract_diseases(text: str) -> List[str]:
    """快捷函数：提取疾病"""
    return _get_ner().extract_diseases(text)

def analyze(text: str) -> Dict[str, object]:
    """快捷函数：综合分析"""
    return _get_ner().analyze(text)
