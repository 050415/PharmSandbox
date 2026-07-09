# -*- coding: utf-8 -*-
"""
PharmSandbox - API 安检门（Layer 1: API Gate）
主API: 阿里云用药安全（国内可用，免费额度1000次/月）
回退: 本地临床知识库 + FDA/PubMed（海外可用时）
"""

import json
import time
import os
import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class FDAResult:
    """OpenFDA 查询结果"""
    drug_name: str
    has_boxed_warning: bool = False
    boxed_warning_text: str = ""
    contraindications: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    adverse_reactions: List[str] = field(default_factory=list)
    source: str = "FDA OpenFDA"

    def to_dict(self):
        return {
            "drug": self.drug_name,
            "boxed_warning": self.has_boxed_warning,
            "boxed_text": self.boxed_warning_text[:500] if self.boxed_warning_text else "",
            "contraindications": self.contraindications[:5],
            "warnings": self.warnings[:5],
            "source": self.source,
        }


class APIGate:
    """
    API 安检门：阿里云用药安全（主）+ FDA/PubMed（回退）
    """

    # 阿里云用药安全 API（免费额度 1000次/月）
    ALIBABA_BASE = "https://drugsafe.shumaidata.com/v1"
    # 回退：美国 FDA/PubMed（海外服务器可用时）
    OPENFDA_BASE = "https://api.fda.gov/drug/label.json"
    RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
    PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, alibaba_appcode: str = None, cache_ttl: int = 3600):
        self._alibaba_appcode = alibaba_appcode or os.environ.get("ALIBABA_DRUGSAFE_APPCODE", "")
        self._cache: Dict[str, FDAResult] = {}
        self._cache_ttl = cache_ttl
        self._last_request = 0.0

    # ==================== 阿里云用药安全查询 ====================

    def _alibaba_drug_info(self, drug_name: str) -> Optional[Dict]:
        """查询阿里云用药安全 - 药品说明书精简版"""
        if not self._alibaba_appcode:
            return None
        try:
            url = f"{self.ALIBABA_BASE}/search?keyword={urllib.parse.quote(drug_name)}"
            req = urllib.request.Request(url, headers={
                'Authorization': f'APPCODE {self._alibaba_appcode}',
                'User-Agent': 'PharmSandbox/1.0'
            })
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception:
            return None

    def _alibaba_drug_interactions(self, drug_a: str, drug_b: str) -> Optional[Dict]:
        """查询阿里云 - 药物相互作用"""
        if not self._alibaba_appcode:
            return None
        try:
            url = f"{self.ALIBABA_BASE}/interaction?drugA={urllib.parse.quote(drug_a)}&drugB={urllib.parse.quote(drug_b)}"
            req = urllib.request.Request(url, headers={
                'Authorization': f'APPCODE {self._alibaba_appcode}',
                'User-Agent': 'PharmSandbox/1.0'
            })
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception:
            return None

    # ==================== FDA 黑框警告查询 ====================

    def check_boxed_warning(self, drug_name: str) -> FDAResult:
        """
        查询药品安全警告：阿里云优先，FDA 回退
        """
        result = FDAResult(drug_name=drug_name)

        # 优先：阿里云用药安全
        ali_data = self._alibaba_drug_info(drug_name)
        if ali_data and ali_data.get('data'):
            ali_info = ali_data['data']
            warnings = ali_info.get('warnings', []) if isinstance(ali_info, dict) else []
            if warnings:
                result.has_boxed_warning = any(
                    '禁' in str(w) or '忌' in str(w) or 'black' in str(w).lower()
                    for w in (warnings if isinstance(warnings, list) else [warnings])
                )
                result.warnings = warnings[:5] if isinstance(warnings, list) else [str(warnings)]
                result.contraindications = ali_info.get('contraindications', [])[:5]
                result.source = "Alibaba Cloud DrugSafe"
            return result

        # 回退：FDA（海外服务器）
        try:
            data = self._fda_search(f'openfda.brand_name:"{drug_name}"'
                                    f'+OR+openfda.generic_name:"{drug_name}"')
            if data and data.get('results'):
                label = data['results'][0]

                # 检查黑框警告
                bw = label.get('boxed_warning', [])
                if bw and isinstance(bw, list) and len(bw) > 0:
                    result.has_boxed_warning = True
                    result.boxed_warning_text = str(bw[0])[:2000]

                # 检查禁忌症
                ci = label.get('contraindications', [])
                if ci and isinstance(ci, list):
                    result.contraindications = [
                        str(c)[:200] for c in ci[:5] if str(c).strip()
                    ]

                # 检查警告
                warns = label.get('warnings', [])
                if warns and isinstance(warns, list):
                    result.warnings = [
                        str(w)[:200] for w in warns[:5] if str(w).strip()
                    ]

                # 不良反应
                ar = label.get('adverse_reactions', [])
                if ar and isinstance(ar, list):
                    result.adverse_reactions = [
                        str(a)[:200] for a in ar[:5] if str(a).strip()
                    ]

        except urllib.error.HTTPError as e:
            if e.code == 404:
                pass  # 药品不在 FDA 数据库中（非美国药或未收录）
        except Exception:
            pass  # 网络错误不影响主流程

        self._cache[drug_name] = (result, time.time())
        return result

    def check_contraindication(self, drug_name: str,
                               condition: str) -> Tuple[bool, str]:
        """
        检查特定药品-疾病组合是否有禁忌

        Returns:
            (is_contraindicated: bool, evidence_text: str)
        """
        result = self.check_boxed_warning(drug_name)
        condition_lower = condition.lower()

        # 在禁忌症中搜索
        for ci in result.contraindications:
            if condition_lower in ci.lower():
                return (True, f"FDA标签禁忌: {ci[:200]}")

        # 在黑框警告中搜索
        if result.boxed_warning_text:
            if condition_lower in result.boxed_warning_text.lower():
                return (True, f"FDA黑框警告: {result.boxed_warning_text[:300]}")

        # 在警告中搜索
        for w in result.warnings:
            if condition_lower in w.lower():
                return (False, f"FDA警告（非绝对禁忌）: {w[:200]}")

        return (False, "")

    # ==================== RxNav 药物名标准化 ====================

    def rxnav_normalize(self, drug_name: str) -> Optional[str]:
        """使用 RxNav 将药物名标准化为 RxNorm 规范名"""
        try:
            url = f"{self.RXNORM_BASE}/approximateTerm.json?term={urllib.parse.quote(drug_name)}&maxEntries=1"
            data = self._get_json(url)
            candidates = data.get('approximateGroup', {}).get('candidate', [])
            if candidates:
                return candidates[0].get('rxcui', '')
        except Exception:
            pass
        return None

    def rxnav_get_atc(self, rxnorm_id: str) -> List[str]:
        """通过 RxNorm ID 获取 ATC 编码"""
        try:
            url = f"{self.RXNORM_BASE}/rxclass/class/byRxcui.json?rxcui={rxnorm_id}&relaSource=ATC"
            data = self._get_json(url)
            classes = data.get('rxclassDrugInfoList', {}).get('rxclassDrugInfo', [])
            return [c.get('classId', '') for c in classes]
        except Exception:
            pass
        return []

    # ==================== PubMed 文献搜索 ====================

    def search_pubmed(self, query: str, max_results: int = 3) -> List[Dict]:
        """搜索 PubMed 文献（用于 Layer 3 RAG 溯源）"""
        try:
            # 搜索 PMID
            search_url = (f"{self.PUBMED_BASE}/esearch.fcgi?"
                         f"db=pubmed&retmax={max_results}&retmode=json&sort=relevance"
                         f"&term={urllib.parse.quote(query)}")
            data = self._get_json(search_url)
            ids = data.get('esearchresult', {}).get('idlist', [])

            if not ids:
                return []

            # 获取摘要
            fetch_url = (f"{self.PUBMED_BASE}/esummary.fcgi?"
                        f"db=pubmed&retmode=json&id={','.join(ids)}")
            summary = self._get_json(fetch_url)
            results = summary.get('result', {})

            articles = []
            for pid in ids:
                article = results.get(pid, {})
                if article:
                    articles.append({
                        "pmid": pid,
                        "title": article.get('title', ''),
                        "journal": article.get('source', ''),
                        "pubdate": article.get('pubdate', ''),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                    })
            return articles
        except Exception:
            pass
        return []

    # ==================== 综合安检 ====================

    def full_gate_check(self, drug_name: str,
                        patient_conditions: List[str] = None
                        ) -> Dict:
        """
        对单个药物执行完整安检流程

        Returns:
            {
                "passed": True/False,
                "drug": str,
                "fda": FDAResult dict,
                "contraindications_found": [...],
                "pubmed_evidence": [...] if failed
            }
        """
        result = {
            "passed": True,
            "drug": drug_name,
            "fda": None,
            "contraindications_found": [],
            "pubmed_evidence": [],
            "recommendation": "",
        }

        # Step 1: FDA 查询
        fda = self.check_boxed_warning(drug_name)
        result["fda"] = fda.to_dict()

        # Step 2: 黑框警告 = 直接熔断
        if fda.has_boxed_warning:
            result["passed"] = False
            result["recommendation"] = (
                f"FDA黑框警告: {drug_name} 存在严重安全风险"
            )
            # 补充 PubMed 引用
            result["pubmed_evidence"] = self.search_pubmed(
                f"{drug_name} boxed warning contraindication"
            )
            return result

        # Step 3: 患者状态禁忌检查
        for condition in (patient_conditions or []):
            is_ci, evidence = self.check_contraindication(drug_name, condition)
            if is_ci:
                result["contraindications_found"].append({
                    "condition": condition,
                    "evidence": evidence,
                })
                result["passed"] = False

        if not result["passed"]:
            result["recommendation"] = (
                f"FDA禁忌: {drug_name} 在 {', '.join(patient_conditions or [])} 患者中禁用"
            )
            result["pubmed_evidence"] = self.search_pubmed(
                f"{drug_name} contraindication {' '.join(patient_conditions or [])}"
            )

        return result

    # ==================== 内部工具 ====================

    def _fda_search(self, query: str) -> Optional[Dict]:
        """调用 OpenFDA API"""
        url = f"{self.OPENFDA_BASE}?search={query}&limit=1"
        return self._get_json(url)

    def _rate_limit(self):
        """保持每秒不超过 5 次请求"""
        elapsed = time.time() - self._last_request
        if elapsed < 0.2:
            time.sleep(0.2 - elapsed)
        self._last_request = time.time()

    def _get_json(self, url: str) -> Optional[Dict]:
        """带速率限制的 HTTP GET"""
        self._rate_limit()
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'PharmSandbox/1.0 (Clinical Decision Support)'
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception:
            return None
