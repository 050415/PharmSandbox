"""
RiskScorer 单元测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.decision.risk_scorer.scorer import RiskScorer


def test_basic():
    """基本功能测试"""
    scorer = RiskScorer()
    scorer._load_data()
    print(f"已加载 {len(scorer._drug_se_map)} 种药物的副作用数据")

    # 取第一个有数据的 CID 测试
    sample_cids = [cid for cid in scorer._drug_se_map if scorer._drug_se_map[cid]]
    assert len(sample_cids) > 0, "副作用数据为空"
    print(f"样本 CID: {sample_cids[:3]}")


def test_single_drug_risk():
    """单药风险评分测试"""
    scorer = RiskScorer()
    scorer._load_data()

    sample_cids = [cid for cid in scorer._drug_se_map if scorer._drug_se_map[cid]]
    cid = sample_cids[0]

    # 无患者信息
    result = scorer.calculate_drug_risk(cid)
    assert 'score' in result, "返回结果缺少 score"
    assert 'level' in result, "返回结果缺少 level"
    assert 0 <= result['score'] <= 100, f"评分超出范围: {result['score']}"
    print(f"单药风险 ({cid}): {result['score']} ({result['level']})")

    # 有患者信息
    patient = {'age': 75, 'gender': 'M', 'conditions': ['CKD', 'diabetes']}
    result2 = scorer.calculate_drug_risk(cid, patient)
    assert result2['score'] >= result['score'], "老年+CKD 患者风险应不低于无患者信息"
    print(f"老年患者风险: {result2['score']} ({result2['level']})")


def test_combination_risk():
    """多药联合风险评分测试"""
    scorer = RiskScorer()
    scorer._load_data()

    sample_cids = [cid for cid in scorer._drug_se_map if scorer._drug_se_map[cid]]
    cids = sample_cids[:3]

    result = scorer.calculate_combination_risk(cids)
    assert 'final_score' in result, "返回结果缺少 final_score"
    assert 'level' in result, "返回结果缺少 level"
    assert 'individual_risks' in result, "返回结果缺少 individual_risks"
    assert len(result['individual_risks']) == len(cids), "个别风险数量不匹配"
    print(f"联合风险 ({len(cids)} 药): {result['final_score']} ({result['level']})")


def test_patient_risk_factors():
    """患者风险因子测试"""
    scorer = RiskScorer()

    # 年轻健康人
    young = scorer._get_patient_risk_factors({'age': 30, 'conditions': []})
    # 老年肾病患者
    old_ckd = scorer._get_patient_risk_factors({'age': 80, 'conditions': ['CKD']})

    assert old_ckd['renal'] > young['renal'], "老年CKD患者肾脏风险应更高"
    print(f"年轻人肾脏因子: {young['renal']}, 老年CKD: {old_ckd['renal']}")


if __name__ == "__main__":
    print("=" * 50)
    print("  RiskScorer 测试")
    print("=" * 50)

    tests = [
        ("基本功能", test_basic),
        ("单药风险", test_single_drug_risk),
        ("联合风险", test_combination_risk),
        ("患者因子", test_patient_risk_factors),
    ]

    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name} 通过\n")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name} 失败: {e}\n")

    print(f"结果: {passed}/{len(tests)} 通过")
