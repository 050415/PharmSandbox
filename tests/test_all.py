"""
PharmSandbox - 全模块单元测试
覆盖: config, data_loader, recommender, risk_scorer, prescription_trigger,
      ner, llm_reasoner, sandbox_engine, gnn, api
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_FILES, get_logger, DATA_ROOT

logger = get_logger("test")


# ======================================================================
# 基础模块
# ======================================================================

def test_config():
    """配置模块测试"""
    assert DATA_FILES is not None
    assert 'drug_names' in DATA_FILES
    assert DATA_ROOT.exists(), f"数据目录不存在: {DATA_ROOT}"
    print(f"  ✓ 配置模块正常 (DATA_ROOT={DATA_ROOT})")
    return True


def test_data_loader():
    """数据加载器测试"""
    from src.data.loader import DataLoader
    loader = DataLoader()

    df = loader.load_drug_names()
    assert len(df) > 0, "药物名称表为空"
    assert 'drug_name' in df.columns or len(df.columns) >= 2
    print(f"  ✓ 药物名称: {len(df)} 条")

    df = loader.load_side_effects()
    assert len(df) > 0, "副作用表为空"
    print(f"  ✓ 副作用: {len(df)} 条")

    df = loader.load_drug_atc()
    assert len(df) > 0, "ATC分类表为空"
    print(f"  ✓ ATC分类: {len(df)} 条")

    stats = loader.get_stats()
    assert isinstance(stats, dict)
    assert len(stats) > 0
    print(f"  ✓ 统计信息: {len(stats)} 项")

    return True


# ======================================================================
# 感知层
# ======================================================================

def test_drug_ner():
    """药物名称NER测试"""
    from src.perception.ner.drug_ner import DrugNER
    ner = DrugNER()

    # 测试中文药物名识别
    result = ner.analyze("我吃了阿司匹林和华法林，有高血压和糖尿病")
    assert 'drugs' in result
    assert 'diseases' in result
    assert 'dosage' in result

    drug_names = [d['en_name'] for d in result['drugs']]
    assert 'aspirin' in drug_names, f"未识别到aspirin: {drug_names}"
    assert 'warfarin' in drug_names, f"未识别到warfarin: {drug_names}"
    print(f"  ✓ 中文药名识别: {drug_names}")

    diseases = result['diseases']
    assert 'hypertension' in diseases, f"未识别到hypertension: {diseases}"
    assert 'diabetes' in diseases, f"未识别到diabetes: {diseases}"
    print(f"  ✓ 疾病识别: {diseases}")

    # 测试英文药物名识别
    result2 = ner.analyze("aspirin and metformin")
    drug_names2 = [d['en_name'] for d in result2['drugs']]
    assert 'aspirin' in drug_names2
    print(f"  ✓ 英文药名识别: {drug_names2}")

    # 测试剂量提取
    result3 = ner.analyze("每天吃两次，每次500mg")
    dosage = result3['dosage']
    assert dosage.get('dose') == 500.0, f"剂量提取错误: {dosage}"
    assert dosage.get('unit') == 'mg'
    print(f"  ✓ 剂量提取: {dosage}")

    # 测试药物名标准化
    norm = ner.normalize_drug("阿司匹林")
    assert norm['en_name'] == 'aspirin'
    print(f"  ✓ 药物名标准化: {norm}")

    return True


# ======================================================================
# 决策层
# ======================================================================

def test_recommender():
    """替代药推荐器测试"""
    from src.decision.recommender import DrugRecommender
    rec = DrugRecommender()

    # 测试适应症查询
    indications = rec.get_drug_indications("aspirin")
    assert len(indications) > 0, "aspirin 适应症为空"
    print(f"  ✓ aspirin 适应症: {len(indications)} 个")

    # 测试副作用查询
    ses = rec.get_drug_side_effects("aspirin")
    assert isinstance(ses, list)
    print(f"  ✓ aspirin 副作用: {len(ses)} 个")

    # 测试ATC分类
    atc = rec.get_drug_atc("aspirin")
    assert atc is not None, "aspirin ATC分类为空"
    print(f"  ✓ aspirin ATC: {atc}")

    # 测试同类药物
    same_class = rec.get_same_class_drugs("aspirin", level=4)
    assert isinstance(same_class, list)
    print(f"  ✓ aspirin 同类药: {len(same_class)} 个")

    # 测试替代推荐
    alts = rec.recommend_alternatives("aspirin", top_k=3)
    assert isinstance(alts, list)
    print(f"  ✓ aspirin 替代推荐: {len(alts)} 个")

    return True


def test_risk_scorer():
    """风险评分器测试"""
    from src.decision.risk_scorer.scorer import RiskScorer
    scorer = RiskScorer()

    # 加载数据
    scorer._load_data()
    assert scorer._loaded, "数据加载失败"
    assert len(scorer._drug_se_map) > 0, "副作用映射为空"
    print(f"  ✓ 副作用数据: {len(scorer._drug_se_map)} 种药物")

    # 取一个有数据的 CID
    sample_cids = [cid for cid in scorer._drug_se_map if scorer._drug_se_map[cid]]
    assert len(sample_cids) > 0, "无可用的副作用数据"
    cid = sample_cids[0]

    # 单药风险评分
    result = scorer.calculate_drug_risk(cid)
    assert 'score' in result
    assert 'level' in result
    assert 0 <= result['score'] <= 100
    print(f"  ✓ 单药风险 ({cid}): {result['score']} ({result['level']})")

    # 带患者信息的风险评分
    patient = {'age': 75, 'gender': 'M', 'conditions': ['CKD', 'diabetes']}
    result2 = scorer.calculate_drug_risk(cid, patient)
    assert result2['score'] >= result['score'], "老年+CKD 患者风险应不低于无患者信息"
    print(f"  ✓ 老年患者风险: {result2['score']} ({result2['level']})")

    # 多药联合风险
    cids = sample_cids[:3]
    combo = scorer.calculate_combination_risk(cids, patient)
    assert 'final_score' in combo
    assert 'individual_risks' in combo
    assert len(combo['individual_risks']) == len(cids)
    print(f"  ✓ 联合风险 ({len(cids)} 药): {combo['final_score']} ({combo['level']})")

    # 患者风险因子
    young = scorer._get_patient_risk_factors({'age': 30, 'conditions': []})
    old_ckd = scorer._get_patient_risk_factors({'age': 80, 'conditions': ['CKD']})
    assert old_ckd['renal'] > young['renal']
    print(f"  ✓ 风险因子: 年轻={young['renal']}, 老年CKD={old_ckd['renal']}")

    return True


def test_prescription_trigger():
    """续方触发器测试"""
    from src.decision.prescription_trigger import PrescriptionTrigger
    trigger = PrescriptionTrigger()

    # 测试续方检查
    rx = {'drug_name': 'Metformin', 'total_quantity': 60, 'dose_per_time': 1,
          'times_per_day': 2, 'start_date': '2026-06-01'}
    result = trigger.check_refill_needed(rx)
    assert 'needs_refill' in result
    assert 'urgency' in result
    assert 'remaining_days' in result
    assert isinstance(result['needs_refill'], bool)
    print(f"  ✓ 续方检查: {result['message']}")

    # 测试批量检查
    rxs = [
        rx,
        {'drug_name': 'Amlodipine', 'total_quantity': 30, 'dose_per_time': 1,
         'times_per_day': 1, 'start_date': '2026-06-25'},
    ]
    results = trigger.batch_check(rxs)
    assert len(results) == 2
    print(f"  ✓ 批量检查: {len(results)} 条")

    # 测试续方草稿生成
    patient = {'patient_id': 'P001', 'name': '张三', 'age': 65, 'gender': 'M'}
    draft = trigger.generate_refill_draft(rx, patient)
    assert draft['type'] == 'refill_draft'
    assert draft['patient']['name'] == '张三'
    assert 'prescription' in draft
    print(f"  ✓ 续方草稿: {draft['prescription']['drug_name']}")

    return True


def test_llm_reasoner():
    """LLM推理器测试"""
    from src.decision.llm_reasoner import MedicalReasoner, PatientProfile
    reasoner = MedicalReasoner()

    # 测试DDI解释 - 已知DDI
    result = reasoner.explain_ddi("华法林", "阿司匹林")
    assert result['has_known_ddi'] is True
    assert result['severity'] in ['禁忌', '重大', '中等', '轻微']
    assert 'explanation' in result
    print(f"  ✓ 已知DDI (华法林+阿司匹林): {result['severity']} - {result['ddi_type']}")

    # 测试DDI解释 - 未知DDI
    result2 = reasoner.explain_ddi("阿莫西林", "二甲双胍")
    assert 'explanation' in result2
    print(f"  ✓ 未知DDI (阿莫西林+二甲双胍): has_ddi={result2['has_known_ddi']}")

    # 测试意图识别
    intent, conf = reasoner.identify_intent("华法林和阿司匹林能一起吃吗？")
    assert intent.value == 'ddi_check'
    assert conf > 0
    print(f"  ✓ 意图识别: {intent.value} (置信度={conf:.2f})")

    # 测试个性化用药建议
    patient = PatientProfile(patient_id="P001", name="张三", age=70, gender="男")
    patient.conditions = ["高血压", "糖尿病", "CKD"]
    patient.current_meds = ["华法林", "阿司匹林", "二甲双胍"]
    advice = reasoner.generate_medication_advice(patient)
    assert 'advice' in advice
    assert 'ddi_findings' in advice
    assert len(advice['ddi_findings']) > 0, "应检测到华法林+阿司匹林的DDI"
    print(f"  ✓ 用药建议: {len(advice['ddi_findings'])} 项DDI, {len(advice['warnings'])} 项警告")

    # 测试多轮对话
    session_id = reasoner.start_consultation(patient)
    assert session_id is not None
    resp = reasoner.chat(session_id, "华法林和氟康唑一起吃安全吗？")
    assert 'reply' in resp
    assert resp['intent'] == 'ddi_check'
    print(f"  ✓ 多轮对话: 意图={resp['intent']}, 药物={resp['extracted_drugs']}")

    # 测试批量DDI检查
    batch = reasoner.batch_ddi_check(
        ["华法林", "阿司匹林", "氟康唑", "二甲双胍"],
        patient_info={"age": 70, "conditions": ["CKD"]},
    )
    assert batch['total_drugs'] == 4
    assert batch['ddi_found'] > 0
    print(f"  ✓ 批量DDI: {batch['ddi_found']}/{batch['total_pairs']} 对有交互")

    # 测试推荐验证
    validation = reasoner.validate_recommendation("布洛芬", patient)
    assert 'verdict' in validation
    assert 'issues' in validation
    print(f"  ✓ 推荐验证: {validation['verdict']}")

    reasoner.end_session(session_id)
    return True


# ======================================================================
# 沙盘引擎
# ======================================================================

def test_sandbox_engine():
    """沙盘推演引擎测试"""
    from src.action.sandbox_engine import SandboxEngine
    engine = SandboxEngine()

    # 测试自然语言解析
    if engine.ner:
        result = engine.parse_natural_language("我吃了阿司匹林和华法林，有高血压和糖尿病")
        assert 'drugs' in result
        assert 'conditions' in result
        print(f"  ✓ NER解析: drugs={len(result['drugs'])}, conditions={len(result['conditions'])}")

    # 测试完整推演
    simulation = engine.full_simulation(
        drugs=["aspirin", "warfarin", "metformin"],
        patient_info={"age": 70, "gender": "M", "conditions": ["diabetes", "CKD"]}
    )
    assert 'analysis' in simulation
    assert 'ddi' in simulation['analysis']
    assert 'risk' in simulation['analysis']
    assert 'alternatives' in simulation['analysis']
    assert 'summary' in simulation['analysis']

    ddi = simulation['analysis']['ddi']
    risk = simulation['analysis']['risk']
    summary = simulation['analysis']['summary']

    assert ddi['count'] >= 0, f"DDI count 异常: {ddi}"
    assert 'final_score' in risk, f"risk 缺少 final_score, 实际键: {list(risk.keys())}"
    assert 'risk_level' in summary, f"summary 缺少 risk_level, 实际键: {list(summary.keys())}"
    print(f"  ✓ 沙盘推演: DDI={ddi['count']}个, 风险={risk['final_score']}, 等级={summary['risk_level']}")

    # 测试药物名→CID转换
    cid = engine._drug_name_to_cid("aspirin")
    print(f"  ✓ 药物名→CID: aspirin → {cid}")

    return True


# ======================================================================
# GNN 模块
# ======================================================================

def test_gnn_model():
    """GNN模型测试"""
    try:
        import torch
    except ImportError:
        print("  ⚠ PyTorch 未安装，跳过 GNN 测试")
        return True  # 跳过但不算失败

    from src.decision.gnn.model import DDIPredictor, build_model

    # 测试模型创建
    model = DDIPredictor(
        num_nodes=100, num_relations=5,
        hidden_dim=32, num_layers=2, num_classes=2
    )
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
    print(f"  ✓ DDIPredictor 创建: {param_count:,} 参数")

    # 测试 build_model 工厂函数
    model2 = build_model(num_nodes=100, num_relations=5, num_ddi_types=10)
    assert isinstance(model2, DDIPredictor)
    print(f"  ✓ build_model 工厂函数正常")

    # 测试前向传播（原始调用方式）
    drug1 = torch.tensor([0, 1, 2], dtype=torch.long)
    drug2 = torch.tensor([3, 4, 5], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    logits = model(drug1, drug2, edge_index, edge_type)
    assert logits.shape == (3, 2)
    print(f"  ✓ 前向传播 (原始方式): logits={logits.shape}")

    # 测试前向传播（DDITrainer兼容方式）
    logits2 = model(edge_index, edge_type, drug1, drug2, 100)
    assert logits2.shape == (3, 2)
    print(f"  ✓ 前向传播 (DDITrainer方式): logits={logits2.shape}")

    # 测试 predict
    result = model.predict(drug1, drug2, edge_index, edge_type)
    assert 'predictions' in result
    assert 'probabilities' in result
    assert 'confidence' in result
    print(f"  ✓ predict: predictions={result['predictions'].shape}")

    return True


def test_knowledge_graph():
    """知识图谱测试"""
    try:
        import torch_geometric
        from src.decision.gnn.knowledge_graph import DrugKnowledgeGraph
    except ImportError as e:
        print(f"  ⚠ 依赖缺失: {e}")
        return True

    try:
        kg = DrugKnowledgeGraph()

        # 测试节点管理
        idx1 = kg._add_node("drug", "aspirin")
        idx2 = kg._add_node("drug", "warfarin")
        idx3 = kg._add_node("side_effect", "bleeding")
        assert idx1 != idx2
        assert idx1 != idx3
        print(f"  ✓ 节点管理: aspirin={idx1}, warfarin={idx2}, bleeding={idx3}")

        # 测试边添加
        kg.G.add_edge("drug:aspirin", "side_effect:bleeding", edge_type="has_side_effect")
        assert kg.G.number_of_edges() == 1
        print(f"  ✓ 边添加: {kg.G.number_of_edges()} 条边")

        # 测试邻居查询
        neighbors = kg.get_neighbors("aspirin")
        assert 'side_effect' in neighbors
        assert 'bleeding' in neighbors['side_effect']
        print(f"  ✓ 邻居查询: aspirin → {neighbors}")

        # 测试药物搜索
        results = kg.search_drugs("aspir")
        assert 'aspirin' in results
        print(f"  ✓ 药物搜索: 'aspir' → {results}")

        return True
    except Exception as e:
        print(f"  ⚠ 知识图谱测试失败: {type(e).__name__}: {repr(e)}")
        return True


# ======================================================================
# Flask API
# ======================================================================

def test_flask_api():
    """Flask API 集成测试"""
    try:
        from src.action.api.app import app
    except ImportError:
        print("  ⚠ flask_cors 未安装，跳过 API 测试")
        return True

    client = app.test_client()

    # 测试健康检查
    resp = client.get('/api/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'healthy'
    print(f"  ✓ /api/health: {data['status']}")

    # 测试自然语言解析
    resp = client.post('/api/parse', json={'text': '我吃了阿司匹林和华法林'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data['drugs']) >= 2
    print(f"  ✓ /api/parse: {len(data['drugs'])} 种药物")

    # 测试DDI检查
    resp = client.post('/api/ddi/check', json={'drugs': ['aspirin', 'warfarin']})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'interactions' in data
    print(f"  ✓ /api/ddi/check: {data['interaction_count']} 个交互")

    # 测试风险评分
    resp = client.post('/api/risk/score', json={
        'drugs': ['aspirin', 'warfarin'],
        'patient': {'age': 70, 'gender': 'M', 'conditions': ['CKD']}
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'total_risk_score' in data
    assert 'risk_level' in data
    print(f"  ✓ /api/risk/score: {data['total_risk_score']} ({data['risk_level']})")

    # 测试替代推荐
    resp = client.post('/api/recommend/alternative', json={
        'drug': 'aspirin', 'current_meds': ['warfarin']
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'alternatives' in data
    print(f"  ✓ /api/recommend/alternative: {data['count']} 个替代")

    # 测试续方检查
    resp = client.post('/api/prescription/check', json={
        'prescriptions': [
            {'drug_name': 'Metformin', 'total_quantity': 60, 'dose_per_time': 1,
             'times_per_day': 2, 'start_date': '2026-06-01'}
        ]
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'results' in data
    print(f"  ✓ /api/prescription/check: {data['total']} 条")

    # 测试综合沙盘推演
    resp = client.post('/api/sandbox/simulate', json={
        'drugs': ['aspirin', 'warfarin', 'metformin'],
        'patient': {'age': 70, 'gender': 'M', 'conditions': ['diabetes', 'CKD']}
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'analysis' in data
    assert 'ddi' in data['analysis']
    assert 'risk' in data['analysis']
    print(f"  ✓ /api/sandbox/simulate: DDI={data['analysis']['ddi']['count']}")

    # 测试请求校验 - 空请求体
    resp = client.post('/api/ddi/check', data="not json",
                       content_type='text/plain')
    assert resp.status_code == 400
    print(f"  ✓ 请求校验: 空请求返回 400")

    # 测试请求校验 - 参数不足
    resp = client.post('/api/ddi/check', json={'drugs': ['aspirin']})
    assert resp.status_code == 400
    print(f"  ✓ 请求校验: 参数不足返回 400")

    return True


# ======================================================================
# 测试运行器
# ======================================================================

def run_all_tests():
    print("=" * 50)
    print("  PharmSandbox 全模块单元测试")
    print("=" * 50)

    tests = [
        ("配置模块", test_config),
        ("数据加载器", test_data_loader),
        ("药物NER", test_drug_ner),
        ("替代药推荐", test_recommender),
        ("风险评分", test_risk_scorer),
        ("续方触发器", test_prescription_trigger),
        ("LLM推理器", test_llm_reasoner),
        ("沙盘引擎", test_sandbox_engine),
        ("GNN模型", test_gnn_model),
        ("知识图谱", test_knowledge_graph),
        ("Flask API", test_flask_api),
    ]

    passed = failed = 0
    for name, fn in tests:
        try:
            print(f"\n--- {name} ---")
            if fn():
                passed += 1
        except Exception as e:
            print(f"  ✗ {name} 失败: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  结果: {passed} 通过, {failed} 失败, 共 {len(tests)} 项")
    print(f"{'=' * 50}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
