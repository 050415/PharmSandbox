"""
PharmSandbox - 药盘推演智能体 Flask API
整合所有模块：GNN DDI推演、风险评分、替代药推荐、慢病续方
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

_project_root = Path(__file__).resolve().parent.parent.parent.parent
app = Flask(__name__,
            template_folder=str(_project_root / "frontend" / "templates"),
            static_folder=str(_project_root / "frontend" / "static"))
CORS(app)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# ==================== 模块初始化（通过 SandboxEngine 统一管理） ====================
from src.action.sandbox_engine import SandboxEngine

print("正在初始化 PharmSandbox...")
engine = SandboxEngine()

# 快捷引用（各端点直接使用引擎的子模块）
data_loader = engine.data_loader
recommender = engine.recommender
prescription_trigger = engine.prescription_trigger
ner = engine.ner
print("PharmSandbox 初始化完成！")


def get_json_body():
    """统一的请求体校验"""
    if not request.is_json:
        return None, (jsonify({'error': '请求必须为JSON格式，Content-Type: application/json'}), 400)
    data = request.json
    if data is None:
        return None, (jsonify({'error': '请求体不能为空'}), 400)
    return data, None


# ==================== API 路由 ====================

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    stats = data_loader.get_stats()
    return jsonify({
        'status': 'healthy',
        'service': 'PharmSandbox 药盘推演智能体',
        'version': '1.0.0',
        'data_stats': stats
    })


@app.route('/api/parse', methods=['POST'])
def parse_text():
    """
    自然语言解析（中文药名识别）
    请求体: {"text": "我吃了阿司匹林和华法林，有高血压"}
    """
    data, err = get_json_body()
    if err:
        return err

    text = data.get('text', '')
    if not text:
        return jsonify({'error': '请提供 text 字段'}), 400

    if ner is None:
        return jsonify({'error': 'NER模块未加载'}), 503

    result = ner.analyze(text)
    return jsonify({
        'text': text,
        'drugs': result.get('drugs', []),
        'diseases': result.get('diseases', []),
        'dosage': result.get('dosage', {}),
    })

# ==================== 功能①: DDI沙盘推演 ====================

@app.route('/api/ddi/check', methods=['POST'])
def check_ddi():
    """
    检查药物相互作用
    请求体: {"drugs": ["drug1", "drug2", ...], "patient_id": optional}
    """
    data, err = get_json_body()
    if err:
        return err
    drugs = data.get('drugs', [])
    
    if len(drugs) < 2:
        return jsonify({'error': '至少需要2种药物'}), 400
    
    # 检查所有药物对的交互
    interactions = []
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            # 使用SIDER数据检查共同副作用
            se_i = set(recommender.get_drug_side_effects(drugs[i]))
            se_j = set(recommender.get_drug_side_effects(drugs[j]))
            common_se = se_i & se_j
            
            if common_se:
                severity = 'high' if len(common_se) > 10 else ('moderate' if len(common_se) > 5 else 'mild')
                interactions.append({
                    'drug1': drugs[i],
                    'drug2': drugs[j],
                    'severity': severity,
                    'common_side_effects': list(common_se)[:10],
                    'common_count': len(common_se),
                    'warning': f"⚠️ {drugs[i]} + {drugs[j]} 共有 {len(common_se)} 种共同副作用"
                })
    
    return jsonify({
        'drugs': drugs,
        'interactions': interactions,
        'interaction_count': len(interactions),
        'has_risk': len(interactions) > 0
    })

# ==================== 功能②: 风险量化评分 ====================

@app.route('/api/risk/score', methods=['POST'])
def calculate_risk():
    """
    计算用药风险评分 (0-100)
    请求体: {
        "drugs": ["drug1", "drug2"],
        "patient": {"age": 65, "gender": "M", "conditions": ["CKD", "diabetes"]}
    }
    """
    data, err = get_json_body()
    if err:
        return err
    drugs = data.get('drugs', [])
    patient = data.get('patient', {})

    if not drugs:
        return jsonify({'error': '请提供药物列表'}), 400
    
    # 计算每种药物的基础风险
    drug_risks = []
    for drug in drugs:
        ses = recommender.get_drug_side_effects(drug)
        base_risk = min(len(ses) * 0.5, 40)  # 基础风险，最高40
        
        # 患者特异性调整
        age = patient.get('age', 50)
        conditions = [c.lower() for c in patient.get('conditions', [])]
        
        adjustment = 0
        adjustment_reasons = []
        
        # 年龄调整
        if age > 80:
            adjustment += 20
            adjustment_reasons.append("高龄患者(+20)")
        elif age > 65:
            adjustment += 10
            adjustment_reasons.append("老年患者(+10)")
        
        # 肾功能调整
        if any(c in conditions for c in ['ckd', 'renal', 'kidney', '肾功能不全', '肾病']):
            kidney_ses = [se for se in ses if any(k in se.lower() for k in ['renal', 'kidney', 'nephro'])]
            if kidney_ses:
                adjustment += 20
                adjustment_reasons.append(f"肾功能不全+肾毒性副作用(+20)")
        
        # 肝功能调整
        if any(c in conditions for c in ['liver', 'hepatic', '肝功能不全', '肝病']):
            liver_ses = [se for se in ses if any(k in se.lower() for k in ['liver', 'hepatic', 'hepat'])]
            if liver_ses:
                adjustment += 20
                adjustment_reasons.append(f"肝功能不全+肝毒性副作用(+20)")
        
        risk_score = min(100, base_risk + adjustment)
        
        drug_risks.append({
            'drug': drug,
            'base_risk': round(base_risk, 1),
            'adjustment': round(adjustment, 1),
            'total_risk': round(risk_score, 1),
            'side_effect_count': len(ses),
            'adjustment_reasons': adjustment_reasons
        })
    
    # 多药联合风险叠加
    if len(drugs) > 1:
        interaction_bonus = min(len(drugs) * 5, 20)
    else:
        interaction_bonus = 0
    
    total_risk = min(100, sum(d['total_risk'] for d in drug_risks) / len(drug_risks) + interaction_bonus)
    
    # 风险等级
    if total_risk >= 80:
        risk_level = "极度高危"
        risk_color = "#ff0000"
    elif total_risk >= 60:
        risk_level = "高危"
        risk_color = "#ff6600"
    elif total_risk >= 40:
        risk_level = "中等风险"
        risk_color = "#ffcc00"
    elif total_risk >= 20:
        risk_level = "低风险"
        risk_color = "#99cc00"
    else:
        risk_level = "安全"
        risk_color = "#00cc00"
    
    return jsonify({
        'total_risk_score': round(total_risk, 1),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'drug_risks': drug_risks,
        'interaction_bonus': interaction_bonus,
        'patient': patient,
        'recommendation': "建议咨询医生" if total_risk >= 60 else "可继续使用但需监测" if total_risk >= 40 else "用药风险较低"
    })

# ==================== 功能③: 替代药推荐 ====================

@app.route('/api/recommend/alternative', methods=['POST'])
def recommend_alternative():
    """
    推荐替代药物
    请求体: {
        "drug": "aspirin",
        "current_meds": ["warfarin"],
        "conditions": ["hypertension"]
    }
    """
    data, err = get_json_body()
    if err:
        return err
    drug = data.get('drug', '')
    current_meds = data.get('current_meds', [])
    conditions = data.get('conditions', [])
    
    if not drug:
        return jsonify({'error': '请提供需要替换的药物'}), 400
    
    alternatives = recommender.recommend_alternatives(
        drug_name=drug,
        current_meds=current_meds,
        patient_conditions=conditions,
        top_k=5
    )
    
    return jsonify({
        'original_drug': drug,
        'alternatives': alternatives,
        'count': len(alternatives),
        'explanation': "基于ATC分类+适应症匹配+副作用对比推荐"
    })

@app.route('/api/recommend/explain', methods=['POST'])
def explain_recommendation():
    """解释替代推荐的原因"""
    data, err = get_json_body()
    if err:
        return err
    original = data.get('original_drug', '')
    alternative = data.get('alternative', '')
    
    explanation = recommender.explain_recommendation(original, alternative)
    return jsonify(explanation)

# ==================== 功能④: 慢病续方 ====================

@app.route('/api/prescription/check', methods=['POST'])
def check_prescription():
    """
    检查处方续方需求
    请求体: {
        "prescriptions": [
            {"drug_name": "Metformin", "total_quantity": 60, "dose_per_time": 1, 
             "times_per_day": 2, "start_date": "2026-06-01"}
        ],
        "alert_days": 3
    }
    """
    data, err = get_json_body()
    if err:
        return err
    prescriptions = data.get('prescriptions', [])
    alert_days = data.get('alert_days', 3)
    
    results = prescription_trigger.batch_check(prescriptions, alert_days)
    
    return jsonify({
        'results': results,
        'total': len(results),
        'needs_refill': sum(1 for r in results if r['needs_refill'])
    })

@app.route('/api/prescription/refill-draft', methods=['POST'])
def generate_refill():
    """
    生成续方草稿
    请求体: {
        "prescription": {...},
        "patient": {"patient_id": "P001", "name": "张三", "age": 65, "gender": "M"}
    }
    """
    data, err = get_json_body()
    if err:
        return err
    prescription = data.get('prescription', {})
    patient = data.get('patient', {})
    
    draft = prescription_trigger.generate_refill_draft(prescription, patient)
    return jsonify(draft)

# ==================== 综合沙盘推演 ====================

@app.route('/api/sandbox/simulate', methods=['POST'])
def sandbox_simulate():
    """
    综合沙盘推演 - 委托给 SandboxEngine
    请求体: {
        "drugs": ["aspirin", "warfarin", "metformin"],
        "patient": {"age": 70, "gender": "M", "conditions": ["diabetes", "CKD"]},
        "prescriptions": [...]
    }
    """
    data, err = get_json_body()
    if err:
        return err
    drugs = data.get('drugs', [])
    patient = data.get('patient', {})
    prescriptions = data.get('prescriptions', [])

    if not drugs:
        return jsonify({'error': '请提供药物列表'}), 400

    result = engine.full_simulation(
        drugs=drugs,
        patient_info=patient,
        prescriptions=prescriptions,
    )
    result['timestamp'] = datetime.now().isoformat()
    return jsonify(result)


# ==================== UI 页面路由 ====================

@app.route('/sandbox')
def sandbox_page():
    """沙盘推演主页面"""
    return render_template('index.html')

@app.route('/patient')
def patient_page():
    """患者端 H5 页面"""
    return render_template('patient_h5.html')

@app.route('/doctor')
def doctor_page():
    """医生工作台页面"""
    return render_template('doctor.html')


# ==================== EHR 虚拟数据适配器 ====================

MOCK_PATIENTS = [
    {"id":"P1001","name":"张建国","age":72,"gender":"M","bed":"A-12",
     "conditions":["2型糖尿病","CKD 3期","高血压","骨关节炎"],
     "drugs":["metformin","lisinopril","ibuprofen","aspirin"],
     "labs":{"creatinine":168.5,"alt":42.3,"bun":12.8,"inr":1.1},
     "status":"high_risk","status_msg":"高危DDI拦截","status_color":"#A67F78",
     "admitted":"2026-06-15","summary":"CKD患者NSAIDs+ACEI三重交互风险",
     "refill_days":5},
    {"id":"P1002","name":"李秀兰","age":68,"gender":"F","bed":"B-03",
     "conditions":["心房颤动","高血压","骨质疏松"],
     "drugs":["warfarin","aspirin","amlodipine"],
     "labs":{"creatinine":88.2,"alt":28.1,"bun":6.5,"inr":2.8},
     "status":"high_risk","status_msg":"高危DDI拦截","status_color":"#A67F78",
     "admitted":"2026-06-20","summary":"华法林+阿司匹林出血风险极高",
     "refill_days":7},
    {"id":"P1003","name":"王德发","age":55,"gender":"M","bed":"A-05",
     "conditions":["高脂血症","冠心病"],
     "drugs":["atorvastatin","clopidogrel","omeprazole"],
     "labs":{"creatinine":95.0,"alt":55.8,"bun":7.2,"inr":1.0},
     "status":"refill","status_msg":"慢病续方申请","status_color":"#D4955C",
     "admitted":"2026-05-30","summary":"氯吡格雷即将用完，需续方",
     "refill_days":2},
    {"id":"P1004","name":"赵敏","age":45,"gender":"F","bed":"C-11",
     "conditions":["支气管哮喘","过敏性鼻炎"],
     "drugs":["albuterol","montelukast","prednisone"],
     "labs":{"creatinine":62.0,"alt":32.0,"bun":5.1,"inr":0.9},
     "status":"normal","status_msg":"常规复诊","status_color":"#8FA88A",
     "admitted":"2026-06-25","summary":"哮喘控制良好，继续维持治疗",
     "refill_days":14},
    {"id":"P1005","name":"钱伟","age":78,"gender":"M","bed":"B-07",
     "conditions":["心力衰竭","高血压","前列腺增生"],
     "drugs":["furosemide","carvedilol","lisinopril","tamsulosin"],
     "labs":{"creatinine":142.3,"alt":38.5,"bun":15.2,"inr":1.2},
     "status":"high_risk","status_msg":"高危DDI拦截","status_color":"#A67F78",
     "admitted":"2026-06-18","summary":"利尿剂+β阻+ACEI三联低血压+肾损风险",
     "refill_days":6},
    {"id":"P1006","name":"孙丽华","age":62,"gender":"F","bed":"A-09",
     "conditions":["2型糖尿病","周围神经病变"],
     "drugs":["metformin","gabapentin","insulin"],
     "labs":{"creatinine":78.5,"alt":25.0,"bun":5.8,"inr":1.0},
     "status":"refill","status_msg":"慢病续方申请","status_color":"#D4955C",
     "admitted":"2026-05-28","summary":"二甲双胍+胰岛素均需续方",
     "refill_days":1},
    {"id":"P1007","name":"周国强","age":81,"gender":"M","bed":"C-02",
     "conditions":["重度抑郁症","失眠","高血压"],
     "drugs":["sertraline","diazepam","losartan"],
     "labs":{"creatinine":105.0,"alt":45.2,"bun":9.3,"inr":1.1},
     "status":"high_risk","status_msg":"高危DDI拦截","status_color":"#A67F78",
     "admitted":"2026-06-22","summary":"SSRI+苯二氮卓叠加中枢抑制+跌倒风险",
     "refill_days":10},
    {"id":"P1008","name":"吴桂英","age":58,"gender":"F","bed":"B-11",
     "conditions":["类风湿关节炎","消化道溃疡史"],
     "drugs":["ibuprofen","omeprazole","methotrexate"],
     "labs":{"creatinine":72.0,"alt":68.3,"bun":6.9,"inr":1.0},
     "status":"high_risk","status_msg":"高危DDI拦截","status_color":"#A67F78",
     "admitted":"2026-06-19","summary":"MTX+NSAID肝毒性+肾毒性双重风险",
     "refill_days":3},
    {"id":"P1009","name":"郑建国","age":66,"gender":"M","bed":"A-14",
     "conditions":["高血压","高尿酸血症","痛风"],
     "drugs":["losartan","allopurinol","aspirin"],
     "labs":{"creatinine":110.0,"alt":35.0,"bun":8.8,"inr":1.0},
     "status":"refill","status_msg":"慢病续方申请","status_color":"#D4955C",
     "admitted":"2026-06-01","summary":"降压+降尿酸+抗血小板需长期管理",
     "refill_days":4},
    {"id":"P1010","name":"陈美玲","age":52,"gender":"F","bed":"C-08",
     "conditions":["甲状腺功能减退","高脂血症"],
     "drugs":["levothyroxine","atorvastatin","calcium"],
     "labs":{"creatinine":65.0,"alt":22.0,"bun":4.5,"inr":0.9},
     "status":"normal","status_msg":"常规复诊","status_color":"#8FA88A",
     "admitted":"2026-06-28","summary":"甲减+高脂血症稳定控制中"},
]

# 处方数据库（剂量、频次、时间、备注）
_PRESCRIPTION_DB = {
  "P1001":[{"drug":"metformin","dose":500,"unit":"mg","freq":"bid","time":"08:00,20:00","duration":30,"note":"餐后服用"},
           {"drug":"lisinopril","dose":10,"unit":"mg","freq":"qd","time":"08:00","duration":30,"note":"晨起空腹"},
           {"drug":"ibuprofen","dose":200,"unit":"mg","freq":"tid","time":"08:00,14:00,20:00","duration":7,"note":"饭后服用≤7天"},
           {"drug":"aspirin","dose":100,"unit":"mg","freq":"qd","time":"08:00","duration":30,"note":"早餐后"}],
  "P1002":[{"drug":"warfarin","dose":3,"unit":"mg","freq":"qd","time":"18:00","duration":30,"note":"每晚固定时间，监测INR"},
           {"drug":"aspirin","dose":100,"unit":"mg","freq":"qd","time":"08:00","duration":30,"note":"早餐后，关注牙龈出血"},
           {"drug":"amlodipine","dose":5,"unit":"mg","freq":"qd","time":"08:00","duration":30,"note":"晨起服用"}],
  "P1003":[{"drug":"atorvastatin","dose":20,"unit":"mg","freq":"qd","time":"20:00","duration":30,"note":"睡前服用"},
           {"drug":"clopidogrel","dose":75,"unit":"mg","freq":"qd","time":"08:00","duration":30,"note":"早餐后"},
           {"drug":"omeprazole","dose":20,"unit":"mg","freq":"qd","time":"08:00","duration":14,"note":"早餐前30分钟"}],
  "P1004":[{"drug":"albuterol","dose":2,"unit":"puff","freq":"prn","time":"as_needed","duration":30,"note":"必要时≤8次/日"},
           {"drug":"montelukast","dose":10,"unit":"mg","freq":"qd","time":"20:00","duration":30,"note":"睡前服用"},
           {"drug":"prednisone","dose":5,"unit":"mg","freq":"qd","time":"08:00","duration":7,"note":"渐减量勿骤停"}],
  "P1005":[{"drug":"furosemide","dose":40,"unit":"mg","freq":"qd","time":"08:00","duration":30,"note":"晨起，监测体重血压"},
           {"drug":"carvedilol","dose":12.5,"unit":"mg","freq":"bid","time":"08:00,20:00","duration":30,"note":"与食物同服"},
           {"drug":"lisinopril","dose":5,"unit":"mg","freq":"qd","time":"08:00","duration":30,"note":"监测血钾肌酐"},
           {"drug":"tamsulosin","dose":0.4,"unit":"mg","freq":"qd","time":"20:00","duration":30,"note":"睡前服用"}],
  "P1006":[{"drug":"metformin","dose":850,"unit":"mg","freq":"bid","time":"08:00,20:00","duration":30,"note":"餐后服用"},
           {"drug":"gabapentin","dose":300,"unit":"mg","freq":"tid","time":"08:00,14:00,20:00","duration":14,"note":"可致嗜睡"},
           {"drug":"insulin","dose":10,"unit":"IU","freq":"qd","time":"22:00","duration":30,"note":"睡前皮下注射，轮换部位"}],
  "P1007":[{"drug":"sertraline","dose":50,"unit":"mg","freq":"qd","time":"08:00","duration":30,"note":"早餐后，勿突然停药"},
           {"drug":"diazepam","dose":2.5,"unit":"mg","freq":"qd","time":"21:00","duration":7,"note":"短期，依赖风险"},
           {"drug":"losartan","dose":50,"unit":"mg","freq":"qd","time":"08:00","duration":30,"note":"监测血压血钾"}],
  "P1008":[{"drug":"ibuprofen","dose":400,"unit":"mg","freq":"bid","time":"08:00,20:00","duration":5,"note":"饭后，溃疡史慎用"},
           {"drug":"omeprazole","dose":20,"unit":"mg","freq":"bid","time":"07:30,19:30","duration":30,"note":"餐前30分钟"},
           {"drug":"methotrexate","dose":10,"unit":"mg","freq":"qw","time":"08:00(周一)","duration":90,"note":"每周一次，次日补叶酸5mg"}],
  "P1009":[{"drug":"losartan","dose":50,"unit":"mg","freq":"qd","time":"08:00","duration":30,"note":"晨起服用"},
           {"drug":"allopurinol","dose":100,"unit":"mg","freq":"qd","time":"08:00","duration":30,"note":"餐后，多饮水"},
           {"drug":"aspirin","dose":100,"unit":"mg","freq":"qd","time":"08:00","duration":30,"note":"早餐后"}],
  "P1010":[{"drug":"levothyroxine","dose":50,"unit":"mcg","freq":"qd","time":"06:00","duration":30,"note":"空腹，服药后30分钟进食"},
           {"drug":"atorvastatin","dose":10,"unit":"mg","freq":"qd","time":"20:00","duration":30,"note":"睡前服用"},
           {"drug":"calcium","dose":600,"unit":"mg","freq":"qd","time":"12:00","duration":30,"note":"午餐时服用，与甲状腺素间隔4h"}],
}


@app.route('/api/ehr/patients', methods=['GET'])
def ehr_patients():
    """返回待诊患者列表（含处方数据）"""
    enriched = []
    for p in MOCK_PATIENTS:
        p_copy = dict(p)
        p_copy['prescriptions'] = _PRESCRIPTION_DB.get(p['id'], [])
        enriched.append(p_copy)
    return jsonify({'patients': enriched, 'total': len(enriched)})


@app.route('/api/ehr/patient/<patient_id>', methods=['GET'])
def ehr_patient_detail(patient_id):
    """返回指定患者的完整EHR数据"""
    for p in MOCK_PATIENTS:
        if p['id'] == patient_id:
            p_copy = dict(p)
            p_copy['prescriptions'] = _PRESCRIPTION_DB.get(patient_id, [])
            return jsonify({'patient': p_copy, 'status': 'ok'})
    return jsonify({'error': '患者不存在', 'status': 'not_found'}), 404


@app.route('/api/ehr/lab_refresh', methods=['POST'])
def ehr_lab_refresh():
    """
    模拟物联网/新化验单数据刷新
    请求体: {"labs": {"creatinine":168.5,...}}
    """
    import random
    data, err = get_json_body()
    if err:
        return err
    labs = data.get('labs', {})
    refreshed = {}
    for key, val in labs.items():
        fluctuation = val * random.uniform(-0.25, 0.35)
        refreshed[key] = round(max(0, val + fluctuation), 1)
    spike = random.random() < 0.15
    target = None
    if spike:
        target = random.choice(list(refreshed.keys()))
        refreshed[target] = round(refreshed[target] * random.uniform(1.8, 3.5), 1)
    return jsonify({
        'original': labs,
        'refreshed': refreshed,
        'spike': spike,
        'spike_key': target if spike else None,
        'message': '⚠️ 检测到危急值！风险评分已更新' if spike else '化验指标已刷新'
    })


@app.route('/api/gate/check', methods=['POST'])
def api_gate_check():
    """异步 API 安检——不阻塞主推演，前端按需调用"""
    data, err = get_json_body()
    if err: return err
    drug = data.get('drug', '')
    conditions = data.get('conditions', [])
    if not drug:
        return jsonify({'error': '请提供 drug 字段'}), 400
    if engine.api_gate is None:
        return jsonify({'available': False, 'error': 'API Gate 未加载'})
    try:
        result = engine.api_gate.full_gate_check(drug, patient_conditions=conditions)
        return jsonify(result)
    except Exception as e:
        return jsonify({'available': True, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  PharmSandbox 药盘推演智能体")
    print("  服务地址: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
