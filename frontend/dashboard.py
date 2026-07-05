"""
PharmSandbox 药盘推演 — Streamlit 高级仪表板
============================================
多页面应用：
  🏠 首页 · 综合沙盘推演
  📊 数据探索 · SIDER / DrugCentral 浏览
  ℹ️ 关于 · 项目说明
"""

import sys
import os
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# ── 项目根目录 ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# ═══════════════════════════════════════════════════════════════
#  全局配置
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="💊 药盘推演 PharmSandbox",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自定义 CSS ───────────────────────────────────────────────
st.markdown("""
<style>
/* 全局字体 & 背景 */
.reportview-container .main .block-container { padding-top: 1rem; }
/* 风险卡片 */
.risk-card {
    border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12); color: #fff;
}
.risk-critical { background: linear-gradient(135deg, #ff0000, #cc0000); }
.risk-high     { background: linear-gradient(135deg, #ff6600, #e65c00); }
.risk-moderate { background: linear-gradient(135deg, #f0ad4e, #ec971f); }
.risk-low      { background: linear-gradient(135deg, #5cb85c, #4cae4c); }
.risk-safe     { background: linear-gradient(135deg, #3498db, #2980b9); }
/* DDI 卡片 */
.ddi-card {
    border-left: 5px solid #e74c3c; background: #fdf2f2;
    border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
}
.ddi-card-moderate {
    border-left: 5px solid #f39c12; background: #fef9e7;
    border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
}
.ddi-card-mild {
    border-left: 5px solid #27ae60; background: #eafaf1;
    border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
}
/* 替代药推荐 */
.alt-card {
    border: 1px solid #3498db; border-radius: 10px;
    padding: 1rem; margin: 0.4rem 0; background: #f8f9fa;
}
/* 续方提醒 */
.refill-critical { border-left: 5px solid #e74c3c; background: #fdedec; padding: 0.8rem; border-radius: 6px; margin: 0.3rem 0; }
.refill-warning  { border-left: 5px solid #f1c40f; background: #fef9e7; padding: 0.8rem; border-radius: 6px; margin: 0.3rem 0; }
.refill-ok       { border-left: 5px solid #2ecc71; background: #eafaf1; padding: 0.8rem; border-radius: 6px; margin: 0.3rem 0; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  后端模块懒加载
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_backend():
    """懒加载后端模块（缓存一次）"""
    from src.data.loader import DataLoader
    from src.decision.recommender import DrugRecommender
    from src.decision.prescription_trigger import PrescriptionTrigger

    loader = DataLoader()
    recommender = DrugRecommender()
    trigger = PrescriptionTrigger()
    return loader, recommender, trigger


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════

# 常见药物中英文映射
DRUG_ALIASES = {
    "阿司匹林": "aspirin", "华法林": "warfarin", "二甲双胍": "metformin",
    "氨氯地平": "amlodipine", "阿托伐他汀": "atorvastatin", "氯吡格雷": "clopidogrel",
    "奥美拉唑": "omeprazole", "美托洛尔": "metoprolol", "缬沙坦": "valsartan",
    "布洛芬": "ibuprofen", "对乙酰氨基酚": "acetaminophen", "头孢氨苄": "cefalexin",
    "左氧氟沙星": "levofloxacin", "胰岛素": "insulin", "辛伐他汀": "simvastatin",
    "利伐沙班": "rivaroxaban", "阿哌沙班": "apixaban", "氯沙坦": "losartan",
    "氢氯噻嗪": "hydrochlorothiazide", "呋塞米": "furosemide",
    "地高辛": "digoxin", "普萘洛尔": "propranolol", "硝苯地平": "nifedipine",
}

def parse_drug_input(text: str) -> list[str]:
    """从自然语言中提取药物名称"""
    # 去除常见前缀
    text = re.sub(r"我[在]?吃了?|正在服用|使用了?|服用了?", "", text)
    text = re.sub(r"[，,、和+与]+", " ", text)
    text = re.sub(r"[。.!！?？]", "", text)

    drugs = []
    tokens = text.strip().split()
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        # 中文别名
        if token in DRUG_ALIASES:
            drugs.append(DRUG_ALIASES[token])
        else:
            drugs.append(token.lower())
    return [d for d in drugs if d]


def risk_css_class(level: str) -> str:
    mapping = {
        "极度高危": "risk-critical", "高危": "risk-high",
        "中等风险": "risk-moderate", "低风险": "risk-low", "安全": "risk-safe",
    }
    return mapping.get(level, "risk-safe")


def risk_color(level: str) -> str:
    mapping = {
        "极度高危": "#ff0000", "高危": "#ff6600",
        "中等风险": "#f0ad4e", "低风险": "#5cb85c", "安全": "#3498db",
    }
    return mapping.get(level, "#3498db")


# ── 风险雷达图 ───────────────────────────────────────────────
ORGANS_CN = {
    "renal": "肾脏", "hepatic": "肝脏", "cardiac": "心脏",
    "hematologic": "血液", "respiratory": "呼吸", "neurologic": "神经",
}

def build_radar_chart(organ_risks: dict) -> go.Figure:
    """构建器官风险雷达图"""
    categories = [ORGANS_CN.get(k, k) for k in organ_risks.keys()]
    values = list(organ_risks.values())
    # 闭合
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill="toself",
        name="器官风险", line=dict(color="#e74c3c", width=2),
        fillcolor="rgba(231,76,60,0.25)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, margin=dict(t=30, b=30, l=30, r=30), height=350,
    )
    return fig


# ── 知识图谱 ─────────────────────────────────────────────────
def build_drug_knowledge_graph(drugs: list[str], recommender) -> nx.Graph:
    """构建药物知识图谱"""
    G = nx.Graph()
    for drug in drugs:
        G.add_node(drug, node_type="drug")
        ses = recommender.get_drug_side_effects(drug)
        # 取前 8 个副作用
        for se in ses[:8]:
            G.add_node(se, node_type="side_effect")
            G.add_edge(drug, se, edge_type="causes")
    # 药物间 DDI 边
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            se_i = set(recommender.get_drug_side_effects(drugs[i]))
            se_j = set(recommender.get_drug_side_effects(drugs[j]))
            common = se_i & se_j
            if common:
                G.add_edge(drugs[i], drugs[j], edge_type="ddi",
                           weight=len(common))
    return G


def render_knowledge_graph_html(G: nx.Graph) -> str:
    """将 NetworkX 图渲染为 HTML (vis-network.js)"""
    nodes_js = []
    for n, data in G.nodes(data=True):
        nt = data.get("node_type", "drug")
        if nt == "drug":
            color = "#e74c3c"
            shape = "dot"
            size = 30
        else:
            color = "#3498db"
            shape = "box"
            size = 15
        label = n if len(n) < 25 else n[:22] + "…"
        nodes_js.append(f'{{id:"{n}",label:"{label}",color:"{color}",shape:"{shape}",size:{size}}}')

    edges_js = []
    for u, v, data in G.edges(data=True):
        et = data.get("edge_type", "")
        if et == "ddi":
            color = "#e74c3c"
            width = data.get("weight", 1) * 1.5
            dashes = "false"
            title = f"DDI ({data.get('weight', 0)} 共同副作用)"
        else:
            color = "#95a5a6"
            width = 1
            dashes = "true"
            title = "导致"
        edges_js.append(
            f'{{from:"{u}",to:"{v}",color:"{color}",width:{width},dashes:{dashes},title:"{title}"}}'
        )

    html = f"""
    <div id="kg-container" style="width:100%;height:450px;border:1px solid #ddd;border-radius:8px;"></div>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script>
    (function(){{
      var nodes = new vis.DataSet([{','.join(nodes_js)}]);
      var edges = new vis.DataSet([{','.join(edges_js)}]);
      var container = document.getElementById('kg-container');
      var data = {{nodes: nodes, edges: edges}};
      var options = {{
        physics: {{solver: 'forceAtlas2Based', stabilization: {{iterations: 150}}}},
        interaction: {{hover: true, tooltipDelay: 100}},
        edges: {{smooth: {{type: 'continuous'}}}}
      }};
      new vis.Network(container, data, options);
    }})();
    </script>
    """
    return html


# ── 副作用频率分布图 ─────────────────────────────────────────
def build_se_frequency_chart(se_list: list[str]) -> go.Figure:
    """副作用词频分布图"""
    from collections import Counter
    # 提取关键词类别
    keywords = {
        "出血": ["bleed", "hemorrhag", "ecchymosis"],
        "胃肠道": ["nausea", "vomit", "diarrhoea", "diarrhea", "dyspepsia", "abdominal"],
        "神经": ["headache", "dizziness", "somnolence", "confusion", "tremor"],
        "皮肤": ["rash", "pruritus", "urticaria", "dermatitis"],
        "肝毒性": ["hepat", "liver", "bilirubin", "transaminase"],
        "肾毒性": ["renal", "kidney", "nephro", "creatinine"],
        "心血管": ["cardiac", "arrhythmia", "tachycardia", "hypotension"],
        "呼吸": ["dyspnea", "respiratory", "cough", "pneumon"],
        "其他": [],
    }
    counts = defaultdict(int)
    for se in se_list:
        se_lower = se.lower()
        matched = False
        for cat, kws in keywords.items():
            if kws and any(k in se_lower for k in kws):
                counts[cat] += 1
                matched = True
                break
        if not matched:
            counts["其他"] += 1

    cats = list(counts.keys())
    vals = list(counts.values())
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#1abc9c", "#9b59b6",
              "#3498db", "#2ecc71", "#34495e", "#95a5a6"]

    fig = go.Figure(go.Bar(
        x=cats, y=vals,
        marker_color=colors[:len(cats)],
        text=vals, textposition="outside",
    ))
    fig.update_layout(
        title="副作用类别分布", xaxis_title="副作用类别",
        yaxis_title="数量", height=350, margin=dict(t=40, b=40),
    )
    return fig


# ── 用药时间线 ───────────────────────────────────────────────
def build_medication_timeline(prescriptions: list[dict]) -> go.Figure:
    """患者用药时间线甘特图"""
    rows = []
    for rx in prescriptions:
        name = rx.get("drug_name", "Unknown")
        start = rx.get("start_date", datetime.now().strftime("%Y-%m-%d"))
        dose = rx.get("dose_per_time", 1)
        times = rx.get("times_per_day", 1)
        qty = rx.get("total_quantity", 30)
        total_days = max(1, qty / max(dose * times, 1))
        start_dt = datetime.strptime(str(start)[:10], "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=total_days)
        remaining = max(0, total_days - (datetime.now() - start_dt).days)
        status = "已完成" if remaining <= 0 else ("即将用完" if remaining <= 3 else "用药中")
        rows.append(dict(Task=name, Start=start_dt, Finish=end_dt, Status=status))

    if not rows:
        return go.Figure()

    df = pd.DataFrame(rows)
    color_map = {"已完成": "#95a5a6", "即将用完": "#e74c3c", "用药中": "#2ecc71"}
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Status",
                      color_discrete_map=color_map, title="用药时间线")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(250, len(rows) * 50 + 100), margin=dict(t=40, b=40))
    return fig


# ═══════════════════════════════════════════════════════════════
#  侧边栏 — 患者信息
# ═══════════════════════════════════════════════════════════════

def render_sidebar():
    """侧边栏：患者信息输入"""
    with st.sidebar:
        st.markdown("## 👤 患者信息")

        age = st.number_input("年龄", min_value=0, max_value=120, value=65, step=1)
        gender = st.selectbox("性别", ["男", "女", "其他"])

        st.markdown("### 🏥 疾病史")
        conditions_input = st.text_area(
            "输入疾病史（每行一条）",
            value="糖尿病\n高血压",
            height=100,
            help="例如：CKD、糖尿病、高血压、冠心病",
        )
        conditions = [c.strip() for c in conditions_input.strip().split("\n") if c.strip()]

        st.markdown("### 🧪 化验值")
        col1, col2 = st.columns(2)
        with col1:
            creatinine = st.number_input("肌酐 (μmol/L)", 0.0, 1500.0, 88.0, step=1.0)
            alt = st.number_input("ALT (U/L)", 0.0, 1000.0, 25.0, step=1.0)
        with col2:
            bun = st.number_input("BUN (mmol/L)", 0.0, 50.0, 5.5, step=0.1)
            inr = st.number_input("INR", 0.5, 10.0, 1.0, step=0.1)

        lab_values = {
            "creatinine": creatinine, "bun": bun,
            "alt": alt, "inr": inr,
        }

        st.markdown("---")
        st.markdown("### ⚙️ 设置")
        alert_days = st.slider("续方提醒天数", 1, 14, 3)

        patient_info = {
            "age": age,
            "gender": gender,
            "conditions": conditions,
            "lab_values": lab_values,
        }

        return patient_info, alert_days


# ═══════════════════════════════════════════════════════════════
#  页面 🏠 首页 — 综合沙盘推演
# ═══════════════════════════════════════════════════════════════

def page_home():
    """首页：综合沙盘推演"""
    st.markdown("# 💊 药盘推演 PharmSandbox")
    st.markdown("> 基于 SIDER / DrugCentral / MIMIC-IV 数据的药物相互作用沙盘推演系统")

    patient_info, alert_days = render_sidebar()

    # ── 药物输入 ──────────────────────────────────────────────
    st.markdown("## 💉 药物输入")
    col_input, col_example = st.columns([3, 1])
    with col_input:
        drug_text = st.text_input(
            "输入药物名称（支持自然语言）",
            value="我吃了阿司匹林和华法林",
            help="支持格式：阿司匹林、华法林 / aspirin, warfarin / 我吃了阿司匹林和华法林",
        )
    with col_example:
        st.markdown("**示例输入：**")
        st.code("我吃了阿司匹林和华法林")
        st.code("aspirin, metformin, warfarin")

    drugs = parse_drug_input(drug_text)
    if not drugs:
        st.warning("⚠️ 未识别到有效药物名称，请输入至少 2 种药物。")
        return

    st.success(f"✅ 已识别 {len(drugs)} 种药物：**{', '.join(drugs)}**")

    if len(drugs) < 2:
        st.info("💡 输入至少 2 种药物以启用 DDI 检测和沙盘推演。")
        return

    # ── 加载后端 ──────────────────────────────────────────────
    try:
        loader, recommender, trigger = load_backend()
    except Exception as e:
        st.error(f"❌ 后端加载失败: {e}")
        st.info("请确保已安装依赖并配置好数据目录。将使用模拟数据演示。")
        recommender = None
        trigger = None

    # ══════════════════════════════════════════════════════════
    #  DDI 冲突检测
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 🔍 DDI 冲突检测")

    interactions = []
    if recommender:
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                try:
                    se_i = set(recommender.get_drug_side_effects(drugs[i]))
                    se_j = set(recommender.get_drug_side_effects(drugs[j]))
                    common = se_i & se_j
                    if common:
                        cnt = len(common)
                        severity = "high" if cnt > 10 else ("moderate" if cnt > 5 else "mild")
                        interactions.append({
                            "drug1": drugs[i], "drug2": drugs[j],
                            "severity": severity, "common_count": cnt,
                            "common_effects": list(common)[:10],
                        })
                except Exception:
                    pass

    if interactions:
        for ix in interactions:
            css = "ddi-card" if ix["severity"] == "high" else (
                "ddi-card-moderate" if ix["severity"] == "moderate" else "ddi-card-mild")
            sev_cn = {"high": "🔴 高危", "moderate": "🟡 中等", "mild": "🟢 轻微"}
            st.markdown(f"""
            <div class="{css}">
                <b>{sev_cn[ix['severity']]}</b> &nbsp;
                <b>{ix['drug1'].upper()}</b> × <b>{ix['drug2'].upper()}</b>
                — 共有 <b>{ix['common_count']}</b> 种共同副作用
                <br><small>{'、'.join(ix['common_effects'])}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ 未检测到显著药物-药物相互作用（DDI）")

    # ══════════════════════════════════════════════════════════
    #  风险雷达图
    # ══════════════════════════════════════════════════════════
    st.markdown("## ⚠️ 风险量化评分")

    organ_risks = {k: 0.0 for k in ORGANS_CN}
    risk_score = 0
    risk_level = "安全"

    if recommender:
        all_se = []
        for drug in drugs:
            ses = recommender.get_drug_side_effects(drug)
            all_se.extend(ses)

        # 计算器官风险
        organ_kw = {
            "renal": ["renal", "kidney", "nephro", "creatinine"],
            "hepatic": ["hepatic", "liver", "hepat", "bilirubin", "transaminase"],
            "cardiac": ["cardiac", "heart", "arrhythmia", "qt", "tachycardia"],
            "hematologic": ["bleed", "hemorrhag", "thrombo", "platelet", "anemia"],
            "respiratory": ["respiratory", "pulmonary", "broncho", "dyspnea"],
            "neurologic": ["seizure", "neuropath", "tremor", "dizziness", "headache"],
        }
        for se in all_se:
            se_lower = se.lower()
            for organ, kws in organ_kw.items():
                if any(k in se_lower for k in kws):
                    organ_risks[organ] += 1

        # 归一化到 0-100
        max_val = max(organ_risks.values()) if organ_risks.values() else 1
        organ_risks = {k: round(v / max_val * 100, 1) if max_val > 0 else 0
                       for k, v in organ_risks.items()}

        # 综合风险
        age = patient_info["age"]
        conditions = patient_info["conditions"]
        age_factor = 1.8 if age > 80 else (1.4 if age > 65 else (1.2 if age < 18 else 1.0))
        cond_bonus = sum(10 for c in conditions if any(
            k in c.lower() for k in ["ckd", "renal", "liver", "heart", "diabetes", "肝", "肾", "心"]))
        interaction_bonus = min(25, len(interactions) * 8)
        risk_score = min(100, (len(all_se) * 0.2 + interaction_bonus + cond_bonus) * age_factor)

        if risk_score >= 80: risk_level = "极度高危"
        elif risk_score >= 60: risk_level = "高危"
        elif risk_score >= 40: risk_level = "中等风险"
        elif risk_score >= 20: risk_level = "低风险"
        else: risk_level = "安全"

    col_radar, col_score = st.columns([1, 1])
    with col_radar:
        fig_radar = build_radar_chart(organ_risks)
        st.plotly_chart(fig_radar, use_container_width=True)
    with col_score:
        cls = risk_css_class(risk_level)
        st.markdown(f"""
        <div class="risk-card {cls}" style="text-align:center;">
            <h1 style="font-size:3.5rem;margin:0;">{risk_score:.0f}</h1>
            <h3 style="margin:0;">/ 100</h3>
            <h2 style="margin:0.3rem 0;">{risk_level}</h2>
        </div>
        """, unsafe_allow_html=True)
        # 逐药风险
        if recommender:
            st.markdown("**各药物风险：**")
            for drug in drugs:
                ses = recommender.get_drug_side_effects(drug)
                se_count = len(ses)
                mini_score = min(100, se_count * 0.5 * age_factor)
                st.markdown(f"- **{drug}**: {mini_score:.0f}分 ({se_count} 种副作用)")

    # ══════════════════════════════════════════════════════════
    #  替代药推荐
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 💊 替代药物推荐")

    if recommender:
        for drug in drugs:
            try:
                alts = recommender.recommend_alternatives(
                    drug_name=drug, current_meds=drugs,
                    patient_conditions=patient_info["conditions"], top_k=3)
                if alts:
                    st.markdown(f"### 🔄 {drug.upper()} 的替代方案")
                    for alt in alts:
                        reasons_str = "；".join(alt.get("reasons", []))
                        st.markdown(f"""
                        <div class="alt-card">
                            <b>{alt['drug_name']}</b>
                            &nbsp; <span style="color:#3498db;">匹配度: {alt['score']}</span>
                            &nbsp; <span style="color:#7f8c8d;">ATC: {alt.get('atc_code','N/A')}</span>
                            <br><small>📋 {reasons_str}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(f"{drug} 暂无推荐替代药物。")
            except Exception as e:
                st.warning(f"{drug} 替代推荐查询失败: {e}")
    else:
        st.info("后端未加载，无法生成替代推荐。")

    # ══════════════════════════════════════════════════════════
    #  慢病续方提醒
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 📅 慢病续方提醒")

    if trigger:
        # 构造示例处方（基于已知慢病药物）
        chronic_map = {
            "metformin": ("糖尿病", 500, 2), "amlodipine": ("高血压", 5, 1),
            "atorvastatin": ("高血脂", 20, 1), "losartan": ("高血压", 50, 1),
            "insulin": ("糖尿病", 10, 2), "hydrochlorothiazide": ("高血压", 25, 1),
        }
        sample_rxs = []
        for drug in drugs:
            dl = drug.lower()
            if dl in chronic_map:
                dis, dose, times = chronic_map[dl]
                sample_rxs.append({
                    "drug_name": drug, "total_quantity": dose * times * 30,
                    "dose_per_time": dose, "times_per_day": times,
                    "start_date": (datetime.now() - timedelta(days=28)).strftime("%Y-%m-%d"),
                    "dose_unit": "mg",
                })

        if sample_rxs:
            results = trigger.batch_check(sample_rxs, alert_days)
            for r in results:
                cls = "refill-critical" if r["urgency"] in ("critical", "high") else (
                    "refill-warning" if r["urgency"] == "medium" else "refill-ok")
                st.markdown(f'<div class="{cls}">{r["message"]}</div>', unsafe_allow_html=True)
        else:
            st.info("当前药物不涉及常见慢病管理药物，无续方提醒。")
    else:
        st.info("后端未加载，无法检查续方。")

    # ══════════════════════════════════════════════════════════
    #  数据可视化
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 📈 数据可视化")

    tab_graph, tab_se, tab_timeline = st.tabs([
        "🕸️ 药物知识图谱", "📊 副作用分布", "📅 用药时间线"])

    with tab_graph:
        if recommender:
            G = build_drug_knowledge_graph(drugs, recommender)
            if G.number_of_nodes() > 0:
                html_graph = render_knowledge_graph_html(G)
                st.components.v1.html(html_graph, height=480)
                st.caption("🔴 红色圆点 = 药物 &nbsp;|&nbsp; 🔵 蓝色方框 = 副作用 &nbsp;|&nbsp; 红线 = DDI 关联")
            else:
                st.info("无法构建知识图谱。")

    with tab_se:
        if recommender:
            all_se = []
            for drug in drugs:
                all_se.extend(recommender.get_drug_side_effects(drug))
            if all_se:
                fig_se = build_se_frequency_chart(all_se)
                st.plotly_chart(fig_se, use_container_width=True)
            else:
                st.info("未找到副作用数据。")

    with tab_timeline:
        sample_rxs_tl = []
        for drug in drugs:
            dl = drug.lower()
            if dl in chronic_map:
                dis, dose, times = chronic_map[dl]
                sample_rxs_tl.append({
                    "drug_name": drug, "total_quantity": dose * times * 30,
                    "dose_per_time": dose, "times_per_day": times,
                    "start_date": (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d"),
                })
        if sample_rxs_tl:
            fig_tl = build_medication_timeline(sample_rxs_tl)
            st.plotly_chart(fig_tl, use_container_width=True)
        else:
            # 给所有药物一个默认时间线
            default_rxs = [{"drug_name": d, "total_quantity": 60, "dose_per_time": 1,
                            "times_per_day": 2, "start_date": (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")}
                           for d in drugs]
            fig_tl = build_medication_timeline(default_rxs)
            st.plotly_chart(fig_tl, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  页面 📊 数据探索
# ═══════════════════════════════════════════════════════════════

def page_data_explorer():
    """数据探索页面：浏览 SIDER / DrugCentral 数据"""
    st.markdown("# 📊 数据探索")
    st.markdown("> 浏览 SIDER 副作用数据 & DrugCentral 药物靶点数据")

    try:
        loader, _, _ = load_backend()
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return

    tab_sider, tab_dc = st.tabs(["📖 SIDER 副作用数据", "🧬 DrugCentral 靶点数据"])

    # ── SIDER ─────────────────────────────────────────────────
    with tab_sider:
        st.markdown("### 药物名称检索")
        search_term = st.text_input("输入药物名称搜索", value="aspirin", key="sider_search")

        try:
            drug_names_df = loader.load_drug_names()
            if search_term:
                matches = drug_names_df[
                    drug_names_df["drug_name"].str.contains(search_term, case=False, na=False)
                ].drop_duplicates(subset=["cid", "drug_name"]).head(50)
                st.dataframe(matches, use_container_width=True, height=300)

                if not matches.empty:
                    selected_drug = st.selectbox("选择药物查看详情", matches["drug_name"].unique(), key="sel_drug")
                    selected_cid = matches[matches["drug_name"] == selected_drug].iloc[0]["cid"]

                    st.markdown(f"#### {selected_drug} 的副作用列表")
                    try:
                        se_df = loader.load_side_effects()
                        drug_se = se_df[se_df["cid"] == selected_cid][["side_effect_name", "frequency"]].drop_duplicates()
                        st.dataframe(drug_se.head(100), use_container_width=True, height=400)
                        st.caption(f"共 {len(drug_se)} 条副作用记录")
                    except Exception as e:
                        st.warning(f"副作用数据加载失败: {e}")
        except Exception as e:
            st.warning(f"SIDER 数据加载失败: {e}")

        st.markdown("---")
        st.markdown("### 数据集统计")
        try:
            stats = loader.get_stats()
            cols = st.columns(3)
            labels = {"drug_names": "药物名称", "side_effects": "副作用记录",
                      "indications": "适应症记录", "drug_targets": "药物靶点",
                      "patients": "患者记录", "prescriptions": "处方记录"}
            for i, (k, v) in enumerate(stats.items()):
                with cols[i % 3]:
                    st.metric(labels.get(k, k), f"{v:,}")
        except Exception:
            st.info("统计数据不可用。")

    # ── DrugCentral ──────────────────────────────────────────
    with tab_dc:
        st.markdown("### 药物-靶点相互作用")
        try:
            dt_df = loader.load_drug_target_interactions()
            st.dataframe(dt_df.head(100), use_container_width=True, height=400)
            st.caption(f"共 {len(dt_df):,} 条药物-靶点记录")

            if "DRUG_NAME" in dt_df.columns:
                drug_filter = st.text_input("筛选药物", value="", key="dc_filter")
                if drug_filter:
                    filtered = dt_df[dt_df["DRUG_NAME"].str.contains(drug_filter, case=False, na=False)]
                    st.dataframe(filtered.head(100), use_container_width=True)
        except Exception as e:
            st.warning(f"DrugCentral 数据加载失败: {e}")

        st.markdown("---")
        st.markdown("### 药物分子结构 (SMILES)")
        try:
            smiles_df = loader.load_drug_smiles()
            st.dataframe(smiles_df.head(50), use_container_width=True, height=300)
        except Exception as e:
            st.warning(f"SMILES 数据加载失败: {e}")


# ═══════════════════════════════════════════════════════════════
#  页面 ℹ️ 关于
# ═══════════════════════════════════════════════════════════════

def page_about():
    """关于页面"""
    st.markdown("# ℹ️ 关于 PharmSandbox 药盘推演")

    st.markdown("""
    ## 🎯 项目简介

    **PharmSandbox 药盘推演** 是一个基于图神经网络 (GNN) 的药物相互作用沙盘推演系统，
    致力于在处方前为医生和药师提供全面的用药风险评估。

    ### 核心功能

    | 功能 | 描述 |
    |------|------|
    | 🔍 **DDI 沙盘推演** | 基于 SIDER / TWOSIDES 数据检测药物-药物相互作用 |
    | ⚠️ **风险量化评分** | 结合患者年龄、疾病史、化验值的 0-100 动态评分 |
    | 💊 **替代药推荐** | 基于 ATC 分类树 + 适应症匹配 + 指南约束的无害化替代 |
    | 📅 **慢病续方提醒** | 根据用药周期计算剩余药量，断药前自动触发续方提醒 |
    | 🕸️ **知识图谱** | 药物-副作用-靶点知识图谱可视化 |

    ### 数据来源

    - **SIDER** (Side Effect Resource): 药物副作用信息
    - **DrugCentral**: 药物靶点与分子结构
    - **MIMIC-IV**: 重症监护真实世界数据
    - **NSIDES / TWOSIDES**: 药物联合副作用数据

    ### 技术栈

    - **后端**: Python, Flask, PyTorch, DGL (图神经网络)
    - **前端**: Streamlit, Plotly, NetworkX, vis-network.js
    - **数据**: SIDER TSV, DrugCentral, MIMIC-IV CSV

    ### 参考项目

    - [rugved18-dev/Adverse-Drug-Reaction-Analysis-Dashboard](https://github.com/rugved18-dev/Adverse-Drug-Reaction-Analysis-Dashboard)
    - [SumGNN](https://github.com/mims-harvard/SumGNN)

    ---

    ### ⚠️ 免责声明

    > 本系统仅供科研与教育用途，**不能替代专业医疗建议**。
    > 所有用药决策请咨询合格的医疗专业人员。

    ---

    *PharmSandbox v1.0 · 药盘推演*
    """)


# ═══════════════════════════════════════════════════════════════
#  多页面路由
# ═══════════════════════════════════════════════════════════════

# 使用 st.navigation (Streamlit ≥1.30) 或手动 tab 路由
def main():
    """多页面入口"""
    # 尝试使用 st.navigation（新版 Streamlit）
    try:
        page_home_ = st.Page(page_home, title="🏠 首页 · 综合沙盘推演", icon="🏠", default=True)
        page_data_ = st.Page(page_data_explorer, title="📊 数据探索", icon="📊")
        page_about_ = st.Page(page_about, title="ℹ️ 关于", icon="ℹ️")
        nav = st.navigation([page_home_, page_data_, page_about_])
        nav.run()
    except AttributeError:
        # 旧版 Streamlit fallback
        page = st.sidebar.radio(
            "导航", ["🏠 首页 · 综合沙盘推演", "📊 数据探索", "ℹ️ 关于"],
            index=0,
        )
        if page == "🏠 首页 · 综合沙盘推演":
            page_home()
        elif page == "📊 数据探索":
            page_data_explorer()
        else:
            page_about()


if __name__ == "__main__":
    main()
