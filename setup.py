"""
PharmSandbox - 一键安装脚本
"""
import subprocess
import sys
import os

def install():
    print("=" * 50)
    print("  PharmSandbox 药盘推演智能体 - 安装脚本")
    print("=" * 50)
    
    # 安装依赖
    print("\n[1/3] 安装Python依赖...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # 创建必要目录
    print("\n[2/3] 创建目录结构...")
    dirs = [
        "data/sider", "data/drugcentral", "data/mimic", "data/nsides", "data/drugbank",
        "src/perception/ner", "src/decision/gnn", "src/decision/risk_scorer",
        "src/decision/recommender", "src/action/api", "frontend/templates", "models", "logs"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  ✓ {d}")
    
    # 验证安装
    print("\n[3/3] 验证安装...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except:
        print("  ✗ PyTorch 安装失败")
    
    try:
        import torch_geometric
        print(f"  ✓ PyTorch Geometric {torch_geometric.__version__}")
    except:
        print("  ✗ PyTorch Geometric 安装失败")
    
    try:
        import flask
        print(f"  ✓ Flask {flask.__version__}")
    except:
        print("  ✗ Flask 安装失败")
    
    try:
        import streamlit
        print(f"  ✓ Streamlit {streamlit.__version__}")
    except:
        print("  ✗ Streamlit 安装失败")
    
    print("\n" + "=" * 50)
    print("  安装完成！")
    print("  启动Flask: python run.py")
    print("  启动Streamlit: streamlit run frontend/dashboard.py")
    print("=" * 50)

if __name__ == "__main__":
    install()
