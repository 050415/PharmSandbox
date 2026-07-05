"""
PharmSandbox - 启动脚本
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == '__main__':
    print("=" * 60)
    print("  PharmSandbox - Drug Interaction Sandbox")
    print("  GNN-based Drug-Drug Interaction Simulation System")
    print("=" * 60)
    print()
    
    # 检查数据
    data_root = project_root / "data"
    print(f"数据目录: {data_root}")
    
    datasets = {
        'SIDER': data_root / "sider" / "drug_names.tsv",
        'DrugCentral': data_root / "drugcentral" / "drugcentral_structures_smiles.tsv",
        'MIMIC-IV': data_root / "mimic" / "patients.csv.gz",
    }
    
    for name, path in datasets.items():
        status = "[OK]" if path.exists() else "[MISSING]"
        print(f"  {status} {name}")
    
    print()
    print("正在启动Web服务...")
    print("访问地址: http://localhost:5000")
    print()
    
    # 启动Flask
    from src.action.api.app import app
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
