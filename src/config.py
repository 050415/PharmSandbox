"""
PharmSandbox - 全局配置
统一管理数据路径、模型参数、日志等配置
"""
import os
import logging
from pathlib import Path

# ==================== 路径配置 ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path(os.environ.get("PHARMSANDBOX_DATA", PROJECT_ROOT / "data"))
MODEL_ROOT = PROJECT_ROOT / "models"
LOG_ROOT = PROJECT_ROOT / "logs"

LOG_ROOT.mkdir(exist_ok=True)

# ==================== 日志配置 ====================
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"pharmsandbox.{name}")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
        logger.addHandler(ch)
        fh = logging.FileHandler(LOG_ROOT / "pharmsandbox.log", encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
        logger.addHandler(fh)
    return logger

# ==================== 数据文件路径 ====================
DATA_FILES = {
    'drug_names': DATA_ROOT / "sider" / "drug_names.tsv",
    'drug_atc': DATA_ROOT / "sider" / "drug_atc.tsv",
    'side_effects': DATA_ROOT / "sider" / "meddra_all_se.tsv.gz",
    'indications': DATA_ROOT / "sider" / "meddra_all_indications.tsv.gz",
    'se_freq': DATA_ROOT / "sider" / "meddra_freq.tsv.gz",
    'drug_targets': DATA_ROOT / "drugcentral" / "drugcentral_drug_target_interactions.tsv.gz",
    'smiles': DATA_ROOT / "drugcentral" / "drugcentral_structures_smiles.tsv",
    'patients': DATA_ROOT / "mimic" / "patients.csv.gz",
    'diagnoses': DATA_ROOT / "mimic" / "diagnoses_icd.csv.gz",
    'prescriptions': DATA_ROOT / "mimic" / "prescriptions.csv.gz",
    'labevents': DATA_ROOT / "mimic" / "labevents.csv.gz",
    'twosides': DATA_ROOT / "nsides" / "TWOSIDES.csv.gz",
    'offsides': DATA_ROOT / "nsides" / "OFFSIDES.csv.gz",
}

# ==================== GNN 推理开关 ====================
# 默认启用GNN推理，环境变量 PHARMSANDBOX_GNN=false 可关闭
ENABLE_GNN_INFERENCE = os.environ.get("PHARMSANDBOX_GNN", "true").lower() != "false"

# ==================== API配置 ====================
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': os.environ.get('PHARMSANDBOX_DEBUG', 'false').lower() == 'true',
}
