"""
PharmSandbox 数据加载器
加载SIDER、DrugCentral、MIMIC-IV、NSIDES等数据集
"""
import pandas as pd
import gzip
from pathlib import Path
from src.config import DATA_FILES, get_logger

logger = get_logger("data.loader")


class DataLoader:
    """统一数据加载器，支持懒加载和缓存"""
    
    def __init__(self, data_root=None):
        if data_root:
            self._data_files = self._rebase_paths(data_root)
        else:
            self._data_files = DATA_FILES
        self._cache = {}
    
    @staticmethod
    def _rebase_paths(data_root):
        root = Path(data_root)
        mapping = {}
        for key, path in DATA_FILES.items():
            rel = Path(*path.parts[path.parts.index('data') + 1:])
            mapping[key] = root / rel
        return mapping
    
    def _load_tsv(self, key, names=None, **kwargs):
        if key in self._cache:
            return self._cache[key]
        path = self._data_files.get(key)
        if not path or not path.exists():
            logger.warning(f"数据文件不存在: {path}")
            return pd.DataFrame()
        try:
            logger.info(f"加载 {key} ...")
            if str(path).endswith('.gz'):
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    df = pd.read_csv(f, sep='\t', header=None if names else 'infer', names=names, **kwargs)
            else:
                df = pd.read_csv(path, sep='\t', header=None if names else 'infer', names=names, **kwargs)
            self._cache[key] = df
            logger.info(f"  ✓ {key}: {len(df):,} 行")
            return df
        except Exception as e:
            logger.error(f"  ✗ 加载 {key} 失败: {e}")
            return pd.DataFrame()
    
    def load_drug_names(self):
        return self._load_tsv('drug_names', names=['cid', 'drug_name', 'side_effect_id', 'umls_id'])
    
    def load_drug_atc(self):
        return self._load_tsv('drug_atc', names=['cid', 'drug_name', 'atc_code'])
    
    def load_side_effects(self):
        return self._load_tsv('side_effects', names=['cid', 'related_cid', 'umls_cui',
            'method', 'umls_cui_to', 'side_effect_name'])
    
    def load_indications(self):
        return self._load_tsv('indications', names=['cid', 'umls_cui_from', 'method',
            'indication_name', 'umls_cui_to', 'mesh_id', 'max_phase', 'evidence_type'])
    
    def load_side_effect_freq(self):
        return self._load_tsv('se_freq', names=['stitch_id_flat', 'stitch_id_stereo', 'umls_cui',
            'method', 'side_effect', 'placebo', 'freq_description', 'freq_lower', 'freq_upper', 'freq_point'])
    
    def load_drug_target_interactions(self):
        return self._load_tsv('drug_targets')
    
    def load_drug_smiles(self):
        return self._load_tsv('smiles')
    
    def _load_csv_gz(self, key):
        if key in self._cache:
            return self._cache[key]
        path = self._data_files.get(key)
        if not path or not path.exists():
            logger.warning(f"数据文件不存在: {path}")
            return pd.DataFrame()
        try:
            logger.info(f"加载 {key} ...")
            df = pd.read_csv(path, compression='gzip')
            self._cache[key] = df
            logger.info(f"  ✓ {key}: {len(df):,} 行")
            return df
        except Exception as e:
            logger.error(f"  ✗ 加载 {key} 失败: {e}")
            return pd.DataFrame()
    
    def load_patients(self):
        return self._load_csv_gz('patients')
    
    def load_diagnoses(self):
        return self._load_csv_gz('diagnoses')
    
    def load_prescriptions(self):
        return self._load_csv_gz('prescriptions')
    
    def load_twosides(self):
        if 'twosides' in self._cache:
            return self._cache['twosides']
        path = self._data_files.get('twosides')
        if not path or not path.exists():
            logger.warning(f"TWOSIDES数据不存在: {path}")
            return pd.DataFrame()
        try:
            logger.info("加载 twosides ...")
            df = pd.read_csv(path, compression='gzip', nrows=500000)
            self._cache['twosides'] = df
            logger.info(f"  ✓ twosides: {len(df):,} 行")
            return df
        except Exception as e:
            logger.error(f"  ✗ TWOSIDES加载失败: {e}")
            return pd.DataFrame()
    
    def get_stats(self):
        stats = {}
        loaders = {
            'drug_names': self.load_drug_names,
            'side_effects': self.load_side_effects,
            'indications': self.load_indications,
            'drug_targets': self.load_drug_target_interactions,
            'patients': self.load_patients,
            'prescriptions': self.load_prescriptions,
        }
        for name, loader in loaders.items():
            try:
                stats[name] = len(loader())
            except Exception:
                stats[name] = 0
        return stats
    
    def clear_cache(self):
        self._cache.clear()
        logger.info("数据缓存已清空")


if __name__ == "__main__":
    loader = DataLoader()
    stats = loader.get_stats()
    print("\n=== PharmSandbox 数据集统计 ===")
    for k, v in stats.items():
        print(f"  {k}: {v:,} 条记录")
