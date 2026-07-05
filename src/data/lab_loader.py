"""
PharmSandbox - 化验数据加载器
高效加载 MIMIC-IV labevents.csv.gz (2.5GB, 1.58亿行)
支持按患者ID + 化验项目ID 快速查询
"""
import pandas as pd
import gzip
from pathlib import Path
from collections import defaultdict


# MIMIC-IV 常见化验项目 itemid 映射
LAB_ITEMS = {
    # 肾功能
    50912: {'name': 'Creatinine', 'name_cn': '肌酐', 'unit': 'mg/dL', 
            'normal_low': 0.6, 'normal_high': 1.2, 'organ': 'renal'},
    51006: {'name': 'BUN', 'name_cn': '血尿素氮', 'unit': 'mg/dL',
            'normal_low': 7, 'normal_high': 20, 'organ': 'renal'},
    50920: {'name': 'eGFR', 'name_cn': '肾小球滤过率', 'unit': 'mL/min/1.73m2',
            'normal_low': 60, 'normal_high': 120, 'organ': 'renal'},
    
    # 肝功能
    50861: {'name': 'ALT', 'name_cn': '谷丙转氨酶', 'unit': 'IU/L',
            'normal_low': 7, 'normal_high': 56, 'organ': 'hepatic'},
    50878: {'name': 'AST', 'name_cn': '谷草转氨酶', 'unit': 'IU/L',
            'normal_low': 10, 'normal_high': 40, 'organ': 'hepatic'},
    50862: {'name': 'Albumin', 'name_cn': '白蛋白', 'unit': 'g/dL',
            'normal_low': 3.5, 'normal_high': 5.0, 'organ': 'hepatic'},
    50885: {'name': 'Bilirubin', 'name_cn': '胆红素', 'unit': 'mg/dL',
            'normal_low': 0.1, 'normal_high': 1.2, 'organ': 'hepatic'},
    
    # 凝血功能
    50976: {'name': 'PT', 'name_cn': '凝血酶原时间', 'unit': 'sec',
            'normal_low': 11, 'normal_high': 13.5, 'organ': 'hematologic'},
    51237: {'name': 'INR', 'name_cn': '国际标准化比值', 'unit': '',
            'normal_low': 0.8, 'normal_high': 1.2, 'organ': 'hematologic'},
    50977: {'name': 'PTT', 'name_cn': '活化部分凝血活酶时间', 'unit': 'sec',
            'normal_low': 25, 'normal_high': 35, 'organ': 'hematologic'},
    
    # 电解质
    50971: {'name': 'Potassium', 'name_cn': '钾', 'unit': 'mEq/L',
            'normal_low': 3.5, 'normal_high': 5.0, 'organ': 'cardiac'},
    50983: {'name': 'Sodium', 'name_cn': '钠', 'unit': 'mEq/L',
            'normal_low': 136, 'normal_high': 145, 'organ': 'general'},
    50902: {'name': 'Chloride', 'name_cn': '氯', 'unit': 'mEq/L',
            'normal_low': 98, 'normal_high': 106, 'organ': 'general'},
    
    # 血常规
    51222: {'name': 'Hemoglobin', 'name_cn': '血红蛋白', 'unit': 'g/dL',
            'normal_low': 12, 'normal_high': 17.5, 'organ': 'hematologic'},
    51265: {'name': 'Platelet', 'name_cn': '血小板', 'unit': 'K/uL',
            'normal_low': 150, 'normal_high': 400, 'organ': 'hematologic'},
    51301: {'name': 'WBC', 'name_cn': '白细胞', 'unit': 'K/uL',
            'normal_low': 4.5, 'normal_high': 11.0, 'organ': 'general'},
    
    # 血糖
    50931: {'name': 'Glucose', 'name_cn': '葡萄糖', 'unit': 'mg/dL',
            'normal_low': 70, 'normal_high': 100, 'organ': 'general'},
    
    # 心肌标志物
    51003: {'name': 'Troponin_T', 'name_cn': '肌钙蛋白T', 'unit': 'ng/mL',
            'normal_low': 0, 'normal_high': 0.04, 'organ': 'cardiac'},
}

# 需要关注的关键itemid（用于快速过滤）
KEY_ITEMIDS = set(LAB_ITEMS.keys())


class LabEventsLoader:
    """
    MIMIC-IV 化验数据加载器
    支持按患者ID查询最新化验值，用于风险评分

    优先使用预建索引（快速），若无索引则回退到全量扫描（慢）。
    预建索引可通过 scripts/build_lab_index.py 生成。
    """

    def __init__(self, data_root=None):
        if data_root is None:
            from src.config import DATA_ROOT
            data_root = DATA_ROOT
        self.data_root = Path(data_root)
        self._lab_path = self.data_root / "mimic" / "labevents.csv.gz"
        self._index_dir = self.data_root / "mimic" / "lab_index"
        self._patient_cache = {}
        self._index_loaded = False
        self._index_data = {}  # {subject_id: DataFrame}

        # 检查预建索引
        self._has_index = (self._index_dir / "manifest.json").exists()
        if self._has_index:
            print(f"  ✓ 化验数据索引已找到: {self._index_dir}")
    
    def get_patient_labs(self, subject_id, latest_only=True):
        """
        获取指定患者的化验结果

        Args:
            subject_id: 患者ID
            latest_only: 是否只返回每项化验的最新结果

        Returns:
            化验结果字典 {itemid: {name, value, unit, flag, time, normal_range}}
        """
        if subject_id in self._patient_cache:
            data = self._patient_cache[subject_id]
        elif self._has_index:
            data = self._read_from_index(subject_id)
            self._patient_cache[subject_id] = data
        else:
            data = self._read_patient_labs(subject_id)
            self._patient_cache[subject_id] = data

        if latest_only:
            # 每个itemid只保留最新的记录
            latest = {}
            for itemid, records in data.items():
                if records:
                    latest[itemid] = max(records, key=lambda r: r.get('charttime', ''))
            return latest
        return data

    def _read_from_index(self, subject_id):
        """从预建索引中快速读取指定患者的化验数据。"""
        result = defaultdict(list)
        bucket_key = (subject_id // 1000) * 1000
        file_path = self._index_dir / f"subject_{bucket_key:06d}.pkl"

        if not file_path.exists():
            return dict(result)

        try:
            df = pd.read_pickle(str(file_path))
            patient_data = df[df['subject_id'] == subject_id]

            for _, row in patient_data.iterrows():
                itemid = row['itemid']
                if pd.notna(row.get('valuenum')):
                    item_info = LAB_ITEMS.get(itemid, {})
                    result[itemid].append({
                        'itemid': itemid,
                        'name': item_info.get('name', f'Item_{itemid}'),
                        'name_cn': item_info.get('name_cn', ''),
                        'value': row['valuenum'],
                        'unit': row.get('valueuom', item_info.get('unit', '')),
                        'flag': row.get('flag', ''),
                        'charttime': str(row.get('charttime', '')),
                        'normal_low': row.get('ref_range_lower', item_info.get('normal_low')),
                        'normal_high': row.get('ref_range_upper', item_info.get('normal_high')),
                        'organ': item_info.get('organ', 'general'),
                    })
        except Exception as e:
            print(f"从索引读取患者 {subject_id} 数据失败: {e}")

        return dict(result)
    
    def _read_patient_labs(self, subject_id):
        """分块读取指定患者的化验数据"""
        result = defaultdict(list)
        
        try:
            chunk_iter = pd.read_csv(
                self._lab_path, compression='gzip',
                chunksize=100000,
                usecols=['subject_id', 'itemid', 'charttime', 'valuenum', 
                         'valueuom', 'ref_range_lower', 'ref_range_upper', 'flag'],
                dtype={'subject_id': int, 'itemid': int, 'valuenum': float}
            )
            
            for chunk in chunk_iter:
                patient_data = chunk[chunk['subject_id'] == subject_id]
                if len(patient_data) == 0:
                    continue
                
                # 只保留关键化验项目
                patient_data = patient_data[patient_data['itemid'].isin(KEY_ITEMIDS)]
                
                for _, row in patient_data.iterrows():
                    itemid = row['itemid']
                    if pd.notna(row['valuenum']):
                        item_info = LAB_ITEMS.get(itemid, {})
                        result[itemid].append({
                            'itemid': itemid,
                            'name': item_info.get('name', f'Item_{itemid}'),
                            'name_cn': item_info.get('name_cn', ''),
                            'value': row['valuenum'],
                            'unit': row.get('valueuom', item_info.get('unit', '')),
                            'flag': row.get('flag', ''),
                            'charttime': str(row.get('charttime', '')),
                            'normal_low': row.get('ref_range_lower', item_info.get('normal_low')),
                            'normal_high': row.get('ref_range_upper', item_info.get('normal_high')),
                            'organ': item_info.get('organ', 'general')
                        })
        except Exception as e:
            print(f"读取化验数据失败: {e}")
        
        return dict(result)
    
    def get_organ_function(self, subject_id):
        """
        获取患者各器官功能评估
        
        Returns:
            器官功能状态 {organ: {status, key_values}}
        """
        labs = self.get_patient_labs(subject_id)
        organ_status = {}
        
        organ_mapping = {
            'renal': {'items': [50912, 51006, 50920], 'label': '肾功能'},
            'hepatic': {'items': [50861, 50878, 50862, 50885], 'label': '肝功能'},
            'hematologic': {'items': [51237, 50976, 51222, 51265], 'label': '凝血/血液'},
            'cardiac': {'items': [50971, 51003], 'label': '心脏功能'},
        }
        
        for organ, config in organ_mapping.items():
            values = {}
            abnormal = False
            
            for itemid in config['items']:
                if itemid in labs:
                    lab = labs[itemid]
                    values[lab['name']] = {
                        'value': lab['value'],
                        'unit': lab['unit'],
                        'name_cn': lab['name_cn'],
                        'normal_range': f"{lab['normal_low']}-{lab['normal_high']}",
                        'is_abnormal': lab.get('flag') == 'abnormal' or 
                                      lab['value'] < lab['normal_low'] or 
                                      lab['value'] > lab['normal_high']
                    }
                    if values[lab['name']]['is_abnormal']:
                        abnormal = True
            
            if values:
                organ_status[organ] = {
                    'label': config['label'],
                    'status': 'abnormal' if abnormal else 'normal',
                    'values': values
                }
        
        return organ_status
    
    def get_risk_factors(self, subject_id):
        """
        根据化验结果计算风险因子（用于RiskScorer）
        
        Returns:
            各器官系统的风险倍率 {organ: multiplier}
        """
        labs = self.get_patient_labs(subject_id)
        risk_factors = {
            'renal': 1.0, 'hepatic': 1.0, 'cardiac': 1.0,
            'hematologic': 1.0, 'respiratory': 1.0, 'neurologic': 1.0
        }
        
        # 肾功能评估
        if 50912 in labs:  # 肌酐
            cr = labs[50912]['value']
            if cr > 3.0:
                risk_factors['renal'] *= 2.5
            elif cr > 1.5:
                risk_factors['renal'] *= 1.8
            elif cr > 1.2:
                risk_factors['renal'] *= 1.3
        
        if 50920 in labs:  # eGFR
            egfr = labs[50920]['value']
            if egfr < 15:
                risk_factors['renal'] *= 3.0
            elif egfr < 30:
                risk_factors['renal'] *= 2.0
            elif egfr < 60:
                risk_factors['renal'] *= 1.5
        
        # 肝功能评估
        if 50861 in labs:  # ALT
            alt = labs[50861]['value']
            if alt > 200:
                risk_factors['hepatic'] *= 2.5
            elif alt > 100:
                risk_factors['hepatic'] *= 1.8
            elif alt > 56:
                risk_factors['hepatic'] *= 1.3
        
        if 50878 in labs:  # AST
            ast = labs[50878]['value']
            if ast > 200:
                risk_factors['hepatic'] *= 2.0
            elif ast > 40:
                risk_factors['hepatic'] *= 1.5
        
        # 凝血功能评估
        if 51237 in labs:  # INR
            inr = labs[51237]['value']
            if inr > 3.0:
                risk_factors['hematologic'] *= 2.5
            elif inr > 2.0:
                risk_factors['hematologic'] *= 1.8
            elif inr > 1.5:
                risk_factors['hematologic'] *= 1.3
        
        # 心脏功能评估
        if 50971 in labs:  # 钾
            k = labs[50971]['value']
            if k > 6.0 or k < 3.0:
                risk_factors['cardiac'] *= 2.0
            elif k > 5.0 or k < 3.5:
                risk_factors['cardiac'] *= 1.3
        
        if 51003 in labs:  # 肌钙蛋白T
            tnt = labs[51003]['value']
            if tnt > 0.1:
                risk_factors['cardiac'] *= 2.5
            elif tnt > 0.04:
                risk_factors['cardiac'] *= 1.5
        
        return risk_factors


if __name__ == "__main__":
    loader = LabEventsLoader()
    
    # 测试：查询患者10000032的化验结果
    print("=== 化验数据加载器测试 ===")
    print(f"支持的化验项目: {len(LAB_ITEMS)} 种")
    for itemid, info in LAB_ITEMS.items():
        print(f"  {itemid}: {info['name_cn']} ({info['name']}) - {info['organ']}")
