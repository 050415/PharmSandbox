"""
PharmSandbox - 慢病续方触发器
根据用药周期计算剩余药量，在断药前自动触发续方提醒
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json


class PrescriptionTrigger:
    """慢病续方触发器"""
    
    def __init__(self):
        # 常见慢病用药周期参考
        self._chronic_disease_drugs = {
            'hypertension': ['amlodipine', 'lisinopril', 'losartan', 'valsartan',
                           'hydrochlorothiazide', 'metoprolol', 'atenolol', 'bisoprolol'],
            'diabetes': ['metformin', 'glipizide', 'glyburide', 'pioglitazone',
                        'sitagliptin', 'empagliflozin', 'insulin'],
            'heart_failure': ['carvedilol', 'sacubitril', 'valsartan', 'spironolactone',
                             'furosemide', 'digoxin'],
            'asthma': ['budesonide', 'fluticasone', 'montelukast', 'albuterol',
                      'tiotropium', 'formoterol'],
            'hyperlipidemia': ['atorvastatin', 'rosuvastatin', 'simvastatin',
                              'pravastatin', 'fenofibrate', 'ezetimibe']
        }
        # 预建小写索引，避免每次查询时重建
        self._chronic_disease_lower = {
            disease: {d.lower() for d in drugs}
            for disease, drugs in self._chronic_disease_drugs.items()
        }
    
    def calculate_remaining_days(self, prescription: Dict) -> int:
        """
        计算剩余药量可用天数
        
        Args:
            prescription: 处方信息字典
                - total_quantity: 总开药量
                - dose_per_time: 每次剂量
                - times_per_day: 每日次数
                - start_date: 开始日期 (YYYY-MM-DD)
        
        Returns:
            剩余可用天数
        """
        total = prescription.get('total_quantity', 0)
        dose = prescription.get('dose_per_time', 1)
        times = prescription.get('times_per_day', 1)
        start = prescription.get('start_date', datetime.now().strftime('%Y-%m-%d'))
        
        if dose <= 0 or times <= 0:
            return 0
        
        total_days = total / (dose * times)
        
        try:
            start_dt = datetime.strptime(str(start)[:10], '%Y-%m-%d')
            elapsed = (datetime.now() - start_dt).days
            remaining = max(0, total_days - elapsed)
            return int(remaining)
        except (ValueError, TypeError, AttributeError):
            return int(total_days)
    
    def check_refill_needed(self, prescription: Dict, alert_days: int = 3) -> Dict:
        """
        检查是否需要续方
        
        Args:
            prescription: 处方信息
            alert_days: 提前提醒天数（默认3天）
        
        Returns:
            续方检查结果
        """
        remaining = self.calculate_remaining_days(prescription)
        
        result = {
            'drug_name': prescription.get('drug_name', 'Unknown'),
            'remaining_days': remaining,
            'needs_refill': remaining <= alert_days,
            'urgency': 'normal'
        }
        
        if remaining <= 0:
            result['urgency'] = 'critical'
            result['message'] = f"[!!] {result['drug_name']} has been used up, please refill immediately!"
        elif remaining <= 1:
            result['urgency'] = 'high'
            result['message'] = f"[HIGH] {result['drug_name']} only {remaining} days left, refill ASAP"
        elif remaining <= alert_days:
            result['urgency'] = 'medium'
            result['message'] = f"[MED] {result['drug_name']} will run out in {remaining} days, consider refilling"
        else:
            result['message'] = f"[OK] {result['drug_name']} {remaining} days remaining, no refill needed"
        
        return result
    
    def generate_refill_draft(self, prescription: Dict, patient_info: Dict) -> Dict:
        """
        生成续方草稿
        
        Args:
            prescription: 原处方信息
            patient_info: 患者信息
        
        Returns:
            续方草稿
        """
        draft = {
            'type': 'refill_draft',
            'generated_at': datetime.now().isoformat(),
            'patient': {
                'id': patient_info.get('patient_id', ''),
                'name': patient_info.get('name', ''),
                'age': patient_info.get('age', 0),
                'gender': patient_info.get('gender', '')
            },
            'prescription': {
                'drug_name': prescription.get('drug_name', ''),
                'dosage': prescription.get('dose_per_time', ''),
                'dose_unit': prescription.get('dose_unit', 'mg'),
                'frequency': f"每日{prescription.get('times_per_day', 1)}次",
                'route': prescription.get('route', 'PO'),
                'duration_days': 30,  # 默认续方30天
                'quantity': prescription.get('dose_per_time', 1) * prescription.get('times_per_day', 1) * 30
            },
            'notes': [],
            'status': 'draft_pending_review'
        }
        
        # 添加用药注意事项
        drug_lower = prescription.get('drug_name', '').lower()
        for disease, lower_drugs in self._chronic_disease_lower.items():
            if drug_lower in lower_drugs:
                draft['notes'].append(f"该药物用于{disease}的长期管理")
                draft['chronic_disease'] = disease
                break
        
        return draft
    
    def batch_check(self, prescriptions: List[Dict], alert_days: int = 3) -> List[Dict]:
        """批量检查多个处方的续方需求"""
        results = []
        for rx in prescriptions:
            check = self.check_refill_needed(rx, alert_days)
            results.append(check)
        
        # 按紧急程度排序
        urgency_order = {'critical': 0, 'high': 1, 'medium': 2, 'normal': 3}
        results.sort(key=lambda x: urgency_order.get(x['urgency'], 99))
        return results


if __name__ == "__main__":
    trigger = PrescriptionTrigger()
    
    # 测试：模拟患者处方
    test_prescriptions = [
        {'drug_name': 'Metformin', 'total_quantity': 60, 'dose_per_time': 1, 
         'times_per_day': 2, 'start_date': '2026-06-01', 'dose_unit': '500mg'},
        {'drug_name': 'Amlodipine', 'total_quantity': 30, 'dose_per_time': 1,
         'times_per_day': 1, 'start_date': '2026-06-25', 'dose_unit': '5mg'},
        {'drug_name': 'Atorvastatin', 'total_quantity': 30, 'dose_per_time': 1,
         'times_per_day': 1, 'start_date': '2026-06-28', 'dose_unit': '20mg'},
    ]
    
    print("=== 慢病续方触发器测试 ===\n")
    results = trigger.batch_check(test_prescriptions)
    for r in results:
        print(r['message'])
    
    # 生成续方草稿
    print("\n=== 续方草稿示例 ===")
    patient = {'patient_id': 'P001', 'name': '张三', 'age': 65, 'gender': 'M'}
    draft = trigger.generate_refill_draft(test_prescriptions[0], patient)
    print(json.dumps(draft, indent=2, ensure_ascii=False))
