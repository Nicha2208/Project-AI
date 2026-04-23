# -*- coding: utf-8 -*-
"""
ไฟล์: models/bab_dataset_analyzer.py
หน้าที่: ข้อมูลความเครียดสำหรับ น้อง BAB
ธรรมชาติ: แปลงจาก Google Colab ให้ใช้ได้กับ FastAPI
"""

import pandas as pd
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple
import json


STRESS_LEVELS = {
    'Low Stress': {'min': 25, 'max': 58, 'label': 'ความเครียดต่ำ'},
    'Medium Stress': {'min': 59, 'max': 92, 'label': 'ความเครียดปานกลาง'},
    'High Stress': {'min': 93, 'max': 125, 'label': 'ความเครียดสูง'}
}

# =====================================================
# คลาส: น้อง BAB DataSet Analyzer
# =====================================================

class BabDatasetAnalyzer:
    """
    ข้อมูลความเครียดสำหรับ น้อง BAB
    โดยใช้ระบบ scoring เพื่อให้ได้ความเครียดที่แม่นยำ
    """
    
    def __init__(self):
        """เตรียมตัว analyzer สำหรับ น้อง BAB"""
        self.df = None
        self.score_cols = None
        self.cause_col = None
        self.method_col = None
        self.top_words_cache = {}
        
    def load_dataset(self, file_path: str) -> Tuple[bool, str]:
        """
        โหลดชุดข้อมูล
        
        Args:
            file_path: เส้นทางไฟล์ CSV
            
        Returns:
            (success: bool, message: str)
        """
        try:
            self.df = pd.read_csv(file_path)
            
            # ตรวจสอบจำนวนคอลัมน์
            if self.df.shape[1] < 33:
                return False, f"❌ ไฟล์มีเพียง {self.df.shape[1]} คอลัมน์ แต่ต้องการอย่างน้อย 33 คอลัมน์"
            
            # เตรียมคอลัมน์
            self.score_cols = self.df.columns[6:31]  # ข้อมูล scoring
            self.cause_col = self.df.columns[31]      # สาเหตุความเครียด
            self.method_col = self.df.columns[32]     # วิธีบรรเทา
            
            return True, f"✅ โหลดข้อมูลสำเร็จ! ({self.df.shape[0]} แถว, {self.df.shape[1]} คอลัมน์)"
            
        except FileNotFoundError:
            return False, f"❌ ไม่พบไฟล์: {file_path}"
        except Exception as e:
            return False, f"❌ เกิดข้อผิดพลาด: {str(e)}"
    
    def clean_dataset(self) -> Tuple[bool, str]:
        """
        ทำความสะอาดข้อมูล
        - แปลงเป็น numeric
        - เติมค่าที่หายไป
        - ลบข้อมูลที่ซ้ำกัน
        - คำนวณ total score
        """
        try:
            if self.df is None:
                return False, "❌ ต้องโหลดข้อมูลก่อน"
            
            # แปลง score columns เป็น numeric
            self.df[self.score_cols] = self.df[self.score_cols].apply(
                pd.to_numeric, errors='coerce'
            )
            
            # เติมค่าที่หายไปด้วย median
            self.df[self.score_cols] = self.df[self.score_cols].fillna(
                self.df[self.score_cols].median()
            )
            
            # ลบข้อมูลที่ซ้ำกัน
            original_count = len(self.df)
            self.df = self.df.drop_duplicates().reset_index(drop=True)
            duplicates_removed = original_count - len(self.df)
            
            # คำนวณ total score
            self.df['total_score'] = self.df[self.score_cols].sum(axis=1)
            
            # สร้าง sentiment category
            self.df['sentiment'] = self.df['total_score'].apply(self._categorize_stress)
            
            return True, f"✅ ทำความสะอาดเสร็จ! (ลบข้อมูลซ้ำ {duplicates_removed} แถว)"
            
        except Exception as e:
            return False, f"❌ เกิดข้อผิดพลาด: {str(e)}"
    
    @staticmethod
    def _categorize_stress(score: float) -> str:
        """
        แปลง score เป็น category
        
        Logic: Min=25, Max=125, Range=100
        - Low: 25-58
        - Medium: 59-92
        - High: 93-125
        """
        if score <= 58:
            return 'Low Stress'
        elif score <= 92:
            return 'Medium Stress'
        else:
            return 'High Stress'
    
    def get_statistics(self) -> Dict:
        """
        ได้สถิติสรุปข้อมูล
        """
        if self.df is None:
            return {}
        
        stats = {
            'total_records': len(self.df),
            'score_stats': {
                'mean': float(self.df['total_score'].mean()),
                'median': float(self.df['total_score'].median()),
                'std': float(self.df['total_score'].std()),
                'min': float(self.df['total_score'].min()),
                'max': float(self.df['total_score'].max())
            },
            'stress_distribution': self.df['sentiment'].value_counts().to_dict()
        }
        return stats
    
    def clean_text(self, text: str) -> str:
        """
        ทำความสะอาดข้อความ
        รักษา: ตัวอักษรไทย, อังกฤษ, ช่องว่าง
        """
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # เก็บเฉพาะตัวอักษรไทย (U+0E00-U+0E7F) และอังกฤษ
        text = re.sub(r'[^\u0E00-\u0E7Fa-z\s]', '', text)
        return text.strip()
    
    def analyze_top_words(self, column_name: Optional[str] = None, 
                         top_n: int = 10) -> List[Tuple[str, int]]:
        """
        วิเคราะห์คำที่ใช้บ่อยที่สุด
        
        Args:
            column_name: ชื่อคอลัมน์ที่ต้องการวิเคราะห์ (default: cause_col)
            top_n: จำนวนคำที่ต้องการ (default: 10)
            
        Returns:
            [(word, count), ...] ตัวอักษรสำเร็จ 10 อันดับแรก
        """
        if self.df is None:
            return []
        
        if column_name is None:
            column_name = self.cause_col
        
        if column_name not in self.df.columns:
            return []
        
        # ล้างข้อความ
        cleaned = self.df[column_name].apply(self.clean_text)
        
        # รวมและแบ่งเป็นคำ
        words = ' '.join(cleaned).split()
        
        # กรองคำสั้น (ต่ำกว่า 2 ตัวอักษร)
        words = [w for w in words if len(w) > 1]
        
        # นับที่ใช้บ่อย
        return Counter(words).most_common(top_n)
    
    def get_stress_causes(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """ได้ 10 สาเหตุความเครียดชั้นนำ"""
        return self.analyze_top_words(self.cause_col, top_n)
    
    def get_stress_methods(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """ได้ 10 วิธีบรรเทาความเครียดชั้นนำ"""
        return self.analyze_top_words(self.method_col, top_n)
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict:
        """
        สร้างรายงานวิเคราะห์ข้อมูล
        """
        if self.df is None:
            return {'error': 'ต้องโหลดข้อมูลก่อน'}
        
        report = {
            'title': 'ความเครียด: รายงานวิเคราะห์โดย น้อง BAB',
            'statistics': self.get_statistics(),
            'top_stress_causes': [
                {'word': w, 'count': c} 
                for w, c in self.get_stress_causes()
            ],
            'top_relief_methods': [
                {'method': w, 'count': c} 
                for w, c in self.get_stress_methods()
            ]
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report



def create_analyzer() -> BabDatasetAnalyzer:
    """สร้าง analyzer instance"""
    return BabDatasetAnalyzer()


if __name__ == "__main__":
    # ตัวอย่างการใช้
    print("🌿 น้อง BAB Dataset Analyzer 🌿")
    print("=" * 50)
    
    analyzer = BabDatasetAnalyzer()
    
    # ตัวอย่าง: โหลดไฟล์
    success, msg = analyzer.load_dataset('data/stress_survey.csv')
    print(msg)
    
    if success:
        success, msg = analyzer.clean_dataset()
        print(msg)
        
        # แสดงสถิติ
        stats = analyzer.get_statistics()
        print("\n📊 สถิติ:")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        
        # แสดงสาเหตุความเครียดชั้นนำ
        top_causes = analyzer.get_stress_causes()
        print("\n😓 สาเหตุความเครียดชั้นนำ 10:")
        for word, count in top_causes:
            print(f"  - {word}: {count} ครั้ง")
