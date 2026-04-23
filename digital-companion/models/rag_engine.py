# ==============================================
# ไฟล์: models/rag_engine.py
# หน้าที่: RAG (Retrieval-Augmented Generation)
# วิธีทำงาน: ค้นหาบทสนทนาที่คล้ายกันจาก knowledge base
#            แล้วส่งเป็น context ให้ LLM ตอบได้แม่นยำขึ้น
# ==============================================

import json
import os
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class RagEngine:
    """
    Retrieval-Augmented Generation Engine สำหรับ BAB
    
    วิธีทำงาน:
    1. โหลด knowledge base จากไฟล์ JSON
    2. สร้าง TF-IDF vectors จากข้อความทั้งหมด
    3. เมื่อรับ query → หาข้อความที่ใกล้เคียงที่สุด
    4. คืนค่า context สำหรับใส่ใน prompt
    """

    def __init__(self, knowledge_base_path: str = "data/sample_chats.json"):
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base: List[Dict] = []
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",   # ใช้ character n-gram (เหมาะกับภาษาไทย)
            ngram_range=(2, 4),   # bigram ถึง 4-gram
            min_df=1,
            sublinear_tf=True
        )
        self.tfidf_matrix = None
        self._load_and_index()

    # --------------------------------------------------
    # โหลดและ index knowledge base
    # --------------------------------------------------
    def _load_and_index(self):
        """โหลดไฟล์ JSON และสร้าง TF-IDF index"""
        try:
            if not os.path.exists(self.knowledge_base_path):
                print(f"⚠️  ไม่พบ knowledge base: {self.knowledge_base_path}")
                print("   RAG จะทำงานโดยไม่มี context")
                return

            with open(self.knowledge_base_path, "r", encoding="utf-8") as f:
                self.knowledge_base = json.load(f)

            if not self.knowledge_base:
                print("⚠️  Knowledge base ว่างเปล่า")
                return

            # ใช้ข้อความของ user เป็น corpus สำหรับ indexing
            corpus = [item.get("user", "") for item in self.knowledge_base]
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

            print(f"✅ RAG Engine พร้อมแล้ว! (โหลด {len(self.knowledge_base)} รายการ)")

        except json.JSONDecodeError as e:
            print(f"❌ ไฟล์ JSON ไม่ถูกต้อง: {e}")
        except Exception as e:
            print(f"❌ RAG Engine เกิดข้อผิดพลาด: {e}")

    # --------------------------------------------------
    # ค้นหา context ที่ใกล้เคียงที่สุด
    # --------------------------------------------------
    def retrieve(self, query: str, top_k: int = 2, threshold: float = 0.1) -> List[Dict]:
        """
        ค้นหาบทสนทนาที่ใกล้เคียง query มากที่สุด

        Args:
            query     : ข้อความจากผู้ใช้
            top_k     : จำนวน context สูงสุดที่ต้องการ (default: 2)
            threshold : ค่า similarity ขั้นต่ำ (default: 0.1)

        Returns:
            List ของ dict ที่มี user, mali_response, stress_level, score
        """
        if self.tfidf_matrix is None or not self.knowledge_base:
            return []

        try:
            # แปลง query เป็น vector
            query_vec = self.vectorizer.transform([query])

            # คำนวณ cosine similarity กับทุก entry
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            # เรียงจากมากไปน้อย
            ranked_indices = np.argsort(scores)[::-1]

            results = []
            for idx in ranked_indices[:top_k]:
                if scores[idx] >= threshold:
                    entry = self.knowledge_base[idx].copy()
                    entry["similarity_score"] = round(float(scores[idx]), 3)
                    results.append(entry)

            return results

        except Exception as e:
            print(f"⚠️  RAG retrieve error: {e}")
            return []

    def build_context(self, query: str, top_k: int = 2) -> str:
        """
        สร้างข้อความ context สำหรับใส่ใน system prompt

        Args:
            query : ข้อความจากผู้ใช้
            top_k : จำนวน context สูงสุด

        Returns:
            str ที่พร้อมนำไปต่อท้าย system prompt
        """
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return ""

        lines = ["--- ตัวอย่างการตอบสนองที่เคยใช้ได้ผล ---"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. ผู้ใช้พูดว่า: \"{r.get('user', '')}\"\n"
                f"   ระดับความเครียด: {r.get('stress_level', 'unknown')}\n"
                f"   BAB ตอบว่า: \"{r.get('mali_response', '')}\"\n"
                f"   (ความคล้ายคลึง: {r.get('similarity_score', 0)})"
            )
        lines.append("--- ใช้ตัวอย่างนี้เป็นแนวทาง ไม่ต้องตอบเหมือนทุกคำ ---")

        return "\n".join(lines)
    def add_entry(self, user_text: str, response: str,
                  stress_level: str = "unknown", keywords: List[str] = None):
        """
        เพิ่มบทสนทนาใหม่เข้า knowledge base และ re-index

        Args:
            user_text    : ข้อความของผู้ใช้
            response     : คำตอบของ BAB
            keywords     : คำสำคัญ (optional)
        """
        new_entry = {
            "id": len(self.knowledge_base) + 1,
            "user": user_text,
            "stress_level": stress_level,
            "keywords": keywords or [],
            "mali_response": response
        }
        self.knowledge_base.append(new_entry)

        # Re-index
        corpus = [item.get("user", "") for item in self.knowledge_base]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    @property
    def size(self) -> int:
        """จำนวน entry ใน knowledge base"""
        return len(self.knowledge_base)

_rag_instance: RagEngine | None = None


def get_rag_engine(knowledge_base_path: str = "data/sample_chats.json") -> RagEngine:
    """คืนค่า singleton instance ของ RagEngine"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RagEngine(knowledge_base_path)
    return _rag_instance


if __name__ == "__main__":
    engine = RagEngine("../data/sample_chats.json")

    test_queries = [
        "เครียดเรื่องสอบมากเลย",
        "นอนไม่หลับทุกคืนเลย",
        "วันนี้สบายดีค่ะ",
        "หมดแรงมากเลย ท้อมากๆ"
    ]

    print("=== ทดสอบ RAG Engine ===\n")
    for q in test_queries:
        print(f"Query : {q}")
        results = engine.retrieve(q, top_k=2)
        for r in results:
            print(f"  → [{r['stress_level']}] {r['user']}  (score: {r['similarity_score']})")
        print()

    print("=== Context ที่จะใส่ใน Prompt ===")
    print(engine.build_context("เครียดเรื่องสอบมากเลย"))