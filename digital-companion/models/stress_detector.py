# ==============================================
# ไฟล์: models/stress_detector.py
# หน้าที่: วิเคราะห์ระดับความเครียดจากข้อความ
# วิธีทำงาน: หา keyword ภาษาไทยที่บ่งบอกความเครียด
# ==============================================

# --------------------------------------------------
# รายการคำที่บ่งบอกความเครียดระดับต่างๆ
# --------------------------------------------------

# คำที่บ่งบอกวิกฤต (ต้องแจ้งเตือนทันที)
CRISIS_KEYWORDS = [
    "อยากตาย",
    "ไม่อยากมีชีวิต",
    "ทนไม่ไหวแล้ว",
    "หมดสิ้นแล้ว",
    "ไม่มีทางออก",
    "อยากหายไป"
]

# คำที่บ่งบอกความเครียดทั่วไป
STRESS_KEYWORDS = [
    "เครียด", "เหนื่อย", "ท้อ", "หมดแรง",
    "กังวล", "กลัว", "เศร้า", "ซึมเศร้า",
    "ไม่ไหว", "อยากร้องไห้", "นอนไม่หลับ",
    "กดดัน", "หนักใจ", "สิ้นหวัง", "โกรธ",
    "หัวร้อน", "หงุดหงิด", "กลุ้ม", "วิตก",
    "ล้า", "burnout", "burn out",
    "ไม่มีแรง", "ไม่อยากทำอะไร", "เบื่อ",
    "หมดไฟ", "ท้อแท้", "สิ้นหวัง"
]


# --------------------------------------------------
# ฟังก์ชันหลัก: วิเคราะห์ระดับความเครียด
# --------------------------------------------------
def detect_stress_level(text: str) -> dict:
    """
    รับข้อความจากผู้ใช้ แล้วคืนค่าเป็น dict ที่มี:
    - level   : ระดับความเครียด (none / mild / moderate / high / severe)
    - score   : คะแนน 0.0 ถึง 1.0
    - keywords_found : คำที่พบในข้อความ
    - crisis  : True ถ้าพบคำวิกฤต (ต้องแสดงเบอร์สายด่วน)

    ตัวอย่างการใช้:
        result = detect_stress_level("เครียดมากเลย นอนไม่หลับ")
        # result = {"level": "moderate", "score": 0.67, ...}
    """

    # แปลงเป็นตัวพิมพ์เล็กเพื่อให้เปรียบเทียบง่ายขึ้น
    text_lower = text.lower()

    # ---- ตรวจสอบคำวิกฤตก่อนเลย ----
    for keyword in CRISIS_KEYWORDS:
        if keyword in text_lower:
            return {
                "level": "severe",
                "score": 1.0,
                "keywords_found": [keyword],
                "crisis": True   # จะทำให้แสดงแบนเนอร์เบอร์สายด่วน
            }

    # ---- นับคำเครียดที่พบในข้อความ ----
    found_keywords = []
    for keyword in STRESS_KEYWORDS:
        if keyword in text_lower:
            found_keywords.append(keyword)

    # ถ้าไม่พบคำเครียดเลย = ปกติ
    if not found_keywords:
        return {
            "level": "none",
            "score": 0.0,
            "keywords_found": [],
            "crisis": False
        }

    # ---- คำนวณ score จากจำนวนคำที่พบ ----
    # ถ้าพบ 3 คำขึ้นไป = score เต็ม 1.0
    score = min(len(found_keywords) / 3, 1.0)
    score = round(score, 2)

    # ---- แปลง score เป็น level ----
    if score < 0.34:
        level = "mild"       # เครียดน้อย (พบ 1 คำ)
    elif score < 0.67:
        level = "moderate"   # เครียดปานกลาง (พบ 2 คำ)
    else:
        level = "high"       # เครียดมาก (พบ 3 คำขึ้นไป)

    return {
        "level": level,
        "score": score,
        "keywords_found": found_keywords,
        "crisis": False
    }


# --------------------------------------------------
# ทดสอบ: รันไฟล์นี้โดยตรงเพื่อดูผลลัพธ์
# คำสั่ง: python models/stress_detector.py
# --------------------------------------------------
if __name__ == "__main__":
    test_messages = [
        "สบายดีค่ะ วันนี้อารมณ์ดี",
        "เหนื่อยนิดหน่อย",
        "เครียดมาก นอนไม่หลับ",
        "หมดแรงเลย เครียดมาก กดดันสุดๆ",
        "ทนไม่ไหวแล้ว"
    ]

    print("=== ทดสอบ Stress Detector ===\n")
    for msg in test_messages:
        result = detect_stress_level(msg)
        print(f"ข้อความ : {msg}")
        print(f"ระดับ   : {result['level']}  (score: {result['score']})")
        print(f"พบคำ    : {result['keywords_found']}")
        print(f"วิกฤต   : {result['crisis']}")
        print("-" * 40)