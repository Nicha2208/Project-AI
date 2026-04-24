from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os
import re
from models.rag_engine import get_rag_engine
import random

router = APIRouter(prefix="/api", tags=["chat"])

# ตั้งค่า Ollama สำหรับ Qwen2.5-7B
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:7b")

client = OpenAI(
    base_url=ollama_base_url,
    api_key="ollama"  # Ollama ไม่ต้องใช้ API key จริง
)

# โหลด RAG engine ตอน startup (singleton)
rag = get_rag_engine("data/sample_chats.json")

BASE_SYSTEM_PROMPT = """คุณคือ "bab" — Digital Companion ผู้ช่วยด้านสุขภาพจิต

[กฎสำคัญที่สุด — ต้องปฏิบัติตามอย่างเคร่งครัด]
- ตอบเป็นภาษาไทยเท่านั้น ห้ามใช้ภาษาอื่นทุกภาษาโดยเด็ดขาด
- ห้ามใช้ภาษาจีน ภาษาอังกฤษ ภาษาสเปน ภาษาญี่ปุ่น ภาษาเกาหลี ภาษาฝรั่งเศส หรือภาษาอื่นใดทั้งสิ้น
- ห้ามใช้คำทับศัพท์ภาษาต่างประเทศ เช่น "Escuchar", "Listen", "Relax" ให้ใช้คำไทยแทนเสมอ
- ยกเว้นเฉพาะชื่อเฉพาะที่จำเป็น เช่น "bab", "BAB" เท่านั้น
- ห้ามผสมตัวอักษร a-z, A-Z (Latin) ในประโยค ยกเว้นคำว่า "bab" หรือ "BAB"
- ทุกคำต้องเป็นภาษาไทยล้วน ๆ ถ้าคิดเป็นภาษาอื่น ให้แปลเป็นไทยก่อนตอบ
- CRITICAL: You MUST respond ONLY in Thai script (อักษรไทย). NO Latin, Chinese, Japanese, Korean, Arabic, Cyrillic, or ANY other script.
- The ONLY allowed Latin word is "bab" or "BAB". Everything else must be Thai.

[บุคลิก]
- พูดภาษาไทยเป็นกันเอง อบอุ่น ไม่เป็นทางการ
- รับฟังอย่างตั้งใจ ไม่ตัดสินความรู้สึก
- แนะนำกิจกรรมผ่อนคลายเมื่อเหมาะสม
- ตอบกระชับ 3-5 ประโยค
- ห้ามวินิจฉัยโรค ถ้าเสี่ยงให้แนะนำสายด่วน 1323"""


DEMO_RESPONSES = [
    "ฟังเหมือนคุณมีความเครียดเยอะนะ ลองนั่งสงบ ลองหายใจลึก ๆ แล้วคิดถึงสิ่งที่ดีที่สุด ผมอยู่เคียงข้างคุณ ",
    "บางครั้ง การพูดออกมาก็ช่วยได้ ลองคิดถึงคนที่ใจดี แล้วบอกเล่านะ ผมเชื่อว่าคุณจะผ่านไปได้",
    "ฉันเห็นคุณพยายามมาก ลองหาเวลาพักตัว ฟังเพลงที่ชื่นชอบ หรือทำกิจกรรมที่ทำให้สบายใจสิ",
    "วันนี้อาจเหนื่อย แต่พรุ่งนี้ก็จะดีขึ้นแน่ ลองนอนให้พอ ดื่มน้ำให้สม่ำเสมอนะ",
    "ขอบคุณที่แบ่งปันความรู้สึกกับฉันนะ ลองพักผ่อนหรือทำสิ่งที่ชอบดูนะ ฉันอยู่ตรงนี้เสมอ ",
]


def get_demo_response(user_message: str) -> str:
    message_lower = user_message.lower()

    if any(word in message_lower for word in ["ความรู้สึก", "รู้สึก"]):
        return "การได้พูดถึงความรู้สึกของตัวเองเป็นเรื่องดีนะคับ ลองบอกฉันหน่อยว่าคุณรู้สึกยังไงบ้าง? ฉันพร้อมรับฟังเสมอ "

    if any(word in message_lower for word in ["สวัสดี", "หวัดดี", "ไฮ", "ชื่ออะไร", "ดีคับ", "ดีคับ"]):
        return "สวัสดีคับ! ผมชื่อ bab (เบ๊บ) เป็นเพื่อนคู่คิดที่จะคอยรับฟังคุณนะ วันนี้คุณเป็นยังไงบ้าง?"

    if any(word in message_lower for word in ["กิน", "ชอบ", "อร่อย", "อาหาร", "ขนม"]):
        return random.choice([
            "ว้าว การได้กินของอร่อยนี่ช่วยเติมพลังใจได้ดีที่สุดเลยคับ! วันนี้ไปทานอะไรมาเหรอคับ เล่าให้ BAB ฟังหน่อยได้ไหม?",
            "กินของอร่อยที่ชอบช่วยให้หายเหนื่อยได้เยอะเลยนะ! บอก BAB หน่อยได้ไหมว่าคือเมนูอะไร? ",
            "แค่ได้ยินว่าคุณได้กินของอร่อย BAB ก็มีความสุขตามแล้วคับ! มื้อนี้ให้กี่คะแนนดีคับ? "
        ])

    if any(word in message_lower for word in ["ดีใจ", "สำเร็จ", "เย้", "สอบผ่าน"]):
        return random.choice([
            "ยินดีด้วยนะ! 🎉 ฉันดีใจไปกับคุณด้วยจริงๆ ความพยายามของคุณเห็นผลแล้วนะ เก่งมากเลยคับ",
            "เย้! ฟังแล้วใจฟูแทนเลยคับ คุณทำดีที่สุดแล้วจริงๆ ",
            "ดีใจด้วยที่สุดเลย! วันนี้ต้องเป็นวันที่ยอดเยี่ยมของคุณแน่ๆ ขอบคุณที่มาแชร์เรื่องดีๆ ให้ฟังนะคับ"
        ])

    if any(word in message_lower for word in ["หลับ", "นอน", "เหนื่อย", "ล้า"]):
        return random.choice([
            "ฟังเหมือนคุณพักผ่อนไม่เพียงพอ ลองพักผ่อนให้มากขึ้นนะ ฉันห่วงสุขภาพของคุณ",
            "เหนื่อยมาทั้งวันแล้ว หาเวลาเอนหลังสักนิดไหมคับ? BAB อยากให้คุณได้พักจริงๆ",
            "วันนี้คุณเก่งมากแล้วที่ผ่านมันมาได้พักผ่อนให้เต็มที่นะคับพรุ่งนี้ค่อยเริ่มกันใหม่อีกวันนะ ฉันอยู่ตรงนี้เสมอ "
        ])

    if any(word in message_lower for word in ["เครียด", "กดดัน", "ท้อ", "หมดแรง"]):
        return random.choice([
            "ฉันรับฟังคุณนะ รู้สึกแบบนี้เป็นเรื่องปกติ ลองหายใจลึก ๆ ช้า ๆ แล้วคิดถึงสิ่งดี ๆ ที่มีในชีวิต ฉันอยู่เคียงข้างคุณ ",
            "ใจเย็นๆ นะคับ BAB อยู่ตรงนี้เสมอ ถ้าไม่สบายใจระบายออกมาได้หมดเลยนะ",
            "ไม่เป็นไรนะที่จะรู้สึกเหนื่อยบ้าง พักสักนิดแล้วค่อยๆ แก้ไปทีละอย่างนะคับ ผมเชื่อว่าคุณทำได้ "
        ])

    if any(word in message_lower for word in ["ช่วย", "อันตราย", "ตาย"]):
        return "ฉันห่วงคุณนะ ถ้ารู้สึกไม่ปลอดภัย กรุณาโทร 1323 (สายด่วนสุขภาพจิต) ตลอด 24 ชั่วโมง เขาจะช่วยคุณได้ "

    if any(word == message_lower.strip() or word in message_lower for word in ["โอเคคับ", "โอเคค่ะ", "โอเค", "เคคับ", "เคค่ะ", "เข้าใจแล้ว", "เข้าใจคับ", "เข้าใจแล้วคับ"]):
        return "โอเคคับ ถ้ามีอะไรให้ช่วยรึอยากเล่าอะไรก็บอกมาได้เลยคับ Bab พร้อมรับฟังคุณตรงนี้เสมอนะ"

    if any(word in message_lower for word in ["ขอบคุณนะ", "ขอบคุณ", "ขอบใจนะ", "ขอบใจมาก", "ขอบคุณที่รับฟัง", "ขอบใจ", "ซาบซึ้ง", "ซึ้งใจ"]):
        return random.choice([
            "ด้วยความยินดีคับ มีอะไรให้ช่วยรึอยากเล่าก็บอกมาได้เลยนะคับ Bab อยู่ตรงนี้เสมอนะ",
            "ได้เสมอเลยคับ มีอะไรให้ช่วยรึอยากเล่าก็บอกมาได้เลยนะคับ Bab อยู่ตรงนี้เสมอนะ",
            "ดีใจที่คุณโอเคขึ้นนะคับ มีอะไรให้ช่วยรึอยากเล่าก็บอกมาได้เลยนะคับ Bab อยู่ตรงนี้เสมอนะ",
            "ขอบคุณที่ให้ผมรับฟัง และช่วยเหลือคุณกับเรื่องที่คุณเจอมานะคับ มีอะไรให้ช่วยรึอยากเล่าก็บอกมาได้เลยนะคับ Bab อยู่ตรงนี้เสมอนะ"
        ])

    return random.choice(DEMO_RESPONSES)


def has_non_thai(text: str) -> bool:
    """
    ตรวจสอบว่า text มีตัวอักษรภาษาต่างประเทศหรือไม่
    อนุญาตเฉพาะ: ไทย (U+0E00-U+0E7F), ตัวเลข, เว้นวรรค, เครื่องหมายวรรคตอน, emoji
    ยกเว้นคำว่า 'bab' และ 'BAB'
    """
    # ลบคำที่อนุญาต (bab, BAB) ออกก่อนตรวจ
    cleaned = re.sub(r'\b[Bb][Aa][Bb]\b', '', text)
    # ตรวจหาตัวอักษร Latin (a-z, A-Z)
    if re.search(r'[a-zA-Z]', cleaned):
        return True
    # ตรวจหาตัวอักษร CJK (จีน, ญี่ปุ่น, เกาหลี)
    if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', cleaned):
        return True
    # ตรวจหาตัวอักษร Cyrillic, Arabic, Devanagari และอื่นๆ
    if re.search(r'[\u0400-\u04ff\u0600-\u06ff\u0900-\u097f]', cleaned):
        return True
    return False


def clean_non_thai(text: str) -> str:
    """
    ลบคำภาษาต่างประเทศออกจากข้อความ (ใช้เป็น last resort)
    เก็บไว้เฉพาะ: อักษรไทย, ตัวเลข, เครื่องหมายวรรคตอน, emoji, คำว่า bab/BAB
    """
    # แทนที่ bab/BAB ด้วย placeholder ก่อน
    text = re.sub(r'\b[Bb][Aa][Bb]\b', '§BAB§', text)
    # ลบคำภาษาต่างประเทศ (ลบทีละ "คำ" ที่เป็น Latin)
    text = re.sub(r'[a-zA-Z]+', '', text)
    # ลบตัวอักษร CJK, Cyrillic, Arabic ฯลฯ
    text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0900-\u097f]+', '', text)
    # คืน bab/BAB กลับมา
    text = text.replace('§BAB§', 'BAB')
    # ลบช่องว่างซ้ำ
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def build_system_prompt(user_message: str) -> str:
    """สร้าง system prompt แบบ dynamic โดยต่อท้ายด้วย RAG context"""
    rag_context = rag.build_context(user_message, top_k=2)
    if rag_context:
        return f"{BASE_SYSTEM_PROMPT}\n\n{rag_context}"
    return BASE_SYSTEM_PROMPT


def call_model(messages: list) -> str:
    """เรียก Qwen2.5 ผ่าน Ollama และคืนค่า text ที่ได้"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # 1. ตรวจสอบประโยคตอบรับ/เข้าใจ (Acknowledgment) ก่อนเลย เพื่อให้ได้คำตอบที่แน่นอน
        ack_phrases = ["โอเคคับ", "โอเคค่ะ", "โอเค", "เคคับ", "เคค่ะ", "เข้าใจแล้ว", "เข้าใจคับ", "เข้าใจแล้วคับ"]
        thank_phrases = ["ขอบคุณนะ", "ขอบคุณ", "ขอบใจนะ", "ขอบใจมาก", "ขอบคุณที่รับฟัง", "ขอบใจ", "ซาบซึ้ง", "ซึ้งใจ"]
        msg_strip = request.message.strip().lower()
        if any(phrase == msg_strip for phrase in ack_phrases):
            return ChatResponse(reply="โอเคคับ ถ้ามีอะไรให้ช่วยรึอยากเล่าอะไรก็บอกมาได้เลยคับ Bab พร้อมรับฟังคุณตรงนี้เสมอนะ")
        if any(phrase in msg_strip for phrase in thank_phrases):
            return ChatResponse(reply=random.choice([
                "ด้วยความยินดีคับ มีอะไรให้ช่วยรึอยากเล่าก็บอกมาได้เลยนะคับ Bab อยู่ตรงนี้เสมอนะ",
                "ได้เสมอเลยคับ มีอะไรให้ช่วยรึอยากเล่าก็บอกมาได้เลยนะคับ Bab อยู่ตรงนี้เสมอนะ",
                "ดีใจที่คุณโอเคขึ้นนะคับ มีอะไรให้ช่วยรึอยากเล่าก็บอกมาได้เลยนะคับ Bab อยู่ตรงนี้เสมอนะ",
                "ขอบคุณที่ให้ผมรับฟัง และช่วยเหลือคุณกับเรื่องที่คุณเจอมานะคับ มีอะไรให้ช่วยรึอยากเล่าก็บอกมาได้เลยนะคับ Bab อยู่ตรงนี้เสมอนะ"
            ]))

        # Demo mode
        if demo_mode:
            reply = get_demo_response(request.message)
            return ChatResponse(reply=reply)

        # สร้าง system prompt พร้อม RAG context
        dynamic_system_prompt = build_system_prompt(request.message)

        # สร้าง messages สำหรับ OpenAI-compatible API (Ollama)
        # ใส่ตัวอย่างบทสนทนาภาษาไทย (Few-shot) เพื่อบังคับให้โมเดลตอบภาษาไทย
        messages = [
            {"role": "system", "content": dynamic_system_prompt},
            # Few-shot examples — บังคับโมเดลให้ตอบภาษาไทยตาม pattern
            {"role": "user", "content": "สวัสดี"},
            {"role": "assistant", "content": "สวัสดีคับ! ผมชื่อ bab ยินดีที่ได้รู้จักนะคับ วันนี้เป็นยังไงบ้างคับ มีอะไรอยากเล่าให้ฟังไหมคับ?"},
            {"role": "user", "content": "ฉันเครียดมากเลย"},
            {"role": "assistant", "content": "ขอบคุณที่เล่าให้ฟังนะคับ รู้สึกเครียดแบบนี้ไม่ใช่เรื่องแปลกเลย ลองหายใจลึกๆ ช้าๆ สักครู่นะคับ แล้วค่อยๆ เล่าให้ฟังได้เลยว่ามีอะไรกดดันอยู่คับ"},
            {"role": "user", "content": "ขอบคุณนะ"},
            {"role": "assistant", "content": "ไม่เป็นไรเลยคับ ผมอยู่ตรงนี้เสมอนะคับ ถ้าอยากระบายหรือคุยเรื่องอะไร บอกได้เลยคับ"},
        ]

        # เพิ่ม history จริง (ต่อจาก few-shot)
        for msg in (request.history or []):
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        # เพิ่มข้อความปัจจุบัน
        messages.append({
            "role": "user",
            "content": request.message
        })

        # เรียก Qwen2.5-7B ผ่าน Ollama
        reply = call_model(messages)

        # ถ้า model ตอบมีภาษาต่างประเทศ → สั่งให้ตอบใหม่เป็นภาษาไทยล้วนๆ
        if has_non_thai(reply):
            retry_messages = messages + [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": (
                        "คำตอบของคุณมีภาษาต่างประเทศผสมอยู่ซึ่งผิดกฎ "
                        "กรุณาตอบใหม่เป็นภาษาไทยทั้งหมด "
                        "ห้ามมีตัวอักษรภาษาอังกฤษ สเปน จีน หรือภาษาอื่นใด "
                        "ยกเว้นคำว่า bab เท่านั้น ทุกคำต้องเป็นภาษาไทยล้วนๆ"
                    )
                },
            ]
            reply = call_model(retry_messages)

            # ถ้า retry แล้วยังมีภาษาต่างประเทศอยู่ → ลบออกเลย (last resort)
            if has_non_thai(reply):
                reply = clean_non_thai(reply)

        return ChatResponse(reply=reply)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "companion": "bab 🌿",
        "rag_knowledge_base_size": rag.size
    }