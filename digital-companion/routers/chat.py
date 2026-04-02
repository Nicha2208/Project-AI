from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os
from models.stress_detector import detect_stress_level
import random

router = APIRouter(prefix="/api", tags=["chat"])

# ตั้งค่า API Key
api_key = os.getenv("GEMINI_API_KEY", "")
demo_mode = os.getenv("DEMO_MODE", "false").lower() == "true"

if api_key:
    genai.configure(api_key=api_key)

SYSTEM_PROMPT = """คุณคือ "bab" — Digital Companion ผู้ช่วยด้านสุขภาพจิต
- พูดภาษาไทยเป็นกันเอง อบอุ่น ไม่เป็นทางการ
- รับฟังอย่างตั้งใจ ไม่ตัดสินความรู้สึก
- แนะนำกิจกรรมผ่อนคลายเมื่อเหมาะสม
- ตอบกระชับ 3-5 ประโยค
- ห้ามวินิจฉัยโรค ถ้าเสี่ยงให้แนะนำสายด่วน 1323"""


DEMO_RESPONSES = [
    "ฟังเหมือนคุณมีความเครียดเยอะนะ 😔 ลองนั่งสงบ ลองหายใจลึก ๆ แล้วคิดถึงสิ่งที่ดีที่สุด ผมอยู่เคียงข้างคุณ 🌿",
    "เข้าใจจ้า รู้สึกกดดันเวลาแบบนี้ก็เป็นเรื่องปกติ ลองลุกขึ้นเดิน ๆ หรือออกไปหายอากาศสิ ได้เวลา ผมอยู่ที่นี่นะ",
    "ฉันเห็นคุณพยายามมาก 💪 ลองหาเวลาพักตัว ฟังเพลงที่ชื่นชอบ หรือทำกิจกรรมที่ทำให้สบายใจสิ",
    "บางครั้ง การพูดออกมาก็ช่วยได้ ลองคิดถึงคนที่ใจดี แล้วบอกเล่านะ ผมเชื่อว่าคุณจะผ่านไปได้",
    "วันนี้อาจเหนื่อย แต่พรุ่งนี้ก็จะดีขึ้นแน่ 🌅 ลองนอนให้พอ ดื่มน้ำให้สม่ำเสมอนะ",
]

def get_demo_response(user_message: str) -> str:
    """สร้างตัวอย่างคำตอบจาก demo mode ที่เหมาะสมกับข้อความผู้ใช้"""
    message_lower = user_message.lower()
    
    # ตรวจสอบคำสำคัญและตอบตามบริบท
    if any(word in message_lower for word in ["หลับ", "นอน", "เหนื่อย", "ล้า"]):
        return "ฟังเหมือนคุณลา ๆ น้อยหน่อยนะ 😴 ลองพักผ่อนให้เพียงพอสิ นี่คือสิ่งสำคัญที่สุด ผมห่วงสุขภาพของคุณ"
    
    if any(word in message_lower for word in ["เครียด", "กดดัน", "ท้อ", "หมดแรง"]):
        return "เข้าใจจ้า รู้สึกกดดันแบบนี้เป็นเรื่องปกติ ลองหายใจลึก ๆ ช้า ๆ แล้วคิดถึงสิ่งดี ๆ ที่มีในชีวิต ผมอยู่เคียงข้างคุณ 🌿"
    
    if any(word in message_lower for word in ["รัก", "ฉัน", "ชอบ", "สวย", "ขอบคุณ"]):
        return "ขอบคุณนะ ✨ ฉันดีใจที่ได้พูดคุยกับคุณ ลองดูแลตัวเองให้ดี ๆ นะ จะมีวันที่ดีรอคุณยู่ข้างหน้า"
    
    if any(word in message_lower for word in ["ช่วย", "ฉีก", "ชำรุด", "อันตราย", "ตาย"]):
        return "ฉันห่วงคุณนะ ถ้ารู้สึกไม่ปลอดภัย กรุณาโทร 1323 (สายด่วนสุขภาพจิต) ตลอด 24 ชั่วโมง เขาจะช่วยคุณได้ 💙"
    
    # ตอบแบบทั่วไป
    return random.choice(DEMO_RESPONSES)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []


class ChatResponse(BaseModel):
    reply: str
    stress_analysis: dict


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        stress = detect_stress_level(request.message)

        # Demo mode - ใช้ตัวอย่างคำตอบ
        if demo_mode or not api_key:
            reply = get_demo_response(request.message)
            return ChatResponse(
                reply=reply,
                stress_analysis=stress
            )

        # Real mode - เรียก Gemini API
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT
        )

        # สร้าง history list จาก previous messages
        history = []
        for msg in (request.history or []):
            role = "user" if msg.role == "user" else "model"
            history.append({
                "role": role,
                "parts": [{"text": msg.content}]
            })

        # สร้าง chat session ด้วย history
        chat_session = model.start_chat(history=history)
        
        # ส่งข้อความใหม่และรับคำตอบ
        response = chat_session.send_message(request.message)

        return ChatResponse(
            reply=response.text,
            stress_analysis=stress
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")


@router.get("/health")
async def health():
    return {"status": "ok", "companion": "bab 🌿"}