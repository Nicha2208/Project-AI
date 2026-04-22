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
    "ฟังเหมือนคุณมีความเครียดเยอะนะ 😔 ลองนั่งสงบ ลองหายใจลึก ๆ แล้วคิดถึงสิ่งที่ดีที่สุด ผมอยู่เคียงข้างคุณ 🌿","บางครั้ง การพูดออกมาก็ช่วยได้ ลองคิดถึงคนที่ใจดี แล้วบอกเล่านะ ผมเชื่อว่าคุณจะผ่านไปได้",
    "ฉันเห็นคุณพยายามมาก 💪 ลองหาเวลาพักตัว ฟังเพลงที่ชื่นชอบ หรือทำกิจกรรมที่ทำให้สบายใจสิ","วันนี้อาจเหนื่อย แต่พรุ่งนี้ก็จะดีขึ้นแน่ 🌅 ลองนอนให้พอ ดื่มน้ำให้สม่ำเสมอนะ",
    "ขอบคุณที่แบ่งปันความรู้สึกกับฉันนะ ลองพักผ่อนหรือทำสิ่งที่ชอบดูนะ ฉันอยู่ตรงนี้เสมอ 🌿",
]

def get_demo_response(user_message: str) -> str:
    """สร้างตัวอย่างคำตอบจาก demo mode ที่เหมาะสมกับข้อความผู้ใช้"""

    message_lower = user_message.lower()

    # 1. เช็คเรื่องความรู้สึก
    if any(word in message_lower for word in ["ความรู้สึก", "รู้สึก"]):
        return "การได้พูดถึงความรู้สึกของตัวเองเป็นเรื่องดีนะครับ ลองบอกฉันหน่อยว่าคุณรู้สึกยังไงบ้าง? ฉันพร้อมรับฟังเสมอ 🌿"

    # 2. เช็คการทักทาย
    if any(word in message_lower for word in ["สวัสดี", "หวัดดี", "ไฮ", "ชื่ออะไร"]):
        return "สวัสดีครับ! ผมชื่อ bab (แบ๊บ) เป็นเพื่อนคู่คิดที่จะคอยรับฟังคุณนะ วันนี้คุณเป็นยังไงบ้าง?"

    # 3. เช็คเรื่องอาหาร (ใช้ random.choice ครอบ List ของประโยค)
    if any(word in message_lower for word in ["กิน", "ชอบ", "อร่อย", "อาหาร", "ขนม"]):
        return random.choice([
            "ว้าว การได้กินของอร่อยนี่ช่วยเติมพลังใจได้ดีที่สุดเลยค่ะ! 🍰 วันนี้ไปทานอะไรมาเหรอคะ เล่าให้ BAB ฟังหน่อยได้ไหม?",
            "ว้าว การได้กินของอร่อยนี่ช่วยเติมพลังใจได้ดีที่สุดเลยค่ะ! 🍰 วันนี้ไปทานอะไรมาเหรอคะ?",
            "กินของอร่อยที่ชอบช่วยให้หายเหนื่อยได้เยอะเลยนะ! บอก BAB หน่อยได้ไหมว่าคือเมนูอะไร? 😋",
            "แค่ได้ยินว่าคุณได้กินของอร่อย BAB ก็มีความสุขตามแล้วค่ะ! มื้อนี้ให้กี่คะแนนดีคะ? ✨"
        ])

    # 4. เรื่องดีๆ
    if any(word in message_lower for word in ["ดีใจ", "สำเร็จ", "เย้", "สอบผ่าน"]):
        return random.choice([
            "ยินดีด้วยนะ! 🎉 ฉันดีใจไปกับคุณด้วยจริงๆ ความพยายามของคุณเห็นผลแล้วนะ เก่งมากเลยค่ะ",
            "เย้! ฟังแล้วใจฟูแทนเลยค่ะ คุณทำดีที่สุดแล้วจริงๆ 🌟",
            "ดีใจด้วยที่สุดเลย! วันนี้ต้องเป็นวันที่ยอดเยี่ยมของคุณแน่ๆ ขอบคุณที่มาแชร์เรื่องดีๆ ให้ฟังนะคะ"
        ])

    # 5. ความเหนื่อยล้า
    if any(word in message_lower for word in ["หลับ", "นอน", "เหนื่อย", "ล้า"]):
        return random.choice([
            "ฟังเหมือนคุณพักผ่อนไม่เพียงพอ 😴 ลองพักผ่อนให้มากขึ้นนะ ฉันห่วงสุขภาพของคุณ",
            "เหนื่อยมาทั้งวันแล้ว หาเวลาเอนหลังสักนิดไหมคะ? BAB อยากให้คุณได้พักจริงๆ 🌿",
            "วันนี้คุณเก่งมากแล้วที่ผ่านมันมาได้ พักผ่อนให้เต็มที่นะคะ พรุ่งนี้ค่อยเริ่มกันใหม่ 🌅"
        ])

    # 6. ความเครียด
    if any(word in message_lower for word in ["เครียด", "กดดัน", "ท้อ", "หมดแรง"]):
        return random.choice([
            "ฉันรับฟังคุณนะ รู้สึกแบบนี้เป็นเรื่องปกติ ลองหายใจลึก ๆ ช้า ๆ แล้วคิดถึงสิ่งดี ๆ ที่มีในชีวิต ฉันอยู่เคียงข้างคุณ 🌿",
            "ใจเย็นๆ นะคะ BAB อยู่ตรงนี้เสมอถ้าไม่สบายใจระบายออกมาได้หมดเลยนะ",
            "ไม่เป็นไรนะที่จะรู้สึกเหนื่อยบ้าง พักสักนิดแล้วค่อยๆ แก้ไปทีละอย่างนะคะ ผมเชื่อว่าคุณทำได้ 💪"
        ])

    # 7. เคสฉุกเฉิน
    if any(word in message_lower for word in ["ช่วย", "อันตราย", "ตาย"]):
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