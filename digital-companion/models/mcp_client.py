# ==============================================
# ไฟล์: models/mcp_client.py
# หน้าที่: MCP Client — เชื่อมต่อกับ mcp_server.py
#          และให้ routers/chat.py เรียกใช้งาน
# วิธีทำงาน:
#   1. รัน mcp_server.py เป็น subprocess
#   2. ส่ง request ไปยัง server และรับผลลัพธ์กลับ
#   3. นำผลลัพธ์ใส่ใน context ก่อนส่งให้ AI (Qwen)
# ==============================================

import asyncio
import json
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ใช้ path แบบ absolute เพื่อให้รันจากที่ไหนก็ได้ไม่มีปัญหา
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MCP_SERVER_PATH = os.path.join(BASE_DIR, "mcp_server.py")


async def call_mcp_tool(tool_name: str, arguments: dict = {}) -> dict:
    """
    เรียกใช้งาน Tool บน MCP Server (Async)
    """
    server_params = StdioServerParameters(
        command="python",
        args=[MCP_SERVER_PATH],
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                if result.content:
                    raw = result.content[0].text
                    try:
                        return json.loads(raw)
                    except Exception:
                        return {"raw": raw}
    except Exception as e:
        print(f"⚠️  MCP Client error: {e}")
    return {}


async def get_crisis_info() -> dict:
    """ดึงข้อมูลสายด่วนสุขภาพจิต"""
    return await call_mcp_tool("get_crisis_info")


async def get_relaxation_tip(stress_level: str = "medium") -> str:
    """ดึงเทคนิคผ่อนคลายตามระดับความเครียด"""
    result = await call_mcp_tool("get_relaxation_tip", {"stress_level": stress_level})
    return result.get("tip", "ลองหายใจลึกๆ ช้าๆ ก่อนนะ")


async def log_mood(mood: str, note: str = "") -> bool:
    """บันทึกอารมณ์ของ User"""
    result = await call_mcp_tool("log_mood", {"mood": mood, "note": note})
    return result.get("status") == "success"


# --------------------------------------------------
# ทดสอบการเชื่อมต่อ: python models/mcp_client.py
# --------------------------------------------------
if __name__ == "__main__":
    async def test():
        print("=== ทดสอบ MCP Client (Async) ===\n")
        print("1. get_crisis_info:", await get_crisis_info())
        print("2. get_relaxation_tip:", await get_relaxation_tip("high"))
        print("3. log_mood:", await log_mood("ทดสอบ", "รันจาก terminal"))

    asyncio.run(test())
