

from dotenv import load_dotenv
import os

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from routers.chat import router as chat_router



app = FastAPI(
    title="Digital Companion — bab",
    description="AI เพื่อนเสมือนช่วยลดความเครียด",
    version="1.0.0"
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     
    allow_methods=["*"],      # อนุญาตทุก HTTP method (GET, POST, ...)
    allow_headers=["*"],      # อนุญาตทุก header
)


# Mount static files FIRST
app.mount(
    "/static",                  # URL path
    StaticFiles(directory="frontend"),  # โฟลเดอร์จริง
    name="static"
)


# Then route
@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


# Then router
app.include_router(chat_router)



if __name__ == "__main__":
    import uvicorn

    print("🌿 Digital Companion กำลังเริ่มต้น...")
    print("📡 เปิดเบราว์เซอร์ที่: http://localhost:5000")
    print("📖 ดู API docs ที่: http://localhost:5000/docs")
    print("กด Ctrl+C เพื่อหยุด server\n")

    uvicorn.run(
        "app:app",      
        host="0.0.0.0", # รับการเชื่อมต่อจากทุก IP
        port=5000,      # port ที่ใช้
        reload=True   
    )