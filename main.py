from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from services.whisper_azure import transcribe_and_translate as azure_transcribe
from services.whisper_onnx import transcribe_and_translate as onnx_transcribe
from services.onnx_translate import translate as onnx_translate_text

from starlette.concurrency import run_in_threadpool  # 追加

app = FastAPI()

class TranslateRequest(BaseModel):
    text: str

@app.post("/whisper/azure")
async def whisper_azure(file: UploadFile = File(...)):
    transcript, translation = await azure_transcribe(file)
    return {"transcript": transcript, "translation": translation}

# @app.post("/whisper/onnx")
# async def whisper_onnx(file: UploadFile = File(...)):
#     print("whisper/onnx にリクエスト到達")
#     # 同期処理をスレッドプールで実行
#     transcript, translation = await run_in_threadpool(onnx_transcribe, file)
#     return {"transcript": transcript, "translation": translation}

# from starlette.concurrency import run_in_threadpool

@app.post("/whisper/onnx")
async def whisper_onnx(file: UploadFile = File(...)):
    # 同期関数をスレッドプールで安全に実行
    transcript, translation = await run_in_threadpool(onnx_transcribe, file)
    return {"transcript": transcript, "translation": translation}

@app.post("/speech/onnx")
def text_onnx(req: TranslateRequest):
    translated = onnx_translate_text(req.text)
    return {"translation": translated}