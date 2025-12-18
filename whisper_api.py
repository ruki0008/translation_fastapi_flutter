
from fastapi import FastAPI, UploadFile, File
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(api_key=os.getenv('WHISPER_API'))

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # 一時的にファイルを保存
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    # Whisper APIを呼び出す
    with open(temp_filename, "rb") as audio_data:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_data
        )

    # 一時ファイルを削除
    os.remove(temp_filename)
    
    return {"text": transcript.text}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)