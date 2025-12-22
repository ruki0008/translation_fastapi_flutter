from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
from openai import OpenAI
import os
import requests
from dotenv import load_dotenv

# --- .env を読み込む ---
load_dotenv()

app = FastAPI()

# --- OpenAI (Whisper) ---
client = OpenAI(api_key=os.getenv("WHISPER_API"))

# --- Azure Translator ---
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION")
AZURE_TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com/translate"

# --- ONNX モデルロード ---
tokenizer = AutoTokenizer.from_pretrained("./tokenizer-ja-en")
encoder = ort.InferenceSession("./onnx-ja-en/encoder_model.onnx")
decoder = ort.InferenceSession("./onnx-ja-en/decoder_model.onnx")

# --- Pydantic モデル ---
class TranslateRequest(BaseModel):
    text: str

# --- ONNX 翻訳関数 ---
def translate_onnx(text: str) -> str:
    inputs = tokenizer(text, return_tensors="np")

    encoder_outputs = encoder.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
    )
    encoder_hidden_states = encoder_outputs[0]

    decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
    output_tokens = []

    for _ in range(50):
        decoder_outputs = decoder.run(
            None,
            {
                "input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": inputs["attention_mask"],
            }
        )
        logits = decoder_outputs[0][:, -1, :]
        next_token = np.argmax(logits, axis=-1)

        output_tokens.append(int(next_token))
        decoder_input_ids = np.hstack([decoder_input_ids, next_token.reshape(1, -1)])

        if next_token == tokenizer.eos_token_id:
            break

    translated_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return translated_text

# --- Azure 翻訳関数 ---
def translate_azure(text: str, from_lang="ja", to_lang="en") -> str:
    params = {"api-version": "3.0", "from": from_lang, "to": to_lang}
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
        "Content-Type": "application/json",
    }
    body = [{"text": text}]

    response = requests.post(
        AZURE_TRANSLATOR_ENDPOINT,
        params=params,
        headers=headers,
        json=body
    )
    response.raise_for_status()
    return response.json()[0]["translations"][0]["text"]

# =======================
# --- API エンドポイント ---
# =======================

# 1. テキスト → ONNX 翻訳
@app.post("/speech/onnx")
async def speech_onnx(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    # Whisperで文字起こし
    with open(temp_filename, "rb") as audio_data:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_data
        )

    os.remove(temp_filename)
    translated = translate_onnx(transcript.text)

    return {"transcript": transcript.text, "translation": translated}

# 2. Whisper → ONNX 翻訳 (テキスト受け取り)
@app.post("/whisper/onnx")
def whisper_onnx(req: TranslateRequest):
    translated = translate_onnx(req.text)
    return {"translation": translated}

# 3. Whisper → Azure 翻訳
@app.post("/whisper/azure")
async def whisper_azure(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    # Whisperで文字起こし
    with open(temp_filename, "rb") as audio_data:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_data
        )

    os.remove(temp_filename)
    translated_text = translate_azure(transcript.text)

    return {"transcript": transcript.text, "translation": translated_text}