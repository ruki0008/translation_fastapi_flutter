import os
import requests
from fastapi import UploadFile
from openai import OpenAI

client = OpenAI(api_key=os.getenv("WHISPER_API"))

AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION")
AZURE_TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com/translate"

def translate(text: str, from_lang="ja", to_lang="en") -> str:
    params = {"api-version": "3.0", "from": from_lang, "to": to_lang}
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
        "Content-Type": "application/json"
    }
    body = [{"text": text}]
    response = requests.post(AZURE_TRANSLATOR_ENDPOINT, params=params, headers=headers, json=body)
    response.raise_for_status()
    return response.json()[0]["translations"][0]["text"]

async def transcribe_and_translate(file: UploadFile):
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    with open(temp_filename, "rb") as audio_data:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_data)

    os.remove(temp_filename)
    translated_text = translate(transcript.text)
    return transcript.text, translated_text