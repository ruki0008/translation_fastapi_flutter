import os
from fastapi import UploadFile, Form
from openai import OpenAI
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

client = OpenAI(api_key=os.getenv("WHISPER_API"))

tokenizer = AutoTokenizer.from_pretrained("./tokenizer-ja-en")
encoder = ort.InferenceSession("./onnx-ja-en/encoder_model.onnx")
decoder = ort.InferenceSession("./onnx-ja-en/decoder_model.onnx")


def translate(text: str) -> str:
    inputs = tokenizer(text, return_tensors="np")

    encoder_outputs = encoder.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })
    encoder_hidden_states = encoder_outputs[0]

    decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
    output_tokens = []

    for _ in range(50):
        decoder_outputs = decoder.run(None, {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": inputs["attention_mask"],
        })

        logits = decoder_outputs[0][:, -1, :]
        next_token = np.argmax(logits, axis=-1)

        output_tokens.append(int(next_token))
        decoder_input_ids = np.hstack(
            [decoder_input_ids, next_token.reshape(1, -1)]
        )

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(output_tokens, skip_special_tokens=True)


async def transcribe_and_translate(
    file: UploadFile,
    prompt: str | None = Form(None),   # ← ★ Flutter から受け取る
):
    """音声を文字起こしして翻訳"""

    # 一時ファイル保存
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    # デフォルトプロンプト（何も来ない場合用）
    default_prompt = (
        "以下の音声には固有名詞が含まれます。"
        "登録された固有名詞は正確に出力してください。"
    )

    whisper_prompt = prompt if prompt else default_prompt

    # Whisper API
    with open(temp_filename, "rb") as audio_data:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_data,
            language="ja",
            prompt=whisper_prompt,   # ← ★ ここに適用
        )

    os.remove(temp_filename)

    translated_text = translate(transcript.text)

    return {
        "transcript": transcript.text,
        "translation": translated_text,
    }