import os
from fastapi import UploadFile
from openai import OpenAI
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

# --- OpenAI Whisper API ---
client = OpenAI(api_key=os.getenv("WHISPER_API"))

# --- ONNX 翻訳モデルロード ---
tokenizer = AutoTokenizer.from_pretrained("./tokenizer-ja-en")
encoder = ort.InferenceSession("./onnx-ja-en/encoder_model.onnx")
decoder = ort.InferenceSession("./onnx-ja-en/decoder_model.onnx")


def translate(text: str) -> str:
    """ONNXモデルでテキスト翻訳"""
    inputs = tokenizer(text, return_tensors="np")

    # Encoder 実行
    encoder_outputs = encoder.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })
    encoder_hidden_states = encoder_outputs[0]

    # Decoder 初期化
    decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
    output_tokens = []

    # Greedy デコーディング
    for _ in range(50):
        decoder_outputs = decoder.run(None, {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": inputs["attention_mask"],
        })

        logits = decoder_outputs[0][:, -1, :]
        next_token = np.argmax(logits, axis=-1)

        output_tokens.append(int(next_token))
        decoder_input_ids = np.hstack([decoder_input_ids, next_token.reshape(1, -1)])

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(output_tokens, skip_special_tokens=True)


def transcribe_and_translate(file: UploadFile):
    """音声ファイルをWhisper APIで文字起こしし、ONNX翻訳"""
    # 一時ファイルに保存
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as f:
        f.write(file.file.read())  # 同期処理

    # Whisper APIで文字起こし
    with open(temp_filename, "rb") as audio_data:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_data
        )

    # 一時ファイル削除
    os.remove(temp_filename)

    # ONNX 翻訳
    translated_text = translate(transcript.text)

    return transcript.text, translated_text