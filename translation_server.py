from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

# --- モデルロード ---
tokenizer = AutoTokenizer.from_pretrained("./tokenizer-ja-en")
encoder = ort.InferenceSession("./onnx-ja-en/encoder_model.onnx")
decoder = ort.InferenceSession("./onnx-ja-en/decoder_model.onnx")

app = FastAPI()

class TranslateRequest(BaseModel):
    text: str

# --- 翻訳関数 ---
def translate(text: str) -> str:
    # トークン化
    inputs = tokenizer(text, return_tensors="np")

    # Encoder 実行
    encoder_outputs = encoder.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
    )
    encoder_hidden_states = encoder_outputs[0]

    # Decoder 初期化
    decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
    output_tokens = []

    # Greedy デコーディング
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

# --- API エンドポイント ---
@app.post("/speech/onnx")
def translate_endpoint(req: TranslateRequest):
    translated = translate(req.text)
    return {"translation": translated}