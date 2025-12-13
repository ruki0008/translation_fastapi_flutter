from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

# tokenizer
tokenizer = AutoTokenizer.from_pretrained("./tokenizer-ja-en")

# load onnx encoder/decoder
encoder = ort.InferenceSession("./onnx-ja-en/encoder_model.onnx")
decoder = ort.InferenceSession("./onnx-ja-en/decoder_model.onnx")

# input text
text = "こんにちは"
inputs = tokenizer(text, return_tensors="np")

# 1. run encoder
encoder_outputs = encoder.run(
    None,
    {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
)

encoder_hidden_states = encoder_outputs[0]

# 2. prepare decoder start token
decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)

output_tokens = []

# 3. greedy decoding loop
for _ in range(50):
    decoder_outputs = decoder.run(
        None,
        {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": inputs["attention_mask"],
        }
    )

    logits = decoder_outputs[0][:, -1, :]  # 最後の Token の logits
    next_token = np.argmax(logits, axis=-1)

    output_tokens.append(int(next_token))
    decoder_input_ids = np.hstack([decoder_input_ids, next_token.reshape(1, -1)])

    if next_token == tokenizer.eos_token_id:
        break

# decode output
translated_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
print("翻訳:", translated_text)