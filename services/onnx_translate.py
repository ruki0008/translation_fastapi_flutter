from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

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
        decoder_input_ids = np.hstack([decoder_input_ids, next_token.reshape(1, -1)])
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(output_tokens, skip_special_tokens=True)