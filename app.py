import os
import torch
from PIL import Image
import gradio as gr
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True
    )
    model = model.eval().cuda().to(torch.bfloat16)
    return tokenizer, model

tokenizer, model = load_model()

def infer_image(image: Image.Image, prompt: str = "<image>\nConvert the document to Markdown.") -> str:
    image = image.convert("RGB")
    # Save uploaded image temporarily
    tmp_path = "temp_upload.jpg"
    image.save(tmp_path)
    # Run inference
    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=tmp_path,
        output_path=None,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=False,
        test_compress=False
    )
    return res

iface = gr.Interface(
    fn=infer_image,
    inputs=[gr.Image(type="pil"), gr.Textbox(lines=2, placeholder="Prompt (optional)")],
    outputs=gr.Textbox(lines=10),
    title="DeepSeek-OCR Demo",
    description="Upload an image and get OCR output using DeepSeek-OCR"
)

if __name__ == "__main__":
    iface.launch()
