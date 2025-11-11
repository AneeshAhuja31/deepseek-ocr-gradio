import gradio as gr
from transformers import AutoTokenizer, AutoModel
from PIL import Image

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()

def ocr_image(image):
    if image is None:
        return "No image uploaded."
    image = image.convert("RGB")
    result = model.infer(
        tokenizer,
        prompt="<image>\nExtract text:",
        image_file=image,
        output_path=None,
        base_size=768,
        image_size=512,
        crop_mode=True,
        save_results=False,
        test_compress=False
    )
    return result

demo = gr.Interface(
    fn=ocr_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Recognized Text"),
    title="DeepSeek-OCR Demo",
    description="Simple OCR demo using the DeepSeek-OCR model."
)

if __name__ == "__main__":
    demo.launch()
