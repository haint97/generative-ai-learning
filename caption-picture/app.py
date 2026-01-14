
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(raw_image: Image) -> str:
    inputs = processor(raw_image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption



import requests

def generate_caption(image=None, image_url=None):
    raw_image = None
    if image is not None:
        raw_image = image
    elif image_url:
        try:
            raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
        except Exception as e:
            return f"Error loading image from URL: {e}"
    else:
        return "Please upload an image or enter an image URL."
    inputs = processor(raw_image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

image_input = gr.Image(label="Input Image")
url_input = gr.Textbox(label="Image URL", placeholder="Paste image URL here")
caption_output = gr.Textbox(label="Generated Caption", placeholder="The generated caption will appear here")

demo = gr.Interface(
    fn=generate_caption,
    inputs=[image_input, url_input],
    outputs=[caption_output],
    title="Picture Caption Generator",
    description="Upload an image or enter an image URL to generate a caption using BLIP."
)

if __name__ == "__main__":
    demo.launch()
