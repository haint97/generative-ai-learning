
import requests
import numpy as np
import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import io
import base64
import tempfile
import os
from gtts import gTTS

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def get_image_from_url(url):
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        return image
    except Exception as e:
        return None

def generate_caption(image=None, image_url=None, prompt=None):
    raw_image = None
    image_preview = None
    if image is not None:
        raw_image = image
        image_preview = image
    elif image_url:
        raw_image = get_image_from_url(image_url)
        image_preview = raw_image
        if raw_image is None:
            return None, "Error loading image from URL.", None, None
    else:
        return None, "Please upload an image or enter an image URL.", None, None

    inputs = processor(raw_image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, output_scores=True, return_dict_in_generate=True)
    caption = processor.decode(output.sequences[0], skip_special_tokens=True)
    # Confidence score: use mean of max softmax scores for each token
    if hasattr(output, 'scores') and output.scores:
        # Get max probability for each token
        probs = [torch.nn.functional.softmax(s, dim=-1).max().item() for s in output.scores]
        confidence = round(np.mean(probs) * 100, 2)
    else:
        confidence = None

    # Audio output using gTTS
    tts_path = None
    try:
        tts = gTTS(text=caption, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            tts.save(f.name)
            tts_path = f.name
    except Exception:
        tts_path = None

    return image_preview, caption, confidence, tts_path

with gr.Blocks(title="Picture Caption Generator") as demo:
    gr.Markdown("# Picture Caption Generator\nUpload an image or enter an image URL to generate a caption using BLIP.")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Input Image")
            url_input = gr.Textbox(label="Image URL", placeholder="Paste image URL here")
            submit_btn = gr.Button("Generate Caption")
        with gr.Column():
            image_preview = gr.Image(label="Image Preview")
            caption_output = gr.Textbox(label="Generated Caption", placeholder="The generated caption will appear here")
            confidence_output = gr.Number(label="Caption Confidence Score")
            audio_output = gr.Audio(label="Audio Caption", type="filepath")


    def preview_image(image, image_url):
        if image is not None:
            return image
        elif image_url:
            img = get_image_from_url(image_url)
            return img
        return None

    image_input.change(
        fn=lambda img: preview_image(img, None),
        inputs=image_input,
        outputs=image_preview
    )
    url_input.change(
        fn=lambda url: preview_image(None, url),
        inputs=url_input,
        outputs=image_preview
    )

    def process_inputs(image, image_url):
        return generate_caption(image, image_url)

    submit_btn.click(
        process_inputs,
        inputs=[image_input, url_input],
        outputs=[image_preview, caption_output, confidence_output, audio_output]
    )

if __name__ == "__main__":
    demo.launch()
