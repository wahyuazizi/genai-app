

import os
from typing import List, Union
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# import huggingface
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from transformers import BlipProcessor, BlipForConditionalGeneration

# Konfigurasi Model
MODEL = {
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
    "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
    "blip-image": "Salesforce/blip-image-captioning-large"
}

# Load model text
@st.cache_resource
def load_model_text(model_name: str = "mistral-7b"):

    """
    Load Huggingface txt generation model
    """
    model_path = MODEL[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        load_in_4bit=True # Hemat memori
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024
    )

# Load model image
@st.cache_resource
def load_image_model():
    """
    Load Huggingface image captioning model
    """
    processor = BlipProcessor.from_pretrained(MODEL["blip-image"])
    model = BlipForConditionalGeneration.from_pretrained(
        MODEL["blip-image"],
    )

    return processor, model

def generate_text(prompt: str, pipe, temperature=0.7, max_length=2048):
    """
    Generate text from prompt
    """

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    outputs = pipe(
        prompt,
        do_sample=True,
        max_length=max_length,
        temperature=temperature,
    )

    return outputs[0]["generated_text"]

def generate_image_caption(image_url: str):
    """
    Generate caption for image
    """
    processor, model = load_image_model()
    
    # Download image
    response = requests.get(image_url)
    raw_image = Image.open(BytesIO(response.content)).convert('RGB')

    # Conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# UI dari streamlit
st.header("Huggingface LLM and Image Captioning Playground", divider="rainbow")

# Pilihan model
selected_model = st.radio(
    "Pilih Model: ",
    list(MODEL.keys())[:3],
    key="selected_model",
    horizontal=True
)

# Tab untuk text generation
freeform_tab, image_tab = st.tabs(["Text Generation", "Image Captioning"])

with freeform_tab:
    st.subheader("Text Generation")

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05
    )

    max_length = st.slider(
        "Max Length",
        min_value=10,
        max_value=4096,
        value=1024,
        step=100
    )

    prompt = st.text_area(
        "Masukkan Prompt disini ...",
        height=200,
        value="Sebutkan 5 nama hewan peliharaan yang lucu dan menggemaskan",
    )

    if st.button("Generate Text"):
        with st.spinner(f"Lagi proses dengan model {selected_model}..."):
            try:
                pipe = load_model_text(selected_model)
                response = generate_text(prompt, pipe, temperature, max_length)

                st.write(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")

with image_tab:
    st.subheader("Image Captioning")

    image_url = st.text_input(
        "Masukkan URL gambar disini ...",
        value="https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwzNjUyOXwwfDF8c2VhcmNofDF8fGZpc2h8ZW58MHx8fHwxNjg5NzYyNTQx&ixlib=rb-4.0.3&q=80&w=400",
    )

    if st.button("Generate Caption"):
        with st.spinner("Lagi proses ..."):
            try:
                st.image(image_url, width=400)
                caption = generate_image_caption(image_url)
                st.write("**Caption:**", caption)
            except Exception as e:
                st.error(f"Error: {str(e)}")
