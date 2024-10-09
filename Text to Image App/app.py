import streamlit as st
from PIL import Image
from authtoken import auth_token
import torch
from diffusers import StableDiffusionPipeline

# Streamlit app setup
st.title("Stable Bud")
st.write("Enter a prompt and generate an image with Stable Diffusion")

# Input for the prompt
prompt = st.text_input("Prompt", value="A scenic landscape with mountains and a river")

@st.cache_resource(show_spinner=False)
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, revision="fp16", 
        torch_dtype=torch.float32, 
        use_auth_token=auth_token
    )
    pipe.to("cpu")  # Running on CPU
    return pipe

# Load the Stable Diffusion model (cached)
pipe = load_model()

# Button to generate the image
if st.button("Generate"):
    if prompt.strip():
        with st.spinner("Generating image..."):
            try:
                # Generate the image using the prompt
                result = pipe(prompt, guidance_scale=8.5)
                image = result.images[0]
                
                # Save and display the image
                image.save('generated_image.png')
                img = Image.open("generated_image.png")
                st.image(img, caption="Generated Image")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a valid prompt")
