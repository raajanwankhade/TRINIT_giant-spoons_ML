import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor

def denormalize_pixel_values(pixel_values):
    """
    Denormalizes pixel values of an image tensor.

    Args:
        pixel_values (torch.Tensor): A tensor representing the normalized pixel values of an image.
                                     Shape: (channels, height, width).

    Returns:
        PIL.Image.Image: An image with denormalized pixel values.

    Note:
        This function assumes that the input tensor has been normalized using the same mean
        and standard deviation values as used during the normalization process.
    """
    unnormalized_image = (pixel_values.numpy() * np.array(STD)[:, None, None]) + np.array(MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    return Image.fromarray(unnormalized_image)

def model_loading():
    """
    Load our fine-tuned image captioning model from HF.

    Returns:
        Pre-trained image captioning model loaded from the Hugging Face model hub.

    Note:
        This function loads the image captioning model pre-trained on the RSICD dataset.
    """
    model_captioner = AutoModelForCausalLM.from_pretrained("deepakachu/rsicd_image_captioning")
    model_captioner.to(device)
    return model_captioner

def make_inference(image, model):
    """
    Generate a caption for the given image using the provided image captioning model.

    Args:
        image (PIL.Image.Image): The input image for which the caption is to be generated.
        model: Fine-tuned image captioning model.

    Note:
        This function resizes the input image to (224, 224) before applying the transformation.
        It then generates a caption using the provided image captioning model and displays the caption.
    """
    # resize the image before applying the transformation
    resized_image = image.resize((224, 224))
    # apply the transformation
    torch_tensor = transform(resized_image)
    # process the single image
    pixel_values = torch_tensor.unsqueeze(0).to(device)  # Add batch dimension
    # generate caption
    generated_ids = model.generate(pixel_values=pixel_values, max_length=100)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    st.markdown(
        f"<b><p style='font-size:30px; text-align:center; color:#E74C3C;'>Generated Caption: {generated_caption}</p></b>",
        unsafe_allow_html=True
    )
# Load pre-trained model and initialize constants
processor = AutoProcessor.from_pretrained("microsoft/git-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = np.array([123.675, 116.280, 103.530]) / 255
STD = np.array([58.395, 57.120, 57.375]) / 255
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to Tensor
    transforms.Normalize(mean=MEAN, std=STD)  # Normalize the tensor
])

# Streamlit app
st.title("Image Captioning App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if an image is uploaded
if uploaded_file is not None:
    # Make inference when button is clicked
    if st.button("Generate Caption"):
        # Open and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Make inference
        captioner = model_loading()
        make_inference(image, captioner)
