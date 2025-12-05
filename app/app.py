
import streamlit as st
from PIL import Image
import utils

st.set_page_config(
    page_title="MoE-LoRA Texture Classification",
    page_icon="🎨",
    layout="centered"
)

st.title("MoE-LoRA Texture Classification")
st.markdown("### Texture Classification using Mixture of Experts and Low-Rank Adaptation")
st.markdown("Upload an image to classify its texture.")

# Checkpoint loader
with st.spinner("Loading Model..."):
    model = utils.load_model()

if model is None:
    st.error("Failed to load model. Please check checkpoints.")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display image
        st.image(image, caption='Uploaded Image', width="stretch")
        
        if st.button("Classify"):
            with st.spinner("Classifying..."):
                class_name, confidence = utils.predict(image, model)
            
            st.success("Classification Complete!")
            
            # Display Result
            st.metric(label="Predicted Texture", value=class_name)
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
            
    except Exception as e:
        st.error(f"Error processing image: {e}")
