import streamlit as st
from PIL import Image
import utils
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import numpy as np

st.set_page_config(
    page_title="MoE-LoRA Texture Classification",
    page_icon="🎨",
    layout="centered"
)

st.title("MoE-LoRA Texture Classification")
st.markdown("### Texture Classification using Mixture of Experts and Low-Rank Adaptation")

# Checkpoint loader
with st.spinner("Loading Model..."):
    model = utils.load_model()

if model is None:
    st.error("Failed to load model. Please check checkpoints.")
    st.stop()

tab1, tab2 = st.tabs(["Upload Image", "Live Prediction"])

with tab1:
    st.markdown("Upload an image to classify its texture.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    # Define placeholders to anchor elements and prevent layout jumping
    image_placeholder = st.empty()
    button_placeholder = st.empty()
    result_placeholder = st.container() 

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            
            # Use width='stretch' as per 2026 Streamlit standards
            image_placeholder.image(image, caption='Uploaded Image', width='stretch')
            
            # Initialize session state for persistence
            if 'prediction' not in st.session_state:
                st.session_state['prediction'] = None
            if 'last_uploaded_file' not in st.session_state:
                st.session_state['last_uploaded_file'] = None

            # Reset only if a truly new file is detected
            if st.session_state['last_uploaded_file'] != uploaded_file.name:
                st.session_state['prediction'] = None
                st.session_state['last_uploaded_file'] = uploaded_file.name

            # Classification Button
            if button_placeholder.button("Classify"):
                with st.spinner("Classifying..."):
                    class_name, confidence = utils.predict(image, model)
                    st.session_state['prediction'] = (class_name, confidence)
                
            # Display Result exactly as per original design
            if st.session_state['prediction']:
                class_name, confidence = st.session_state['prediction']
                with result_placeholder:
                    st.success("Classification Complete!")
                    st.metric(label="Predicted Texture", value=class_name)
                    st.metric(label="Confidence", value=f"{confidence:.2f}%")
                
        except Exception as e:
            st.error(f"Error processing image: {e}")

with tab2:
    st.header("Live Prediction")
    st.markdown("Ensure your camera is enabled. Models runs real-time.")

    class TextureVideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Convert to PIL for model (RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            
            try:
                # utility predict function returns (class_name, confidence)
                class_name, confidence = utils.predict(pil_image, model)
                text = f"{class_name}: {confidence:.1f}%"
                color = (0, 255, 0)
            except Exception as e:
                text = f"Error: {e}"
                color = (0, 0, 255)

            # Draw text on original BGR image
            cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="texture-classification",
        video_processor_factory=TextureVideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
