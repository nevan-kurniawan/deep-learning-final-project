import streamlit as st
from PIL import Image
import utils
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import numpy as np
import logging

# Configure logging at the top of your script
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="MoE-LoRA Texture Classification",
    page_icon="🎨",
    layout="centered"
)
@st.cache_data(show_spinner=False)
def load_and_preprocess_image(file_content):
    logger.info("Actually processing the image bytes now...")
    return Image.open(file_content).convert('RGB')

@st.cache_data(show_spinner=False)
def cached_predict(_model, _image):
    logger.info("Executing model prediction (not from cache).")
    return utils.predict(_image, _model)

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
    logger.info("--- Rerunning Tab 1 ---")
    st.markdown("Upload an image to classify its texture.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    image_placeholder = st.empty()
    button_placeholder = st.empty()
    result_placeholder = st.container() 

if uploaded_file is not None:
    logger.info(f"File detected: {uploaded_file.name}")
    try:
        # 1. Load image using your cached function
        image = load_and_preprocess_image(uploaded_file)
        image_placeholder.image(image, caption='Uploaded Image', width='stretch')
        
        # 2. Initialize necessary session state keys
        if 'prediction' not in st.session_state:
            st.session_state['prediction'] = None
        if 'last_uploaded_file' not in st.session_state:
            st.session_state['last_uploaded_file'] = None
        if 'do_predict' not in st.session_state:
            st.session_state['do_predict'] = False

        # 3. Check for file changes and reset state if needed
        if st.session_state['last_uploaded_file'] != uploaded_file.name:
            logger.info(f"New file detected. Resetting prediction. (Old: {st.session_state['last_uploaded_file']}, New: {uploaded_file.name})")
            st.session_state['prediction'] = None
            st.session_state['do_predict'] = False  # Ensure we don't auto-predict on upload
            st.session_state['last_uploaded_file'] = uploaded_file.name

        # 4. Classification Button - Just sets a trigger
        if button_placeholder.button("Classify"):
            logger.info(f"Classify button clicked for file: {uploaded_file.name}")
            st.session_state['do_predict'] = True

        # 5. Perform prediction only if the trigger is True
        if st.session_state['do_predict']:
            with st.spinner("Classifying..."):
                # Use your cached_predict function
                class_name, confidence = cached_predict(model, image)
                st.session_state['prediction'] = (class_name, confidence)
                
                # IMPORTANT: Reset the trigger immediately
                # This stops the 1-second "heartbeat" reruns from re-entering this block
                st.session_state['do_predict'] = False
                logger.info(f"Classification complete and trigger reset: {class_name}")

        # 6. Display Result independently of the button state
        if st.session_state['prediction']:
            class_name, confidence = st.session_state['prediction']
            logger.info(f"Displaying stored prediction for {uploaded_file.name}")
            with result_placeholder:
                st.success("Classification Complete!")
                st.metric(label="Predicted Texture", value=class_name)
                st.metric(label="Confidence", value=f"{confidence:.2f}%")
        else:
            logger.info("No prediction stored in session state yet.")
            
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
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
