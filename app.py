import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="YOLO Medical Detector",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 YOLO Medical Detector")
st.write(
    "Upload a medical image (X-ray / scan) and run YOLOv8 detection.\n\n"
    "**Disclaimer:** This is a demo, not a clinical tool."
)

@st.cache_resource
def load_model():
    # make sure yolov8n.pt is in the same folder as app.py
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

conf_thres = st.slider(
    "Confidence threshold",
    0.1, 0.9, 0.3, 0.05
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Original image")
    st.image(image, use_column_width=True)

    img_np = np.array(image)

    with st.spinner("Running YOLO detection..."):
        results = model(img_np, conf=conf_thres)
    result = results[0]

    # annotated image (BGR -> RGB)
    annotated = result.plot()[:, :, ::-1]

    st.subheader("Detections")
    st.image(annotated, use_column_width=True)

    if result.boxes is not None and len(result.boxes) > 0:
        st.subheader("Detection details")
        for box in result.boxes:
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            xyxy = box.xyxy[0].tolist()

            class_name = (
                model.names.get(cls_id, f"class_{cls_id}")
                if hasattr(model, "names") else str(cls_id)
            )

            st.markdown(
                f"- **Class:** `{class_name}` | "
                f"**Conf:** `{conf:.2f}` | "
                f"**Box:** `{[round(v, 1) for v in xyxy]}`"
            )
    else:
        st.info("No objects detected at this threshold.")
else:
    st.info("👆 Upload an image to start.")
