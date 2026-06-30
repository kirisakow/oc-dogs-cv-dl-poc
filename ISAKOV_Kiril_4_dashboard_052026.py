#!.venv/bin/python
"""Streamlit GUI for image classification inference with YOLO models."""
from utils import (
    generate_eigen_cam,
    list_files_from,
    WORDNET_ID_REGEX_PTRN,
)

from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import os
import streamlit as st

DEFAULT_MODEL = "models/CNN__from=yolo26m-cls__n_cls=120__n_eps=20__LR=1e-05__FT=2__stage2.pt"
LAYERS_TO_UNFREEZE = 2


@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


def main():
    st.set_page_config(page_title="YOLO Image Classification Inference", page_icon="🐕", layout="wide")

    col1, col2 = st.columns(2)

    with col1:
        st.header("📁 Image Selection")
        image_files = list_files_from(src_dir=["images", "runs/classify/predict"], ext=["jpg", "jpeg"])
        selected_path = st.selectbox(
            "Select Image:",
            options=[str(f) for f in image_files],
            format_func=lambda x: f"{Path(x).parent.name}/{Path(x).name}",
            disabled=False
        )
        if selected_path:
            try:
                pil_img = Image.open(selected_path)
                breed_label = WORDNET_ID_REGEX_PTRN.split(str(Path(selected_path).parent.name))[1]
            except Exception as e:
                st.error(f"Error opening selected file: {e}")
                pil_img = None
                breed_label = None
        else:
            pil_img = None
            breed_label = None
        if pil_img:
            st.subheader(breed_label)
            st.image(pil_img, width=640)

    with col2:
        st.header("🎯 Prediction")
        model_files = list_files_from(src_dir="models", ext="pt")
        if not model_files:
            st.warning("No model files found in models/ directory")
            st.stop()
        default_index = model_files.index(Path(DEFAULT_MODEL)) if Path(DEFAULT_MODEL) in model_files else 0
        selected_model_path = st.selectbox(
            "Available Models:", options=[str(f) for f in model_files],
            format_func=lambda x: os.path.basename(x),
            index=default_index
        )
        if selected_path and pil_img:
            with st.spinner("Loading model and running prediction..."):
                try:
                    model = load_model(selected_model_path)
                    class_names = model.names
                    img_array = np.array(pil_img)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    results = model.predict(source=pil_img, save=False, save_txt=False)
                    if len(results) > 0:
                        result = results[0]
                        # Show EigenCAM
                        with st.spinner("Generating EigenCAM..."):
                            try:
                                st.subheader("Prediction Results")
                                fig = generate_eigen_cam(img=img_bgr, model=model, layers_to_unfreeze=LAYERS_TO_UNFREEZE, task='cls')
                                st.pyplot(fig, width=640)
                            except Exception as cam_e:
                                st.error(f"Error generating EigenCAM: {cam_e}")
                        st.caption("EigenCAM Visualization")

                        # Display Top 5 Predictions below EigenCAM
                        probs = getattr(result, 'probs', None)
                        boxes = getattr(result, 'boxes', None)
                        if probs is not None:
                            top5 = getattr(probs, 'top5', None)
                            if top5 is not None:
                                top5_conf = probs.top5conf.tolist()
                                st.write("### Top 5 Predictions:")
                                for i, (cls_idx, conf) in enumerate(zip(top5, top5_conf), start=1):
                                    st.write(f"{i}. {class_names[cls_idx]}: {conf:.4f}")
                            else:
                                top1_idx = int(probs.top1)
                                st.write(f"**Predicted Class:** {top1_idx}")
                                st.write(f"**Confidence:** {float(probs.top1conf):.4f}")
                        elif boxes is not None and len(boxes) > 0:
                            st.write(f"**Predicted Class:** {int(boxes.cls[0])}")
                            st.write(f"**Confidence:** {float(boxes.conf[0]):.4f}")
                        else:
                            st.warning("No objects detected")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
