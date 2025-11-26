import time
from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from typing import Optional
import torch

# =========================
# CONFIG
# =========================

DEFAULT_WEIGHTS_PATH = "best.pt"

WORKZONE_CLASSES = {
    0: "Cone",
    1: "Drum",
    2: "Barricade",
    3: "Barrier",
    4: "Vertical Panel",
    5: "Work Vehicle",
    6: "Worker",
    7: "Arrow Board",
    8: "Temporary Traffic Control Message Board",
    9: "Temporary Traffic Control Sign",
}


# =========================
# HELPERS
# =========================

@st.cache_resource
def load_model_cached(weights_bytes: bytes, suffix: str, device: str):
    """
    Cache YOLO model when weights are uploaded.
    """
    tmp_dir = Path(tempfile.gettempdir())
    tmp_path = tmp_dir / f"uploaded_yolo_weights{suffix}"
    with open(tmp_path, "wb") as f:
        f.write(weights_bytes)
    model = YOLO(str(tmp_path))
    # move to device once
    try:
        model.to(device)
    except Exception:
        pass
    return model


@st.cache_resource
def load_model_default(weights_path: str, device: str):
    """
    Cache YOLO model from a fixed path on disk.
    """
    model = YOLO(weights_path)
    try:
        model.to(device)
    except Exception:
        pass
    return model


def compute_workzone_score(class_ids: np.ndarray) -> float:
    """
    Simple work zone score between 0 and 1 based on presence of key classes.
    """
    if class_ids.size == 0:
        return 0.0

    score = 0.0
    # cones, drums, barriers
    if any(c in class_ids for c in [0, 1, 2, 3, 4]):
        score += 0.4
    # worker
    if any(c in class_ids for c in [6]):
        score += 0.3
    # vehicles, signs, arrow board
    if any(c in class_ids for c in [5, 7, 8, 9]):
        score += 0.3

    return float(min(score, 1.0))


def draw_workzone_banner(frame: np.ndarray, score: float) -> np.ndarray:
    """
    Draw a banner at the top of the frame with WORK ZONE or NO WORK ZONE.
    Color and text depend on the score.
    """
    if frame is None:
        return frame

    h, w = frame.shape[:2]
    banner_h = int(0.12 * h)

    # Decide label and color
    if score >= 0.6:
        label = "WORK ZONE - HIGH RISK"
        color = (0, 0, 255)  # red
    elif score >= 0.3:
        label = "WORK ZONE"
        color = (0, 165, 255)  # orange
    else:
        label = "NO WORK ZONE"
        color = (0, 128, 0)  # green

    # Banner
    cv2.rectangle(frame, (0, 0), (w, banner_h), color, thickness=-1)

    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    text_size, _ = cv2.getTextSize(label, font, scale, thickness)
    text_x = int((w - text_size[0]) / 2)
    text_y = int(banner_h * 0.7)

    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )

    return frame

def process_video_batch(
    input_path: Path,
    model: YOLO,
    conf: float,
    iou: float,
    device: str,
    max_frames: Optional[int] = None,
):
    """
    Batch mode.
    Run YOLO on a video and write an annotated output video.
    Returns:
      output_path, global_workzone_score, frame_count, fps, per_class_counts
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error("Could not open video.")
        return None, None, None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames is not None and max_frames > 0:
        frame_limit = min(total_frames, max_frames)
    else:
        frame_limit = total_frames

    tmp_dir = Path(tempfile.gettempdir())
    output_path = tmp_dir / "workzone_annotated_batch.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    progress = st.progress(0)
    st.write(
        f"Video info: {width}x{height} at {fps:.1f} fps, "
        f"{total_frames} frames, processing {frame_limit} frames."
    )

    frame_idx = 0
    workzone_scores = []
    class_counts = {cid: 0 for cid in WORKZONE_CLASSES.keys()}

    while True:
        if frame_idx >= frame_limit:
            break

        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=conf,
            iou=iou,
            verbose=False,
            device=device,
        )
        result = results[0]

        annotated_frame = result.plot()

        if result.boxes is not None and len(result.boxes) > 0:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            score = compute_workzone_score(cls_ids)
            workzone_scores.append(score)
            for cid in cls_ids:
                if cid in class_counts:
                    class_counts[cid] += 1
        else:
            score = 0.0
            workzone_scores.append(0.0)

        # Draw banner on top of annotated frame
        annotated_frame = draw_workzone_banner(annotated_frame, score)

        out.write(annotated_frame)

        frame_idx += 1
        if frame_limit > 0 and frame_idx % 5 == 0:
            progress.progress(min(frame_idx / frame_limit, 1.0))

    cap.release()
    out.release()
    progress.progress(1.0)

    if len(workzone_scores) > 0:
        global_score = float(np.mean(workzone_scores))
    else:
        global_score = 0.0

    return output_path, global_score, frame_idx, fps, class_counts


def run_live_preview(
    input_path: Path,
    model: YOLO,
    conf: float,
    iou: float,
    device: str,
    max_frames: Optional[int] = None,
):
    """
    Live preview mode.
    Process frame by frame and show them as they are processed.
    Does not save a video file, just displays.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error("Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames is not None and max_frames > 0:
        frame_limit = min(total_frames, max_frames)
    else:
        frame_limit = total_frames

    st.write(
        f"Video info: {width}x{height} at {fps:.1f} fps, "
        f"{total_frames} frames, live preview of {frame_limit} frames."
    )

    frame_placeholder = st.empty()
    info_placeholder = st.empty()
    progress = st.progress(0)

    workzone_scores = []
    class_counts = {cid: 0 for cid in WORKZONE_CLASSES.keys()}

    frame_idx = 0
    delay = 1.0 / fps if fps > 0 else 0.03

    while True:
        if frame_idx >= frame_limit:
            break

        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=conf,
            iou=iou,
            verbose=False,
            device=device,
        )
        result = results[0]
        annotated_frame = result.plot()

        if result.boxes is not None and len(result.boxes) > 0:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            score = compute_workzone_score(cls_ids)
            workzone_scores.append(score)
            for cid in cls_ids:
                if cid in class_counts:
                    class_counts[cid] += 1
        else:
            score = 0.0
            workzone_scores.append(0.0)

        annotated_frame = draw_workzone_banner(annotated_frame, score)

        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        if workzone_scores:
            current_score = float(np.mean(workzone_scores))
        else:
            current_score = 0.0

        info_placeholder.markdown(
            f"Frame {frame_idx + 1}/{frame_limit} | "
            f"Current work zone score: **{current_score:.2f}**"
        )

        frame_idx += 1
        if frame_limit > 0 and frame_idx % 5 == 0:
            progress.progress(min(frame_idx / frame_limit, 1.0))

        time.sleep(delay)

    cap.release()
    progress.progress(1.0)

    if workzone_scores:
        final_score = float(np.mean(workzone_scores))
    else:
        final_score = 0.0

    st.success(f"Finished live preview. Final work zone score: {final_score:.2f}")

    st.write("Approximate class counts (over processed frames):")
    rows = []
    for cid, name in WORKZONE_CLASSES.items():
        rows.append(f"{name}: {class_counts.get(cid, 0)}")
    st.text("\n".join(rows))

def resolve_device(device_choice: str) -> str:
    """
    Resolve the device string based on user choice.
    - 'GPU (cuda)': try cuda
    - 'CPU': cpu
    - 'Auto': cuda if available, else cpu
    """
    if device_choice == "GPU (cuda)":
        return "cuda"
    if device_choice == "CPU":
        return "cpu"

    # Auto
    try:
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    except Exception:
        return "cpu"


# =========================
# STREAMLIT UI
# =========================

def main():
    st.set_page_config(page_title="Work Zone YOLO Video Demo", layout="wide")
    st.title("Work Zone Detection - YOLO Video Demo")

    st.markdown(
        "Upload a video, choose YOLO weights, device, thresholds and run detection. "
        "You can use batch mode to save an annotated video or live preview to simulate real time."
    )

    # Sidebar - model and parameters
    st.sidebar.header("Model settings")

    # Device selector
    device_choice = st.sidebar.radio(
        "Device",
        ["Auto", "GPU (cuda)", "CPU"],
        index=0,
        help="GPU requires a working CUDA PyTorch install.",
    )
    device = resolve_device(device_choice)
    st.sidebar.text(f"Using device: {device}")

    use_uploaded_weights = st.sidebar.checkbox(
        "Upload YOLO weights (.pt)", value=False
    )

    uploaded_weights = None

    if use_uploaded_weights:
        uploaded_weights = st.sidebar.file_uploader(
            "Upload YOLO weights file",
            type=["pt"],
            help="Upload your trained YOLO .pt file",
        )
    else:
        st.sidebar.text("Using default weights path:")
        st.sidebar.code(DEFAULT_WEIGHTS_PATH, language="text")

    conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
    iou = st.sidebar.slider("IoU threshold", 0.1, 0.9, 0.5, 0.05)
    max_frames = st.sidebar.number_input(
        "Max frames to process (0 for all)",
        min_value=0,
        value=0,
        step=50,
        help="Useful for quick tests on long videos.",
    )

    mode = st.sidebar.radio(
        "Run mode",
        ["Batch (save annotated video)", "Live preview"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.info("Choose parameters before running detection.")

    # Main - video upload
    st.subheader("Upload video")

    video_file = st.file_uploader(
        "Video file",
        type=["mp4", "mov", "avi", "mkv"],
        help="Upload a sample work zone video.",
    )

    run_clicked = st.button("Run work zone detection", type="primary")

    if run_clicked:
        if video_file is None:
            st.warning("Please upload a video first.")
            return

        # Load model
        with st.spinner("Loading YOLO model..."):
            try:
                if use_uploaded_weights:
                    if uploaded_weights is None:
                        st.error("Please upload YOLO weights (.pt) in the sidebar.")
                        return
                    model = load_model_cached(
                        uploaded_weights.read(),
                        suffix=Path(uploaded_weights.name).suffix,
                        device=device,
                    )
                else:
                    if not Path(DEFAULT_WEIGHTS_PATH).exists():
                        st.error("Default weights path does not exist. Set it correctly.")
                        return
                    model = load_model_default(DEFAULT_WEIGHTS_PATH, device=device)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return

        st.success(f"Model loaded on device: {device}")

        # Save uploaded video to temp
        tmp_video_path = Path(tempfile.gettempdir()) / video_file.name
        with open(tmp_video_path, "wb") as f:
            f.write(video_file.getbuffer())

        frames_limit = None if max_frames == 0 else int(max_frames)

        if mode == "Batch (save annotated video)":
            st.info("Processing video in batch mode. This can take time for longer clips.")
            (
                output_path,
                global_score,
                processed_frames,
                fps,
                class_counts,
            ) = process_video_batch(
                input_path=tmp_video_path,
                model=model,
                conf=conf,
                iou=iou,
                device=device,
                max_frames=frames_limit,
            )

            if output_path is None:
                return

            st.subheader("Results")

            st.write(f"Processed frames: {processed_frames} at {fps:.1f} fps")
            st.write(f"Global work zone score: **{global_score:.2f}** (0 to 1)")

            st.write("Approximate class counts (over processed frames):")
            rows = []
            for cid, name in WORKZONE_CLASSES.items():
                rows.append(f"{name}: {class_counts.get(cid, 0)}")
            st.text("\n".join(rows))

            st.write("Annotated video:")
            st.video(str(output_path))

            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download annotated video",
                    data=f,
                    file_name="workzone_annotated.mp4",
                    mime="video/mp4",
                )
        else:
            st.info("Running live preview mode. Frames will appear as they are processed.")
            run_live_preview(
                input_path=tmp_video_path,
                model=model,
                conf=conf,
                iou=iou,
                device=device,
                max_frames=frames_limit,
            )


if __name__ == "__main__":
    main()