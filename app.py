"""
Face Mask Detection API
=======================
FastAPI backend serving a Faster R-CNN (ResNet50-FPN) model that detects
faces in an uploaded image and classifies them as:
  1 - with_mask
  2 - without_mask
  3 - mask_weared_incorrect

Endpoints:
  POST /predict         — Upload an image, get detections as JSON
  POST /predict/video   — Upload a video, get annotated video back
"""

import io
import os
import json
import tempfile
from typing import List

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "mask_detector.pth")
NUM_CLASSES = 4  # background + 3 classes
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {
    1: "with_mask",
    2: "without_mask",
    3: "mask_weared_incorrect",
}

# Color map for drawing bounding boxes on video frames (BGR for OpenCV)
COLOR_MAP_BGR = {
    "with_mask":            (128, 222, 74),   # Green
    "without_mask":         (113, 113, 248),  # Red
    "mask_weared_incorrect": (36, 191, 251),  # Amber/Yellow
}

LABEL_DISPLAY = {
    "with_mask":            "MASK",
    "without_mask":         "NO MASK",
    "mask_weared_incorrect": "INCORRECT",
}

# ─── Model Loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str) -> torch.nn.Module:
    """Build a Faster R-CNN and load the trained weights."""
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model weights not found at '{model_path}'. "
            "Set the MODEL_PATH environment variable or place the file in the working directory."
        )

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"✅  Model loaded from '{model_path}' on {DEVICE}")
    return model


# ─── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Face Mask Detection API",
    description="Detects faces and classifies mask usage (with_mask / without_mask / mask_weared_incorrect)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Video-Stats"],
)

# Allow large video uploads (up to 500 MB)
class LargeUploadMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request._body_size = 500 * 1024 * 1024  # 500 MB
        return await call_next(request)

app.add_middleware(LargeUploadMiddleware)

# Load model at startup
model = load_model(MODEL_PATH)
transform = T.Compose([T.ToTensor()])


# ─── Schemas ───────────────────────────────────────────────────────────────────

class Detection(BaseModel):
    label: str
    confidence: float
    bbox: List[float]  # [x_min, y_min, x_max, y_max]


class PredictionResponse(BaseModel):
    total_detections: int
    with_mask: int
    without_mask: int
    mask_weared_incorrect: int
    detections: List[Detection]


# ─── Helper: Run inference on a single frame (numpy BGR image) ─────────────────

def predict_frame(frame_bgr: np.ndarray, confidence_threshold: float = CONFIDENCE_THRESHOLD):
    """
    Run the Faster R-CNN model on a single BGR frame.
    Returns a list of dicts: [{ label, confidence, bbox }]
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    img_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    detections = []
    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        if score.item() < confidence_threshold:
            continue
        label_id = label.item()
        label_name = LABEL_MAP.get(label_id)
        if label_name is None:
            continue
        detections.append({
            "label": label_name,
            "confidence": round(score.item(), 4),
            "bbox": [round(c, 2) for c in box.tolist()],
        })

    return detections


def draw_detections_on_frame(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes, labels, and confidence scores on a frame."""
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        label_name = det["label"]
        confidence = det["confidence"]
        color = COLOR_MAP_BGR.get(label_name, (248, 189, 56))
        display_label = LABEL_DISPLAY.get(label_name, label_name)

        # Draw filled bounding box border (thicker for visibility)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        text = f"{display_label} {confidence * 100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        label_bg_y1 = max(y1 - th - 10, 0)
        cv2.rectangle(frame, (x1, label_bg_y1), (x1 + tw + 8, y1), color, -1)

        # Label text
        cv2.putText(
            frame, text,
            (x1 + 4, y1 - 4),
            font, font_scale, (10, 14, 26), thickness, cv2.LINE_AA
        )

    return frame


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def health_check():
    """Simple health-check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_PATH,
        "device": str(DEVICE),
    }


@app.get("/health", tags=["Health"])
async def health():
    """Alias health endpoint for container orchestrators."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and receive face-mask detections.

    Returns bounding boxes, labels, and confidence scores for every
    detection above the configured confidence threshold.
    """
    # ── Validate input ──────────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{file.content_type}'. Use JPEG, PNG, or WebP.",
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    # ── Inference ───────────────────────────────────────────────────────────
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    # ── Post-process ────────────────────────────────────────────────────────
    detections: List[Detection] = []
    counts = {"with_mask": 0, "without_mask": 0, "mask_weared_incorrect": 0}

    for box, label, score in zip(
        outputs["boxes"], outputs["labels"], outputs["scores"]
    ):
        if score.item() < CONFIDENCE_THRESHOLD:
            continue
        label_id = label.item()
        label_name = LABEL_MAP.get(label_id)
        if label_name is None:
            continue  # skip background or unknown

        counts[label_name] += 1
        detections.append(
            Detection(
                label=label_name,
                confidence=round(score.item(), 4),
                bbox=[round(c, 2) for c in box.tolist()],
            )
        )

    return PredictionResponse(
        total_detections=len(detections),
        with_mask=counts["with_mask"],
        without_mask=counts["without_mask"],
        mask_weared_incorrect=counts["mask_weared_incorrect"],
        detections=detections,
    )


@app.post("/predict/video", tags=["Prediction"])
async def predict_video(
    file: UploadFile = File(...),
    frame_skip: int = Query(default=3, ge=1, le=30, description="Process every Nth frame (1 = all frames)"),
):
    """
    Upload a video file and receive an annotated video with face-mask detections.

    The video is processed frame-by-frame. Each processed frame gets bounding boxes
    with labels and confidence scores drawn on it. Un-processed frames (skipped for
    performance) are included without annotations.

    Returns the processed video as a downloadable MP4 file.
    """
    # ── Validate input ──────────────────────────────────────────────────────
    valid_extensions = (".mp4", ".avi", ".mov", ".webm", ".mkv")
    filename = file.filename or ""
    if not any(filename.lower().endswith(ext) for ext in valid_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Use: {', '.join(valid_extensions)}",
        )

    # ── Save uploaded video to a temp file ──────────────────────────────────
    try:
        video_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read video: {exc}")

    # Create temp files for input and output
    input_tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(filename)[1] or ".mp4"
    )
    output_tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4"
    )
    input_tmp.write(video_bytes)
    input_tmp.close()
    output_tmp.close()

    try:
        # ── Open input video ────────────────────────────────────────────────
        cap = cv2.VideoCapture(input_tmp.name)
        if not cap.isOpened():
            raise HTTPException(
                status_code=400, detail="Could not open video file. The format may be unsupported."
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── Open output writer ──────────────────────────────────────────────
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_tmp.name, fourcc, fps, (width, height))

        if not writer.isOpened():
            cap.release()
            raise HTTPException(
                status_code=500, detail="Could not create output video writer."
            )

        # ── Process frames ──────────────────────────────────────────────────
        frame_idx = 0
        frames_processed = 0
        total_counts = {"with_mask": 0, "without_mask": 0, "mask_weared_incorrect": 0}
        last_detections = []  # Carry forward detections to skipped frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # Run inference on this frame
                detections = predict_frame(frame)
                last_detections = detections
                frames_processed += 1

                # Accumulate counts
                for det in detections:
                    label = det["label"]
                    if label in total_counts:
                        total_counts[label] += 1
            else:
                # Use last known detections for smooth visual continuity
                detections = last_detections

            # Draw detections on frame
            annotated_frame = draw_detections_on_frame(frame.copy(), detections)
            writer.write(annotated_frame)

            frame_idx += 1

        cap.release()
        writer.release()

        # ── Build stats ─────────────────────────────────────────────────────
        stats = {
            "total_frames": frame_idx,
            "frames_processed": frames_processed,
            "frame_skip": frame_skip,
            "fps": round(fps, 2),
            "with_mask": total_counts["with_mask"],
            "without_mask": total_counts["without_mask"],
            "mask_weared_incorrect": total_counts["mask_weared_incorrect"],
        }

        # ── Return annotated video ──────────────────────────────────────────
        return FileResponse(
            path=output_tmp.name,
            media_type="video/mp4",
            filename="mask_detection_output.mp4",
            headers={"X-Video-Stats": json.dumps(stats)},
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {exc}")
    finally:
        # Clean up input temp file (output is cleaned up by FileResponse)
        try:
            os.unlink(input_tmp.name)
        except OSError:
            pass


# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))  # 7860 is the HF Spaces default port
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
