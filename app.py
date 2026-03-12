"""
Face Mask Detection API
=======================
FastAPI backend serving a Faster R-CNN (ResNet50-FPN) model that detects
faces in an uploaded image and classifies them as:
  1 - with_mask
  2 - without_mask
  3 - mask_weared_incorrect
"""

import io
import os
from typing import List

import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
)

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


# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
