# ...existing code...
import os
import sys
import io
import base64
import traceback
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import numpy as np
import torch

# Ensure project root is importable so "models" package resolves
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Goes up to SharpXR/
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# import your model class
try:
    from models.dual_decoder import DualDecoderHybrid
except Exception as e:
    DualDecoderHybrid = None
    IMPORT_ERROR = e

app = FastAPI(title="SharpXR PyTorch Inference")


_frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
_allow_all = os.getenv("ALLOW_ALL_ORIGINS", "false").lower() == "true"
origins = ["http://localhost:8000", "http://127.0.0.1:5500", "http://localhost:3000", _frontend_origin]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all else origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model path and device
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_denoiser.pt"))
DEVICE = torch.device("cpu")

def _is_state_dict_like(obj):
    if not isinstance(obj, dict):
        return False
    cnt = 0
    total = 0
    for v in obj.values():
        total += 1
        try:
            if torch.is_tensor(v):
                cnt += 1
        except Exception:
            pass
    return total >= 4 and (cnt / total) >= 0.2

def _find_state_dict(ckpt):
    # direct state-dict saved
    if isinstance(ckpt, dict) and _is_state_dict_like(ckpt):
        return ckpt
    # common wrapper keys
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "state"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # nested search
        for k, v in ckpt.items():
            if isinstance(v, dict) and _is_state_dict_like(v):
                return v
    # module instance?
    try:
        if hasattr(ckpt, "state_dict"):
            return ckpt.state_dict()
    except Exception:
        pass
    return None

def _strip_prefixes(state: dict):
    new = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        new[nk] = v
    return new

def _detect_base_channels(state_dict):
    """
    Detect the base channel count from the checkpoint state dict.
    Returns the base_channels value needed to instantiate the model correctly.
    """
    # Look for the first encoder layer weight to determine base channels
    for key in ['encoder.0.conv.0.weight', 'encoder.0.conv.0.0.weight']:
        if key in state_dict:
            # Shape is [out_channels, in_channels, h, w]
            out_channels = state_dict[key].shape[0]
            print(f"Detected base_channels from {key}: {out_channels}")
            return out_channels
    
    # Fallback: try to infer from bottleneck
    if 'bottleneck.conv.0.weight' in state_dict:
        bottleneck_channels = state_dict['bottleneck.conv.0.weight'].shape[0]
        # Bottleneck is typically base_channels * 8
        base = bottleneck_channels // 8
        if base > 0:
            print(f"Detected base_channels from bottleneck: {base}")
            return base
    
    # Default fallback
    print("Could not detect base_channels, using default: 32")
    return 32

def load_model(path):
    if DualDecoderHybrid is None:
        raise RuntimeError(f"Cannot import DualDecoderHybrid: {IMPORT_ERROR}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    print(f"Loading checkpoint from: {path}")
    ckpt = torch.load(path, map_location=DEVICE)
    
    state = _find_state_dict(ckpt)
    if state is None:
        # maybe the checkpoint is the full module saved
        if hasattr(ckpt, "__class__"):
            try:
                model = ckpt
                model.to(DEVICE).eval()
                print("Loaded checkpoint as full module instance")
                return model
            except Exception:
                pass
        raise RuntimeError("No state_dict found in checkpoint and checkpoint is not a Module instance.")

    # Normalize keys
    state = _strip_prefixes(state)
    
    # Detect the architecture from checkpoint
    base_channels = _detect_base_channels(state)
    
    # Try to instantiate model with detected base_channels
    try:
        # Check if DualDecoderHybrid accepts base_channels parameter
        import inspect
        sig = inspect.signature(DualDecoderHybrid.__init__)
        params = list(sig.parameters.keys())
        
        if 'base_channels' in params or 'base_ch' in params:
            # Model supports base_channels parameter
            try:
                model = DualDecoderHybrid(base_channels=base_channels)
                print(f"Instantiated model with base_channels={base_channels}")
            except:
                model = DualDecoderHybrid(base_ch=base_channels)
                print(f"Instantiated model with base_ch={base_channels}")
        else:
            # Try default constructor
            model = DualDecoderHybrid()
            print("Instantiated model with default constructor")
            
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate DualDecoderHybrid: {e}")

    # Load state dict with strict=False to handle any remaining mismatches
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
        print("Successfully loaded state dict")
    except Exception as e:
        # If strict=False still fails, try to manually match compatible layers
        print(f"Warning: Could not load full state dict: {e}")
        print("Attempting partial weight loading for compatible layers...")
        
        model_dict = model.state_dict()
        compatible_state = {}
        
        for k, v in state.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    compatible_state[k] = v
                else:
                    print(f"Skipping {k}: shape mismatch {v.shape} vs {model_dict[k].shape}")
            else:
                print(f"Skipping {k}: not in model")
        
        if len(compatible_state) == 0:
            raise RuntimeError("No compatible weights found between checkpoint and model!")
        
        model_dict.update(compatible_state)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(compatible_state)}/{len(state)} compatible weights")

    model.to(DEVICE).eval()
    print("Model loaded and set to eval mode")
    return model

# Global variables for model loading
model = None
LOAD_ERROR = None

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    # prevent server from starting silently broken; expose error at /health
    LOAD_ERROR = traceback.format_exc()
    print(f"Failed to load model:\n{LOAD_ERROR}")

def preprocess_image_bytes(data: bytes, size=(128, 128)):
    img = Image.open(io.BytesIO(data)).convert("L").resize(size, resample=Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)  # 1x1xHxW
    return tensor

def postprocess_tensor(tensor: torch.Tensor):
    out = tensor.detach().cpu().squeeze().numpy()
    out = ((out - out.min()) / (out.max() - out.min() + 1e-9) * 255.0).astype(np.uint8)
    pil = Image.fromarray(out)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, out.shape

@app.get("/health")
def health():
    if model is None:
        return JSONResponse(
            {
                "status": "error", 
                "error": "model not loaded", 
                "trace": LOAD_ERROR or "Unknown error"
            }, 
            status_code=500
        )
    return JSONResponse({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(), 
        "model_path": MODEL_PATH,
        "device": str(DEVICE)
    })

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check /health for details.")
    
    data = await file.read()
    try:
        x = preprocess_image_bytes(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    
    try:
        with torch.no_grad():
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")
    
    try:
        b64, shape = postprocess_tensor(out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Postprocessing failed: {e}")
    
    return JSONResponse({"shape": shape, "mask_png_b64": b64})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
