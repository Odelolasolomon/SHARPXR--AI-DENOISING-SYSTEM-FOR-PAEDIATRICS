# ...existing code...
import argparse
import os
import sys
from typing import Optional

import torch

# Ensure project root is on path so `models` can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Try import model class
try:
    from models.dual_decoder import DualDecoderHybrid
except Exception as e:
    DualDecoderHybrid = None
    PRINT_IMPORT_ERROR = e

def load_checkpoint(path: str, device: str = "cpu"):
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    print("Loaded checkpoint type:", type(ckpt))
    return ckpt

def _is_state_dict_like(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    cnt = 0
    total = 0
    for v in d.values():
        total += 1
        try:
            if torch.is_tensor(v):
                cnt += 1
        except Exception:
            pass
    return total >= 4 and (cnt / total) >= 0.2

def find_state_dict(ckpt) -> Optional[dict]:
    # If the checkpoint itself already looks like a state-dict (OrderedDict)
    if isinstance(ckpt, dict) and _is_state_dict_like(ckpt):
        return ckpt

    # Typical keys where state dict might live
    if isinstance(ckpt, dict):
        candidates = ["state_dict", "model_state_dict", "model", "net", "state"]
        for k in candidates:
            if k in ckpt:
                val = ckpt[k]
                if isinstance(val, dict) and _is_state_dict_like(val):
                    return val
                # if saved module object
                try:
                    if hasattr(val, "state_dict"):
                        return val.state_dict()
                except Exception:
                    pass
        # fallback: search nested values for state-dict-like objects
        for k, v in ckpt.items():
            if isinstance(v, dict) and _is_state_dict_like(v):
                print(f"Found state-dict-like object under key: {k}")
                return v

    # If ckpt is a nn.Module instance
    try:
        if hasattr(ckpt, "state_dict"):
            return ckpt.state_dict()
    except Exception:
        pass

    return None

def strip_prefixes(state: dict) -> dict:
    new = {}
    for k, v in state.items():
        nk = k
        # common prefixes to remove
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        new[nk] = v
    return new

def build_model_from_checkpoint(ckpt, device: str):
    if DualDecoderHybrid is None:
        raise RuntimeError(f"Cannot import DualDecoderHybrid: {PRINT_IMPORT_ERROR}")

    # Try instantiate with possible config stored in checkpoint
    model = None
    if isinstance(ckpt, dict):
        for key in ("model_args", "args", "config"):
            if key in ckpt and isinstance(ckpt[key], dict):
                try:
                    print(f"Instantiating DualDecoderHybrid with config from checkpoint key: {key}")
                    model = DualDecoderHybrid(**ckpt[key])
                    break
                except Exception as e:
                    print(f"Failed to instantiate with {key}: {e}")

    if model is None:
        try:
            model = DualDecoderHybrid()
            print("Instantiated DualDecoderHybrid() with default constructor.")
        except Exception as e:
            raise RuntimeError(
                "Failed to instantiate DualDecoderHybrid automatically. "
                "If your model requires constructor args, provide them or add them to the checkpoint under 'model_args'/'args'/'config'."
            ) from e

    state = find_state_dict(ckpt)
    if state is None:
        raise RuntimeError("No state_dict found in checkpoint. Provide a SharpXR denoising checkpoint (state_dict or checkpoint dict).")

    # normalize keys and try loads
    if isinstance(state, dict):
        print("Preparing state_dict for loading (normalizing keys)...")
        state = strip_prefixes(state)

    # try strict load first
    try:
        print("Attempting strict load_state_dict...")
        model.load_state_dict(state)
        print("Loaded checkpoint (strict=True).")
    except Exception as strict_err:
        print("Strict load failed:", strict_err)
        # try non-strict load with stripped keys
        try:
            print("Attempting non-strict load_state_dict (strict=False)...")
            res = model.load_state_dict(state, strict=False)
            missing = res.get("missing_keys", []) if isinstance(res, dict) else []
            unexpected = res.get("unexpected_keys", []) if isinstance(res, dict) else []
            print("Loaded with strict=False.")
            if missing:
                print("Missing keys:", missing[:10])
            if unexpected:
                print("Unexpected keys:", unexpected[:10])
            if missing:
                print("Warning: checkpoint missing parameters for the model. ONNX export may be invalid.")
        except Exception as final_err:
            print("Non-strict load also failed. Example state keys:", list(state.keys())[:40])
            raise RuntimeError("Failed to load checkpoint into DualDecoderHybrid. Use a matching SharpXR denoising checkpoint.") from final_err

    model.to(device).eval()
    return model

def export_onnx(checkpoint_path, onnx_path, device, input_channels, height, width, opset):
    device_t = torch.device(device if device in ["cpu", "cuda"] else device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = load_checkpoint(checkpoint_path, device=device)
    model = build_model_from_checkpoint(ckpt, device_t)

    dummy = torch.randn(1, input_channels, height, width, device=device_t)
    print("Tracing and exporting ONNX...")
    try:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            verbose=False,
        )
        print(f"Exported ONNX model to: {onnx_path}")
    except Exception as e:
        print("ONNX export failed:", e)
        raise

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export SharpXR PyTorch model to ONNX")
    p.add_argument("--checkpoint", "-c", default="checkpoints/denoising_training.pkl")
    p.add_argument("--onnx", "-o", default="models/sharpxr.onnx")
    p.add_argument("--device", "-d", default="cpu", help="cpu or cuda")
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--opset", type=int, default=12)
    args = p.parse_args()
    export_onnx(args.checkpoint, args.onnx, args.device, args.channels, args.height, args.width, args.opset)
# ...existing code...