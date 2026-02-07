import os
import sys
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.dual_decoder import DualDecoderHybrid

def test_model_load():
    print("Python path:", sys.path)
    print("Working directory:", os.getcwd())
    
    # Test model import
    print("\nTesting model import...")
    model_class = DualDecoderHybrid
    print(f"Model class: {model_class}")
    
    # Test model instantiation
    print("\nTesting model creation...")
    model = DualDecoderHybrid()
    print(f"Created model: {model.__class__.__name__}")
    
    # Load checkpoint
    ckpt_path = os.path.join(project_root, "checkpoints", "best_denoiser.pt")
    print(f"\nLoading checkpoint: {ckpt_path}")
    print(f"File exists: {os.path.exists(ckpt_path)}")
    
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint type: {type(checkpoint)}")
    
    # Try loading weights
    print("\nLoading weights...")
    model.load_state_dict(checkpoint, strict=False)
    print("Weights loaded successfully")
    
    return model

if __name__ == "__main__":
    try:
        model = test_model_load()
        print("\nSUCCESS: Model loaded and ready")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise