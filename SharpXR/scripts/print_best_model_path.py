import pickle
import os

d = pickle.load(open("checkpoints/denoising_training.pkl", "rb"))
print("best_model_path:", d.get("best_model_path"))
print("exists:", os.path.exists(d.get("best_model_path") or ""))