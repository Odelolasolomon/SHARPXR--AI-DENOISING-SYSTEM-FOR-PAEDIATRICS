import tensorflow as tf
import pickle
import numpy as np
import os

path = "checkpoints/denoising_training.pkl"

def inspect_tf_checkpoint():
    print(f"Inspecting: {path}")
    print(f"File size: {os.path.getsize(path)} bytes")
    
    # Try loading as pickle first
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print("\nPickle load successful!")
        print("Type:", type(data))
        if isinstance(data, dict):
            print("Keys:", list(data.keys()))
    except Exception as e:
        print("\nPickle load failed:", e)
    
    # Try as TF checkpoint
    try:
        checkpoint = tf.train.load_checkpoint(path)
        var_names = tf.train.list_variables(path)
        print("\nTF Checkpoint variables:")
        for name, shape in var_names:
            print(f"- {name}: {shape}")
    except Exception as e:
        print("\nTF checkpoint load failed:", e)

if __name__ == "__main__":
    inspect_tf_checkpoint()