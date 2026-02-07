DATA_ROOT_DIR = DATA_ROOT_DIR = r"c:\Users\HomePC\Documents\Research\MLC\SharpXR\SharpXR\data\data_xray"
IMAGE_SIZE = (256, 256)

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training settings
BATCH_SIZE = 1
SHUFFLE_TRAIN = True

# Model settings
MODEL_IN_CHANNELS = 1
MODEL_OUT_CHANNELS = 1
MODEL_FEATURES = [64, 128, 256, 512]

# Noise parameters
POISSON_SCALE_RANGE = (50, 100)
GAUSSIAN_STD_RANGE = (5, 15)

# Augmentation parameters
FLIP_PROB = 0.1
ROTATION_RANGE = (-1, 1)
BRIGHTNESS_RANGE = (1.0, 1.0)
CONTRAST_RANGE = (1.0, 1.0)

# Training hyperparameters
EPOCHS = 1
LEARNING_RATE = 1e-4
PATIENCE = 1
USE_PSNR_SSIM = False

# Save paths
BEST_MODEL_PATH = "best_denoiser.pt" 
TRAINING_SESSION_PATH = "denoising_training.pkl" 
SAVE_ALL_AS_PKL = False 