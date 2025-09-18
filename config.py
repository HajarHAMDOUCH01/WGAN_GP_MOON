import torch

class Config:
    DATA_PATH = "/content/drive/MyDrive/moon"
    IMAGE_SIZE = 256
    BATCH_SIZE = 64  
    NUM_WORKERS = 2

    LATENT_DIM = 128
    G_CHANNELS = [512, 256, 128, 64, 32, 16, 3]
    D_CHANNELS = [64, 128, 256, 512, 1024, 1]

    NUM_EPOCHS = 300  
    LEARNING_RATE_G = 0.0001
    LEARNING_RATE_D = 0.0004  
    BETA1 = 0.0  
    BETA2 = 0.9  
    LAMBDA_GP = 10
    N_CRITIC = 5

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SAVE_INTERVAL = 50  
    LOG_INTERVAL = 100 
    SAMPLE_INTERVAL = 25  
    MODEL_SAVE_PATH = "/content/drive/MyDrive/models/"
    SAMPLES_SAVE_PATH = "/content/drive/MyDrive/samples/"

    USE_MIXED_PRECISION = True
    USE_SPECTRAL_NORM = True
    USE_SELF_ATTENTION = False

    