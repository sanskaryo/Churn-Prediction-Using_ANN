import tensorflow as tf

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs are available: {gpus}")
    else:
        print("No GPUs found. Please ensure you have a compatible GPU and the necessary drivers installed.")

check_gpu()