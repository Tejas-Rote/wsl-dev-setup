import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision

# -----------------------------
# Setup
# -----------------------------
print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# Enable mixed precision (RTX 3060 sweet spot)
mixed_precision.set_global_policy("mixed_float16")

# Prevent TF from grabbing all VRAM
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# -----------------------------
# 1. CPU vs GPU matmul test
# -----------------------------
size = 6000

def matmul_test(device):
    with tf.device(device):
        a = tf.random.normal([size, size])
        b = tf.random.normal([size, size])
        start = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()  # force execution
        return time.time() - start

cpu_time = matmul_test("/CPU:0")
gpu_time = matmul_test("/GPU:0")

print(f"CPU matmul time : {cpu_time:.3f}s")
print(f"GPU matmul time : {gpu_time:.3f}s")
print(f"Speedup        : {cpu_time / gpu_time:.1f}x")

# -----------------------------
# 2. Simple neural network training
# -----------------------------
x = np.random.rand(50000, 100).astype("float32")
y = np.sum(x, axis=1, keepdims=True)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

start = time.time()
history = model.fit(
    x, y,
    batch_size=1024,
    epochs=5,
    verbose=1
)
train_time = time.time() - start

print(f"Training time (5 epochs): {train_time:.2f}s")

# -----------------------------
# 3. Plot loss curve (UI test)
# -----------------------------
plt.plot(history.history["loss"])
plt.title("Training Loss (GPU)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()
