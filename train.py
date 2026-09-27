import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten

# =========================================================
# MNIST NEURAL NETWORK TRAINING
# =========================================================

print("\n" + "=" * 55)
print("        MNIST NEURAL VISION - MODEL TRAINING")
print("=" * 55)

# ---------------------------------------------------------
# 1. Load MNIST dataset
# ---------------------------------------------------------

print("\n[1/5] Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Training images : {x_train.shape}")
print(f"Testing images  : {x_test.shape}")

# ---------------------------------------------------------
# 2. Normalize images
# ---------------------------------------------------------

print("\n[2/5] Normalizing pixel values...")

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ---------------------------------------------------------
# 3. Build Neural Network
# ---------------------------------------------------------

print("\n[3/5] Building neural network...")

model = Sequential([
    Input(shape=(28, 28)),

    Flatten(name="flatten"),

    Dense(
        128,
        activation="relu",
        name="hidden_layer_1"
    ),

    Dense(
        64,
        activation="relu",
        name="hidden_layer_2"
    ),

    Dense(
        10,
        activation="softmax",
        name="output_layer"
    )
])

# ---------------------------------------------------------
# 4. Compile model
# ---------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()

# ---------------------------------------------------------
# 5. Train model
# ---------------------------------------------------------

print("\n[4/5] Training model...")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=1
)

# ---------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------

print("\n[5/5] Evaluating model...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

# ---------------------------------------------------------
# 7. Save model
# ---------------------------------------------------------

model.save("mnist_model.keras")

print("\n" + "=" * 55)
print("       MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 55)
print("\nSaved file: mnist_model.keras")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("\nYou can now run:")
print("streamlit run app.py")
print("=" * 55 + "\n")

