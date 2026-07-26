import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Rescaling
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(1)
np.random.seed(1)

train_real = glob.glob("data/train/REAL/*.jpg", recursive=True)
train_fake = glob.glob("data/train/FAKE/*.jpg", recursive=True)
test_real = glob.glob("data/test/REAL/*.jpg", recursive=True)
test_fake = glob.glob("data/test/FAKE/*.jpg", recursive=True)

print(len(train_real), len(train_fake), len(test_real), len(test_fake))


def load_images(image_paths):
    images = []
    for path in image_paths:
        img = image.load_img(path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        images.append(img_array)
    return np.array(images)



X_train_real = load_images(train_real)
X_train_fake = load_images(train_fake)
X_test_real = load_images(test_real)
X_test_fake = load_images(test_fake)

X_train = np.concatenate([X_train_real, X_train_fake])
X_test = np.concatenate([X_test_real, X_test_fake])

y_train = np.concatenate([np.ones(len(train_real)), np.zeros(len(train_fake))])
y_test = np.concatenate([np.ones(len(test_real)), np.zeros(len(test_fake))])

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

shuffle_idx = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]


inputs = Input(shape=(32, 32, 3), name="input")
x = Rescaling(1.0 / 255, name="rescale")(inputs)

x = Conv2D(32, (3, 3), activation="relu", name="conv2d_1")(x)
x = MaxPooling2D((2, 2), name="max_pool_1")(x)

x = Conv2D(32, (3, 3), activation="relu", name="conv2d_2")(x)
x = MaxPooling2D((2, 2), name="max_pool_2")(x)

x = Flatten(name="flatten")(x)
x = Dense(64, activation="relu", name="dense_64")(x)
outputs = Dense(1, activation="sigmoid", name="dense_output")(x)

model = Model(inputs, outputs, name="CIFAKE_CNN")

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)

model.summary()


early_stop = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
)

loss, acc, precision, recall = model.evaluate(X_test, y_test)
f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
print(f"\nTest Loss: {loss:.4f} | Test Accuracy: {acc:.4f} | "
      f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()


model.save("cifake_cnn.keras")
print("Model saved to cifake_cnn.keras")