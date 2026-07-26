import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model


model = load_model("cifake_cnn.keras")
print("Model loaded from cifake_cnn.keras")
model.summary()


test_real = glob.glob("data/test/REAL/*.jpg")[:10]
test_fake = glob.glob("data/test/FAKE/*.jpg")[:10]


def load_images(image_paths):
    images = []
    for path in image_paths:
        img = image.load_img(path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        images.append(img_array)
    return np.array(images)


X_real = load_images(test_real)
X_fake = load_images(test_fake)

X_demo = np.concatenate([X_real, X_fake])
y_demo = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_fake))])

print(f"Loaded {len(X_demo)} demo images")


preds = model.predict(X_demo, verbose=0)
for i in range(len(X_demo)):
    true_label = "REAL" if y_demo[i] == 1 else "FAKE"
    pred_label = "REAL" if preds[i][0] >= 0.5 else "FAKE"
    print(f"Image {i}: true={true_label}, predicted={pred_label} ({preds[i][0]:.3f})")


def get_gradcam_heatmap(model, img_array, last_conv_layer_name="conv2d_2"):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.outputs[0]],
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)
    return heatmap.numpy()


def display_gradcam(original_img, heatmap, alpha=0.4):
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (original_img.shape[0], original_img.shape[1])
    ).numpy().squeeze()

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8] * 255

    overlay = jet_heatmap * alpha + original_img
    overlay = np.clip(overlay, 0, 255).astype("uint8")
    return overlay



def show_gradcam_grid(model, X, y, n_real=4, n_fake=4):
    real_idx = np.where(y == 1)[0][:n_real]
    fake_idx = np.where(y == 0)[0][:n_fake]
    indices = list(real_idx) + list(fake_idx)

    fig, axes = plt.subplots(2, len(indices), figsize=(2 * len(indices), 4))

    for col, idx in enumerate(indices):
        img = X[idx]
        img_batch = np.expand_dims(img, axis=0)

        heatmap = get_gradcam_heatmap(model, img_batch)
        overlay = display_gradcam(img, heatmap)

        pred = model.predict(img_batch, verbose=0)[0][0]
        label = "REAL" if y[idx] == 1 else "FAKE"
        pred_label = "REAL" if pred >= 0.5 else "FAKE"

        axes[0, col].imshow(img.astype("uint8"))
        axes[0, col].set_title(f"{label}\npred={pred_label} ({pred:.2f})", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay)
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_demo.png")
    plt.show()


show_gradcam_grid(model, X_demo, y_demo)
