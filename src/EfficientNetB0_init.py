"""
EfficientNet-B0 transfer-learning model for OCTMNIST (64x64 grayscale, 4 classes).

The OCTMNIST images are single-channel uint8 arrays in [0, 255]. EfficientNet-B0 was
pre-trained on 3-channel ImageNet images and includes its own input rescaling and
normalization, so this builder:
  1. Optionally resizes the 64x64 images (larger inputs give the network more spatial
     detail, at a higher compute cost).
  2. Repeats the grey channel three times to match the RGB input the pre-trained
     weights expect. This is done inside the model, so the dataset stays as compact
     uint8 single-channel arrays in memory.
  3. Passes raw [0, 255] pixels to EfficientNet-B0 (do NOT divide by 255 beforehand).
  4. Adds a small classification head: GlobalAveragePooling -> Dropout -> Dense(softmax).
"""

import tensorflow as tf
from tensorflow.keras import layers


def build_efficientnet_b0(input_shape=(64, 64, 1), num_classes=4, image_size=64,
                          dropout_rate=0.2, weights="imagenet", verbose=True):
    """Return (model, base_model). The base starts frozen for the first training phase."""
    inputs = layers.Input(shape=input_shape, name="input")
    x = inputs
    if image_size != input_shape[0]:
        x = layers.Resizing(image_size, image_size, name="resize")(x)
    # Grayscale -> 3 channels (the pre-trained first convolution expects RGB)
    x = layers.Concatenate(axis=-1, name="gray_to_rgb")([x, x, x])

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=(image_size, image_size, 3),
    )
    base_model.trainable = False  # Phase 1: train only the new head

    # training=False keeps BatchNorm layers in inference mode, even after unfreezing,
    # so fine-tuning does not destroy the pre-trained batch statistics.
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNetB0_OCTMNIST")

    if verbose:
        print(f"EfficientNet-B0 built: input {input_shape} -> resized to {image_size}x{image_size}x3")
        print(f"Total parameters: {model.count_params():,}")
    return model, base_model


def unfreeze_top_layers(base_model, num_layers=40):
    """Phase 2: unfreeze the last `num_layers` layers of the base, keeping BatchNorm frozen."""
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    for layer in base_model.layers[-num_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False