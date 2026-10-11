import tensorflow as tf
import tensorflow.keras.layers as tfl


def build_model(input_shape=(64, 64, 1),num_classes=4,conv_filters=(32, 64, 128, 256),verbose=True):
    """
    Builds the lightweight CNN (Model B) for OCTMNIST classification.

    Architecture:
        Input -> Rescaling
        -> 4 SeparableConv2D + MaxPooling blocks
        -> GlobalAveragePooling2D
        -> Dense(32)
        -> Dense(num_classes)

    Returns:
        tf.keras.Model: Uncompiled Model B.
    """
    model_b = tf.keras.Sequential(name="Model_B_Lightweight_CNN")
    model_b.add(tf.keras.Input(shape=input_shape))
    model_b.add(tfl.Rescaling(1. / 255))

    if verbose:
        print("Building Model B: Lightweight CNN")
        print(f"Input Shape: {input_shape}")
        print(f"Number of Classes: {num_classes}")
        print(f"Convolution Filters: {conv_filters}")
        print("-" * 50)

    # Depthwise-separable convolution blocks
    for i, filters in enumerate(conv_filters, start=1):
        model_b.add(tfl.SeparableConv2D(filters=filters,kernel_size=(3, 3),strides=(1, 1),padding="same",activation="relu",name=f"SeparableConv{i}"))
        model_b.add(tfl.MaxPooling2D(pool_size=(2, 2),strides=(2, 2),name=f"MaxPooling{i}"))
        if verbose:
            print(
                f"Block {i}: {filters} filters | "
                f"Output Shape: {model_b.output_shape}")
    # Global Average Pooling
    model_b.add(tfl.GlobalAveragePooling2D(name="GlobalAveragePooling"))
    if verbose:
        print(f"Global Average Pooling: {model_b.output_shape}")
    # Hidden Dense Layer
    model_b.add(tfl.Dense(units=32,activation="relu",name="DenseHidden"))
    if verbose:
        print(f"Dense Hidden: {model_b.output_shape}")
    # Output Layer
    model_b.add(tfl.Dense(units=num_classes,activation="softmax",name="Output"))
    if verbose:
        print(f"Output Layer: {model_b.output_shape}")
        print("-" * 50)
        print(f"Total Parameters: {model_b.count_params():,}")
        print("Model B built successfully!")
    return model_b