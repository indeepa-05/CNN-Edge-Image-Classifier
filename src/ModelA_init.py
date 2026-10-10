import tensorflow as tf
import tensorflow.keras.layers as tfl

def build_model(input_shape=(64, 64, 1),num_classes=4,conv_filters=(32, 64, 128, 256),verbose=False):
    """
    Builds the standard CNN (Model A) for OCTMNIST classification.

    Architecture:
        Input -> Rescaling
        -> 4 Conv2D + MaxPooling blocks
        -> GlobalAveragePooling2D
        -> Dense(32)
        -> Dense(num_classes)

    Args:
        input_shape: Shape of the input images.
        num_classes: Number of output classes.
        conv_filters: Number of convolution filters in each block.
        verbose: Whether to print model construction details.

    Returns:
        tf.keras.Model: Uncompiled Model A.
    """
    if len(conv_filters) != 4:
        raise ValueError("Model A requires exactly 4 convolutional blocks.")
    if any(not isinstance(f, int) or f <= 0 for f in conv_filters):
        raise ValueError("Filter counts must be positive integers.")
    if verbose:
        print("\nBuilding Model A: Standard CNN")
        print(f"Input Shape: {input_shape}")
        print(f"Number of Classes: {num_classes}")
        print(f"Convolution Filters: {conv_filters}")
        print("-" * 50)

    model_a = tf.keras.Sequential(name="Model_A")
    # Input and normalization
    model_a.add(tf.keras.Input(shape=input_shape))
    model_a.add(tfl.Rescaling(1. / 255))
    # Convolutional blocks
    for i, filters in enumerate(conv_filters, start=1):
        model_a.add(tfl.Conv2D(filters=filters,kernel_size=(3, 3),strides=(1, 1),padding="same",activation="relu",name=f"Convolution{i}"))
        model_a.add(tfl.MaxPooling2D(pool_size=(2, 2),strides=(2, 2),name=f"MaxPooling{i}"))
        if verbose:
            print(
                f"Block {i}: {filters} filters | "
                f"Output Shape: {model_a.output_shape}")
    # Global Average Pooling
    model_a.add(tfl.GlobalAveragePooling2D(name="GlobalAveragePooling"))
    # Hidden Dense Layer
    model_a.add(tfl.Dense(units=32,activation="relu",name="DenseHidden"))
    # Output Layer
    model_a.add(tfl.Dense(units=num_classes,activation="softmax",name="Output"))
    if verbose:
        print("-" * 50)
        print(f"Final Output Shape: {model_a.output_shape}")
        print(f"Total Parameters: {model_a.count_params():,}")
        print("Model A built successfully!\n")
    return model_a