
import tensorflow as tf
import tensorflow.keras.layers as tfl


def build_model(
    input_shape=(64, 64, 1),
    num_classes=4,
    conv_filters=(32, 64, 128, 256),
    spatial_dropout=0.10,
    dense_dropout=0.25,
    verbose=True
):
    """
    Standard CNN for OCTMNIST classification.

    Architecture:
        Input
        -> Rescaling
        -> 4 x (Conv2D -> ReLU -> MaxPooling2D)
        -> SpatialDropout2D
        -> GlobalAveragePooling2D
        -> Dense(32, ReLU)
        -> Dropout
        -> Dense(num_classes, Softmax)

    No Batch Normalization.

    Expected input pixel range: [0, 255].
    """

    if len(conv_filters) != 4:
        raise ValueError("Exactly 4 convolutional blocks required.")

    model_a = tf.keras.Sequential(name="Model_A")

    # Input and normalization
    model_a.add(tf.keras.Input(shape=input_shape))
    model_a.add(tfl.Rescaling(1.0 / 255.0))

    # Convolutional blocks
    for i, filters in enumerate(conv_filters, start=1):

        model_a.add(
            tfl.Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                activation="relu",
                name=f"Convolution{i}"
            )
        )

        model_a.add(
            tfl.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name=f"MaxPooling{i}"
            )
        )

    # Spatial regularization
    if spatial_dropout > 0:
        model_a.add(
            tfl.SpatialDropout2D(
                rate=spatial_dropout,
                name="SpatialDropout"
            )
        )

    # Classification head
    model_a.add(
        tfl.GlobalAveragePooling2D(
            name="GlobalAveragePooling"
        )
    )

    model_a.add(
        tfl.Dense(
            units=32,
            activation="relu",
            name="DenseHidden"
        )
    )

    if dense_dropout > 0:
        model_a.add(
            tfl.Dropout(
                rate=dense_dropout,
                name="DenseDropout"
            )
        )

    model_a.add(
        tfl.Dense(
            units=num_classes,
            activation="softmax",
            name="Output"
        )
    )

    if verbose:
        print("\nModel A: Standard CNN with Dropout")
        model_a.summary()

    return model_a
