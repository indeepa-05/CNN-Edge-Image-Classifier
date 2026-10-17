import tensorflow as tf
import tensorflow.keras.layers as tfl

def build_model(input_shape=(64, 64, 1),num_classes=4,conv_filters=(32, 64, 128, 256),spatial_dropout=0.10,dense_dropout=0.25,verbose=True):
    """
    Lightweight CNN for OCTMNIST classification.

    Architecture:
        Input
        -> Rescaling
        -> 4 x (SeparableConv2D -> ReLU -> MaxPooling2D)
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
    model_b = tf.keras.Sequential(name="Model_B_Lightweight_CNN")
    # Input and normalization
    model_b.add(tf.keras.Input(shape=input_shape))
    model_b.add(tfl.Rescaling(1.0 / 255.0))
    # Depthwise-separable convolutional blocks
    for i, filters in enumerate(conv_filters, start=1):
        model_b.add(tfl.SeparableConv2D(filters=filters,kernel_size=(3, 3),strides=(1, 1),padding="same",depth_multiplier=1,activation="relu",name=f"SeparableConv{i}"))
        model_b.add(tfl.MaxPooling2D(pool_size=(2, 2),strides=(2, 2),name=f"MaxPooling{i}"))
    # Spatial regularization
    if spatial_dropout > 0:
        model_b.add(tfl.SpatialDropout2D(rate=spatial_dropout,name="SpatialDropout"))
    # Classification head
    model_b.add(tfl.GlobalAveragePooling2D(name="GlobalAveragePooling"))
    model_b.add(tfl.Dense(units=32,activation="relu",name="DenseHidden"))
    if dense_dropout > 0:
        model_b.add(tfl.Dropout(rate=dense_dropout,name="DenseDropout"))
    model_b.add(tfl.Dense(units=num_classes,activation="softmax",name="Output"))
    if verbose:
        print("\nModel B: Lightweight CNN with Dropout")
        model_b.summary()
    return model_b
