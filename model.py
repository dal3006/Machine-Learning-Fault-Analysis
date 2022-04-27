import tensorflow as tf
from tensorflow.keras import layers


INPUT_LENGTH = 800


def create_model(input_length):
    """Create model"""
    x = input_tensor = tf.keras.Input(shape=(input_length, 1), name="raw_signal")

    # Encoder
    x = layers.Conv1D(filters=32, kernel_size=2, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(filters=32, kernel_size=2, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(filters=32, kernel_size=2, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    embeddings = layers.Flatten()(x)

    # x = layers.UpSampling1D(2)(x)
    # x = layers.Conv1DTranspose(filters=32, kernel_size=2, padding="same")(x)
    # x = layers.UpSampling1D(2)(x)
    # x = layers.Conv1DTranspose(filters=1, kernel_size=2, padding="same")(x)

    x = layers.Dense(10)(embeddings)
    x = layers.Dense(4, activation="softmax")(x)

    output = x
    return tf.keras.Model(
        inputs=[input_tensor],
        outputs=output,
    )


def compile_model(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def train_model(model, X_train, y_train, validation_data=None):
    """Compile and train. Returns (model, history)"""
    history = model.fit(X_train, y_train,
                        epochs=75,
                        batch_size=512,
                        validation_data=validation_data,
                        shuffle=True)
    return model, history
