import tensorflow as tf
from tensorflow.keras import layers
from mmd import MMDRegularizer

INPUT_LENGTH = 128


def create_model(input_length):
    """Create model"""
    input_src = tf.keras.Input(shape=(input_length, 1), name="input_src")
    input_trg = tf.keras.Input(shape=(input_length, 1), name="input_trg")

    def encoder(x, regularizer):
        x = layers.Conv1D(filters=32, kernel_size=2, padding="same", activation="relu")(x)
        x = layers.MaxPool1D(2)(x)
        x = layers.Conv1D(filters=32, kernel_size=2, padding="same", activation="relu")(x)
        x = layers.MaxPool1D(2)(x)
        x = layers.Conv1D(filters=64, kernel_size=2, padding="same", activation="relu")(x)
        x = layers.MaxPool1D(2)(x)
        x = layers.Conv1D(filters=64, kernel_size=2, padding="same",
                          activation="relu", activity_regularizer=regularizer)(x)
        x = layers.MaxPool1D(2)(x)
        return x

    mmd = MMDRegularizer()
    output_src = encoder(input_src, regularizer=mmd)
    output_trg = encoder(input_trg, regularizer=mmd)

    x = layers.Flatten()(output_src)

    # x = layers.UpSampling1D(2)(x)
    # x = layers.Conv1DTranspose(filters=32, kernel_size=2, padding="same")(x)
    # x = layers.UpSampling1D(2)(x)
    # x = layers.Conv1DTranspose(filters=1, kernel_size=2, padding="same")(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(4, activation="softmax")(x)
    output = x

    return tf.keras.Model(
        inputs=[input_src, input_trg],
        outputs=output,
    )


def compile_model(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def train_model(model, X_train, y_train, validation_data=None):
    """Compile and train. Returns (model, history)"""
    history = model.fit(X_train, y_train,
                        epochs=10,
                        batch_size=512,
                        validation_data=validation_data,
                        shuffle=True)
    return model, history
