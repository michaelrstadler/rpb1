import tensorflow as tf

def train_model(x, y):
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0002,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name='Adam'
    )

    def get_model(n_inputs, n_outputs):
        reg=1e-7
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(500, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l2(reg)))
        model.add(tf.keras.layers.Dense(500, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l2(reg)))
        model.add(tf.keras.layers.Dense(500, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l2(reg)))
        model.add(tf.keras.layers.Dense(500, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l2(reg)))
        model.add(tf.keras.layers.Dense(n_outputs))
        model.compile(loss='mse', optimizer=optimizer)
        return model

    model = get_model(2600, 1)
    history = model.fit(x, y, validation_split=0.1,epochs=2, callbacks=[callback], verbose=False)
    return model, history
