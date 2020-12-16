import keras
from sklearn.preprocessing import LabelBinarizer
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Dropout, BatchNormalization, Activation


def load_dataset(flatten=False):
    """Загружаем датасет и формируем обучающаую и тестовую выборку"""
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    # Оставим 10000 примеров на валидацию
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])
    return X_train, y_train, X_val, y_val, X_test, y_test


def example(x_train, y_train, x_test, y_test):

    from keras.callbacks import TensorBoard

    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=1,
                               write_graph=True, write_images=True)

    input1 = Input(shape=(28 * 28,))
    hidden1 = Dense(512, activation='relu')(input1)
    hidden2 = Dense(256, activation='relu')(hidden1)
    output = Dense(10, activation='softmax')(hidden2)
    model = Model(inputs=input1, outputs=output)
    model.compile(optimizer='sgd', loss='binary_crossentropy',
                  metrics=['accuracy', 'mean_squared_error'])

    model._get_distribution_strategy = lambda: None
    model.run_eagerly = lambda: None

    model.fit(x_train, y_train, epochs=5,
                        validation_data=(x_test, y_test),
                        callbacks=[tensor_board])
    test_loss, test_acc, test_mse = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss}, test mse: {test_mse}, test accuracy: {test_acc}')
    print(model.summary())


def adagrad(x_train, y_train, x_test, y_test):

    input1 = Input(shape=(28 * 28,))
    hidden1 = Dense(512, activation='relu')(input1)
    hidden2 = Dense(256, activation='relu')(hidden1)
    output = Dense(10, activation='softmax')(hidden2)
    model = Model(inputs=input1, outputs=output)
    model.compile(optimizer='Adagrad', loss='binary_crossentropy',
                  metrics=['accuracy', 'mean_squared_error'])

    model._get_distribution_strategy = lambda: None
    model.run_eagerly = lambda: None

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    test_loss, test_acc, test_mse = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss}, test mse: {test_mse}, test accuracy: {test_acc}')
    print(model.summary())
    return model


def adam(x_train, y_train, x_test, y_test):

    input1 = Input(shape=(28 * 28,))
    hidden1 = Dense(512, activation='relu')(input1)
    hidden2 = Dense(256, activation='relu')(hidden1)
    output = Dense(10, activation='softmax')(hidden2)
    model = Model(inputs=input1, outputs=output)
    model.compile(optimizer='Adam', loss='binary_crossentropy',
                  metrics=['accuracy', 'mean_squared_error'])

    model._get_distribution_strategy = lambda: None
    model.run_eagerly = lambda: None

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    test_loss, test_acc, test_mse = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss}, test mse: {test_mse}, test accuracy: {test_acc}')
    print(model.summary())
    return model


def batchnorm_dropout(x_train, y_train, x_test, y_test):

    input1 = Input(shape=(28 * 28,))
    hidden1 = Dense(512, activation='relu')(input1)
    hidden2 = BatchNormalization()(hidden1)
    hidden3 = Activation('relu')(hidden2)
    hidden4 = Dropout(0.5)(hidden3)

    hidden5 = Dense(256)(hidden4)
    hidden6 = BatchNormalization()(hidden5)
    hidden7 = Activation('relu')(hidden6)

    output = Dense(10, activation='softmax')(hidden7)
    model = Model(inputs=input1, outputs=output)
    model.compile(optimizer='Adam', loss='binary_crossentropy',
                  metrics=['accuracy', 'mean_squared_error'])

    model._get_distribution_strategy = lambda: None
    model.run_eagerly = lambda: None

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    test_loss, test_acc, test_mse = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss}, test mse: {test_mse}, test accuracy: {test_acc}')
    print(model.summary())
    return model


def build_model(hp):
    input1 = Input(shape=(28 * 28,))
    hidden1 = Dense(512)(input1)
    hidden2 = BatchNormalization()(hidden1)
    hidden3 = Activation('relu')(hidden2)
    hidden4 = Dropout(0.5)(hidden3)

    hidden5 = Dense(hp.Int('units',
                            min_value=32,
                            max_value=256,
                            step=32))(hidden4)
    hidden6 = BatchNormalization()(hidden5)
    hidden7 = Activation('relu')(hidden6)

    output = Dense(10, activation='softmax')(hidden7)

    adam = keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))

    model = Model(inputs=input1, outputs=output)
    model.compile(optimizer=adam, loss='binary_crossentropy',
                  metrics=['accuracy', 'mean_squared_error'])
    return model


def keras_tuner(x_train, y_train, x_test, y_test):
    from kerastuner.tuners import RandomSearch
    tuner = RandomSearch(build_model, objective='val_accuracy',
                         max_trials=5, executions_per_trial=3,
                         directory='./test', project_name='helloworld')

    tuner.search_space_summary()

    tuner.search(x_train, y_train,
                 epochs=5,
                 validation_data=(x_test, y_test))

    print(tuner.results_summary())


if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Сделаем объекты плоскими N*28*28 to  N*784
    X_train = X_train.reshape(
        (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    X_test = X_test.reshape(
        (X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

    print('Train dimension: ', X_train.shape)
    print('Test dimension: ', X_test.shape)

    # Лейблы нужно сделать One-Hot
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)
    print('Train labels dimension: ', y_train.shape)
    print('Test labels dimension: ', y_test.shape)

    example(X_train, y_train, X_test, y_test)

    # adagrad(X_train, y_train, X_test, y_test)

    # adam(X_train, y_train, X_test, y_test)

    # batchnorm_dropout(X_train, y_train, X_test, y_test)

    # keras_tuner(X_train, y_train, X_test, y_test)





