import keras
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense


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

    model = Sequential()
    model.add(Dense(14, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss='mean_squared_error', optimizer='SGD',
                  metrics=['mean_squared_error', 'accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=10,
              validation_data=(x_test, y_test))

    test_loss, test_mse, test_acc = model.evaluate(x_test, y_test)
    print(
        f'Test loss: {test_loss}, test mse: {test_mse}, test accuracy: {test_acc}')


def try0(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(14, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mean_squared_error', 'accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=128)

    test_loss, test_mse, test_acc = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss}, test mse: {test_mse}, test accuracy: {test_acc}')


def try1(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mean_squared_error', 'accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=128)

    test_loss, test_mse, test_acc = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss}, test mse: {test_mse}, test accuracy: {test_acc}')


def try2(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mean_squared_error', 'accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=128)

    test_loss, test_mse, test_acc = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss}, test mse: {test_mse}, test accuracy: {test_acc}')


def try3(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mean_squared_error', 'accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=128)

    test_loss, test_mse, test_acc = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss}, test mse: {test_mse}, test accuracy: {test_acc}')


def try4(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mean_squared_error', 'accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=128)

    test_loss, test_mse, test_acc = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss}, test mse: {test_mse}, test accuracy: {test_acc}')


def try5(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(512, activation='softmax', input_shape=(28 * 28,)))
    model.add(Dense(256, activation='softmax'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mean_squared_error', 'accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=128)

    test_loss, test_mse, test_acc = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss}, test mse: {test_mse}, test accuracy: {test_acc}')


def try6(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='relu'))

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mean_squared_error', 'accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=128)

    test_loss, test_mse, test_acc = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss}, test mse: {test_mse}, test accuracy: {test_acc}')


def try_sgd(x_train, y_train, x_test, y_test, rate=0.01, momentum=0.0, nesterov=False):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    sgd = SGD(learning_rate=rate, momentum=momentum, nesterov=nesterov)

    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['mean_squared_error', 'accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=128)

    test_loss, test_mse, test_acc = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss}, test mse: {test_mse}, test accuracy: {test_acc}')


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

    """Пример из ноутбука"""
    # example(X_train, y_train, X_test, y_test)

    """Поменяли функцию потерь на кросс-энтропию"""
    try0(X_train, y_train, X_test, y_test)

    """Играемся со слоями"""
    # try1(X_train, y_train, X_test, y_test)

    """Играемся со слоями"""
    # try2(X_train, y_train, X_test, y_test)

    """Играемся со слоями"""
    # try3(X_train, y_train, X_test, y_test)  # Точность не сильно выросла

    """Меняем число нейронов"""
    # try4(X_train, y_train, X_test, y_test)

    """Меняем везде фенкции активации"""
    # try5(X_train, y_train, X_test, y_test)  # softmax

    """Меняем везде фенкции активации"""
    # try6(X_train, y_train, X_test, y_test)  # relu

    """Играемся с параметрами sgd"""
    # try_sgd(X_train, y_train, X_test, y_test, rate=0.5, momentum=0.0, nesterov=False)

