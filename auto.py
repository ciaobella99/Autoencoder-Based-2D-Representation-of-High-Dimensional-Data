import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# 函數化設置隨機種子
def set_seed(seed_value=666):
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)

set_seed()

# 數據加載和預處理
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm, y

X_train_norm, y_train = load_and_preprocess_data("clsn1_trn.csv")

# 定義和編譯模型
def build_autoencoder(input_dim, encoding_dim=2, learning_rate=0.005):
    autoencoder = models.Sequential([
        layers.Dense(27, activation='relu', input_shape=(input_dim,)),
        layers.Dense(7, activation='relu'),
        layers.Dense(encoding_dim, activation='linear'),
        layers.Dense(7, activation='relu'),
        layers.Dense(27, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    return autoencoder

autoencoder = build_autoencoder(X_train_norm.shape[1])

# 設置 Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('report3.h5', monitor='val_loss', save_best_only=True)

# 訓練模型
autoencoder.fit(X_train_norm, X_train_norm, epochs=30, batch_size=27, 
                validation_split=0.2, verbose=1, 
                callbacks=[early_stopping, model_checkpoint])

# 使用訓練好的編碼器部分
encoder = models.Model(inputs=autoencoder.input, 
                       outputs=autoencoder.layers[2].output)
X_encoded = encoder.predict(X_train_norm)

# 繪製降維後的數據
def plot_encoded_data(X_encoded, y_train, class_colors=['red', 'yellow', 'green', 'cyan', 'blue','magenta','black']):
    reduced_data = pd.DataFrame({'dim1': X_encoded[:, 0], 'dim2': X_encoded[:, 1], 'class': y_train})
    plt.figure()
    for i, label in enumerate(np.unique(y_train)):
        mask = reduced_data['class'] == label
        plt.scatter(reduced_data.loc[mask, 'dim1'], reduced_data.loc[mask, 'dim2'], c=class_colors[i], label=label)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Reduced Dimensional Data')
    plt.legend()
    plt.savefig('reduced_data.png')
    plt.show()

plot_encoded_data(X_encoded, y_train)