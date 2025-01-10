import numpy as np
import pandas as pd
import rasterio  # 导入rasterio
import os
import cv2
from sklearn.preprocessing import StandardScaler

# 设置路径
precipitation_folder = "/data/2008-2018_preciption/"  # 降水量图文件夹
NDVI_folder = "/data/NDVI_2008-2018/"  #NDVI图
o3_folder = "/data/o3_2008-2018/"       #臭氧图
crop_data_path = "/data/grain_data_2008_2018.csv"  # 粮食产量数据文件

#co2_folder = "/data/co2_2008-2018/"     #co2图
#temperature_folder = "/data/2008-2018_temperature/"  # 气温图文件夹

# 读取粮食产量数据
crop_data = pd.read_csv(crop_data_path)
years = crop_data['year'].values
yields = crop_data['yield'].values

# 读取并处理卫星遥感图像
def load_and_preprocess_images(precipitation_folder,NDVI_folder,o3_folder, years):# temperature_folder,NDVI_folder,co2_folder,o3_folder, temperature_folder  co2_folder
    images = []
    for year in years:
        # 使用rasterio打开各类卫星图像
        precip_image_path = os.path.join(precipitation_folder, f"SUM_{year}.tif")
        with rasterio.open(precip_image_path) as src:
            precip_image = src.read(1)  # 读取第一个波段

        NDVI_image_path = os.path.join(NDVI_folder, f"MEAN_{year}.tif")
        with rasterio.open(NDVI_image_path) as src:
            NDVI_image = src.read(1) 
        
        


        o3_image_path = os.path.join(o3_folder, f"{year}.tif")
        with rasterio.open(o3_image_path) as src:
            o3_image = src.read(1) 
            
        # temp_image_path = os.path.join(temperature_folder, f"MEAN_{year}.tif")
        # with rasterio.open(temp_image_path) as src:
        #     temp_image = src.read(1) 
        
        # co2_image_path = os.path.join(co2_folder, f"v8.0_FT2022_GHG_CO2_{year}_TOTALS_emi.tif")
        # with rasterio.open(co2_image_path) as src:
        #     co2_image = src.read(1)  

        # 调整图像大小为较低分辨率（如256x256）以节省内存 1024  1920,1080
        precip_image = cv2.resize(precip_image, (1080, 720), interpolation=cv2.INTER_LINEAR)
        NDVI_image = cv2.resize(NDVI_image,(1080, 720), interpolation=cv2.INTER_LINEAR)
        o3_image = cv2.resize(o3_image,(1080, 720), interpolation=cv2.INTER_LINEAR)
        
        # 归一化图像并确保数据类型为float32
        precip_image = precip_image.astype(np.float32)  # 转换为 float32
        NDVI_image = NDVI_image.astype(np.float32)
        o3_image = o3_image.astype(np.float32) 

        precip_image = precip_image / precip_image.max()  # 归一化到 0-1 范围
        NDVI_image = NDVI_image / NDVI_image.max()
        o3_image = o3_image / o3_image.max()
        
        # # 归一化图像
        precip_image = precip_image*10
        NDVI_image = NDVI_image*10
        o3_image = o3_image*10
        
        # 将两个图像堆叠成一个4D图像（4通道，降水量+NDVI+臭氧）
        combined_image = np.stack([precip_image,NDVI_image,o3_image], axis=-1)#, temp_image temp_image,co2_image,

        images.append(combined_image)

    return np.array(images)


# 使用rasterio读取图像并预处理
X = load_and_preprocess_images(precipitation_folder,NDVI_folder,o3_folder, years = range(2008, 2019))#temperature_folder, co2_folder,
# temperature_folder,co2_folder,

from sklearn.preprocessing import MinMaxScaler
#标准化产量数据
scaler = MinMaxScaler()
Y = scaler.fit_transform(yields.reshape(-1, 1))
# Y = yields.reshape(-1,1)

# 将X和Y的数据形状调整为适合CNN-LSTM模型的输入
X = X.reshape((X.shape[0], X.shape[1], X.shape[2], X.shape[3]))  # 样本数, 高, 宽, 通道数
# Y = Y.reshape(-1, 1)

import numpy as np

# 1. 识别 NaN 和 Inf
nan_mask = np.isnan(X)
inf_mask = np.isinf(X)

# 2. 计算均值，忽略 NaN 值
mean_value = np.nanmean(X)

# 3. 用均值填充 NaN 和 Inf
X[nan_mask] = mean_value
X[inf_mask] = mean_value

# 现在 X 中的 NaN 和 Inf 已被均值替换

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Reshape
import tensorflow as tf
import matplotlib.pyplot as plt


# 1. 定义两个模型函数
def build_model_CNN_LSTM(input_shape, optimizer):
    model = Sequential()

    # CNN部分
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # 展平
    model.add(Flatten())

    # LSTM部分
    model.add(Reshape((-1,128)))
    model.add(LSTM(50, activation='tanh', return_sequences=False))

    # 全连接层
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))

    # 输出层
    model.add(Dense(1))  # 预测粮食产量

    # 编译模型
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def build_model_CNN(input_shape, optimizer):
    model = Sequential()

    # CNN部分
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # 展平
    model.add(Flatten())

    # 全连接层
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))

    # 输出层
    model.add(Dense(1))  # 预测粮食产量

    # 编译模型
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# 定义输入形状：256x256图像，2通道（降水量+气温）
input_shape = (1080,720, 3)

# 划分训练集和验证集
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# 2. 使用CNN模型
adam_optimizer = Adam(learning_rate=0.00001)
model_adam = build_model_CNN(input_shape, adam_optimizer)

history_adam = model_adam.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=70, batch_size=8)

# # 保存模型
# model.save('cnn_model.h5')

# 3. 使用CNN_LSTM模型
sgd_optimizer = Adam(learning_rate=0.00001)
model_sgd = build_model_CNN_LSTM(input_shape, sgd_optimizer)

history_sgd = model_sgd.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=70, batch_size=8)

# # 保存模型
# model.save('cnn_lstm_model.h5')

# 4. 绘制训练损失曲线
plt.figure(figsize=(10, 6))

# Adam优化器的训练损失
plt.plot(history_adam.history['loss'], label='CNN - Train Loss', color='blue')

# SGD优化器的训练损失
plt.plot(history_sgd.history['loss'], label='CNN_LSTM - Train Loss', color='green')

# Adam优化器的验证损失
plt.plot(history_adam.history['val_loss'], label='CNN - Val Loss', color='red', linestyle='--')

# SGD优化器的验证损失
plt.plot(history_sgd.history['val_loss'], label='CNN_LSTM - Val Loss', color='orange', linestyle='--')

plt.title('Comparison of CNN and CNN_LSTM Models')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 4. 绘制训练损失曲线
plt.figure(figsize=(10, 6))

# Adam优化器的训练损失
plt.plot(history_adam.history['loss'], label='Adam - Train Loss', color='blue')

# SGD优化器的训练损失
plt.plot(history_sgd.history['loss'], label='SGD - Train Loss', color='green')

# Adam优化器的验证损失
plt.plot(history_adam.history['val_loss'], label='Adam - Val Loss', color='red', linestyle='--')

# SGD优化器的验证损失
plt.plot(history_sgd.history['val_loss'], label='SGD - Val Loss', color='orange', linestyle='--')

plt.title('Comparison of Adam and SGD Optimizers')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 绘制损失曲线
import matplotlib.pyplot as plt

# 获取训练过程中的损失数据
train_loss = history_adam.history['loss']
val_loss = history_adam.history['val_loss']

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Train Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='red')
plt.title('CNN - Train and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 绘制损失曲线
import matplotlib.pyplot as plt

# 获取训练过程中的损失数据
train_loss = history_sgd.history['loss']
val_loss = history_sgd.history['val_loss']

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Train Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='red')
plt.title('CNN_LSTM - Train and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# # 预测
# predictions = model.predict(X_val)

# # # 逆标准化预测值
# # predictions = scaler.inverse_transform(predictions)
# print(predictions)
# print(Y_val)

# # 评估模型
# from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(scaler.inverse_transform(Y_val), predictions)

# print(f'Mean Squared Error: {mse}')
