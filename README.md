# 基于卫星遥感数据的粮食产量预测项目

## 项目简介

该项目使用卫星遥感数据（如降水量、NDVI、臭氧等）以及粮食产量数据，采用深度学习中的卷积神经网络（CNN）和卷积-长短期记忆网络（CNN-LSTM）模型进行粮食产量的预测。项目旨在利用多种环境数据进行粮食产量预测，为农业管理和政策制定提供数据支持。

## 环境要求

在运行此项目之前，请确保你已安装以下依赖：

- Python 3.x
- TensorFlow 2.x
- Keras
- scikit-learn
- numpy
- pandas
- rasterio
- opencv-python
- matplotlib

你可以通过以下命令安装所需的库：

```bash
pip install tensorflow keras scikit-learn numpy pandas rasterio opencv-python matplotlib

```

## 环境要求

本项目使用了以下几类数据：

- **降水量数据**：包含2008-2018年每年的降水量图像，格式为 `.tif`。
- **NDVI数据**：包含2008-2018年每年的NDVI图像，格式为 `.tif`。
- **臭氧数据**：包含2008-2018年每年的臭氧图像，格式为 `.tif`。
- **粮食产量数据**：包含2008-2018年每年对应的粮食产量数据，格式为 `.csv`。

您可以添加其他训练数据，只要您的硬件设备可以处理。如果您需要添加其他类型的训练数据，记得修改模型的输入数据，重点关注模型输入数据的维数，否则程序运行可能出错。

修改代码中的以下部分：
```python
# 定义输入形状：1080*720图像，3通道（降水量+NDVI+臭氧）
input_shape = (1080, 720, 3)

```

```python
# 读取降水量、NDVI和臭氧图像
img_rainfall = load_image('data/rainfall/2018.tif')
img_ndvi = load_image('data/ndvi/2018.tif')
img_ozone = load_image('data/ozone/2018.tif')

```

## 安装与使用
### 数据准备
确保将所有的 .tif 格式的卫星遥感图像和 .csv 格式的粮食产量数据放置在对应的文件夹中。您可以自行添加其他类型的训练数据，只要您的硬件设备可以处理。
我们强烈建议您使用预处理好的数据进行训练，因为原始数据文件较大，可能会占用较大的存储空间。您可以根据硬件条件对预处理过程进行灵活设置

### 数据预处理
代码会自动读取并处理卫星遥感图像和粮食产量数据。使用 load_and_preprocess_images 函数进行图像的预处理，包括图像归一化、大小调整等。

### 模型训练
#### 该项目提供了两个模型架构：

CNN模型：使用卷积神经网络进行预测。
CNN-LSTM模型：在CNN的基础上添加LSTM层，用于时间序列数据的建模。
通过 model.fit() 函数进行模型训练，模型的优化器使用Adam或SGD，可以根据需要调整。

#### 训练与验证
训练时，使用80%的数据作为训练集，20%的数据作为验证集。训练过程中会保存损失曲线，并通过Matplotlib进行可视化。

## 模型架构与训练
### CNN模型
该模型由多个卷积层（Conv2D）和池化层（MaxPooling2D）组成，用于提取图像特征。最后通过全连接层（Dense）输出预测结果。优化器使用Adam优化器，损失函数为均方误差（MSE）。

### CNN-LSTM模型
在CNN模型的基础上，增加了LSTM层，用于捕捉图像的时序特征。LSTM层后的输出通过全连接层进行最终预测。优化器同样使用Adam优化器。

## 结果展示与可视化
训练过程中，使用Matplotlib绘制了损失曲线，展示了不同模型和优化器的训练效果：

CNN模型的训练与验证损失曲线
CNN-LSTM模型的训练与验证损失曲线
不同优化器（Adam与SGD）的损失比较
这些图表可以帮我们了解不同模型和优化器对训练过程的影响。
