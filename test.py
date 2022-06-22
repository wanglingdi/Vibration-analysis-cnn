import os

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.layers import Conv1D, Dense, Dropout, BatchNormalization, MaxPooling1D, Activation, Flatten
from keras.models import Sequential
from keras.utils import plot_model
from keras.regularizers import l2
import preprocess
from keras.callbacks import TensorBoard
import numpy as np
import keras
import pandas as pd


import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Input
# from tfkerassurgeon import delete_layer, insert_layer
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization

def change_model(model, new_input_shape=(None, 6000, 1)):
    # replace input shape of first layer
    model.layers[0]._batch_input_shape = new_input_shape
    # model.layers[1].batch_input_shape = new_input_shape
    # print("model.layers[1].batch_input_shape",model.layers[1].batch_input_shape)
    # # feel free to modify additional parameters of other layers, for example...
    # model.layers[2].pool_size = (8, 8)
    # model.layers[2].strides = (8, 8)

    # rebuild model architecture by exporting and importing via json
    new_model = keras.models.model_from_json(model.to_json())
    new_model.summary()

    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    # test new model on a random input image
    X = np.random.rand(10, 6000, 1)
    y_pred = new_model.predict(X)
    print(y_pred)

    return new_model

# 基于预训练模型Xception的特征提取层，创建新的模型
base_model = keras.models.load_model("mnist_cnn.h5")
# change_model



# new_input = Input(shape=(6000, 1), name='image_input')
# newInput = Input(batch_shape=(6000, 1))
for i,layer in enumerate(base_model.layers):
    print(i,layer.name)
for layer in base_model.layers:
  layer.trainable = False
# print(base_model.summary())


# tf.__version__
#


model2 = Sequential()
for layer in base_model.layers[:-1]:  # 跳过最后一层
   model2.add(layer)
print (model2.summary())
b1= base_model.layers[1].weights
print (b1)
b2 = model2.layers[1].weights
print (b2)


# dd(Conv1D(filters=1, kernel_size=64, name='ok1ok',strides=1, padding='same',kernel_regularizer=l2(1e-4)))

# model2.add(Activation('relu',name='213'))#激活函数全连接
# model2.add(MaxPooling1D(pool_size=2.9,input_shape=(6000,1),name='123123'))#定义最大池函数
# model2.add(Dense(2048,input_shape=(None, 6000, 1)))
print(model2.summary())
# # print(base_model.layers[0].get_weights())
# # model2.layers[0].set_weights(base_model.layers[0].get_weights())
# # model2.layers[1].set_weights(base_model.layers[1].get_weights())
# # model2.layers[2].set_weights(base_model.layers[2].get_weights())
# # for layer in base_model.layers[1:-1]:  # 跳过最后一层
# #    model2.add(layer)
# print (model2.summary())



# b1= base_model.layers[1].weights
# print (b1)
# b2 = model2.layers[1].weights
# print (b2)

# from tfkerassurgeon import delete_layer, insert_layer
# import keras


def loadDataCsv(path):
    f = open(path,"r")
    print("ok")
    res = pd.read_csv(f, delimiter=",",header=None, skiprows=0)
    data = res.to_numpy()
    datalabel = data.copy()
    # print(data)
    n = len(data[:])
    print(n)
    label = data[:, -10:]
    # data = np.delete(data,(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10), axis=1)
    data = data[:,2048:4096]
    print("data.shape",data.shape)
    print("label.shape",label.shape)
    return data,label,datalabel
# def loadDataCsv(path):
#     f = open(path,"r")
#     print("ok")
#     res = pd.read_csv(f,sep='\t', header=None, skiprows=0)
#     data = res.to_numpy()
#     # print(data)
#     n = len(data[:])
#     print(n)
#     label = data[:, -1]
#     data = np.delete(data ,-1, axis=1)
#     # print("data.shape",data.shape)
#     # print("label.shape",label.shape)
#     return data,label
path1 = r'data\SVM_DATA\datatrain.csv'
path2 = r'data\SVM_DATA\datatest.csv'
# my_matrix = np.loadtxt(open(path, "rb"), delimiter="\t", skiprows=0)
# #
# # print(my_matrix[:,6000])
data1,label1,datalabel1 = loadDataCsv(path1)
data2,label2,datalabel2 = loadDataCsv(path2)
model2.add(Dense(500,activation="relu",name = "dense_last",))
model2.add(Dense(10,activation="softmax",name = "dense_last1"))
model2.compile(loss="categorical_crossentropy",optimizer="sgd",metrics="acc")
print (model2.summary())
from sklearn.decomposition import PCA,TruncatedSVD


print(label1)
# print(data1)
print("data.shape",data1.shape)
print("label.shape",label1.shape)

# data_df = pd.DataFrame(datalabel1)  # data_df: 'DataFrame' object
# data_df.to_csv('./data/' +'train2.csv', index=None, header=None)


model2.fit(data1,label1,batch_size=128,epochs=1000)

eva = model2.evaluate(data2,label2,batch_size=1000)
print (eva)


# model2.save('test1_finsh.h5')
# model2.save()
#
# base_model = Model(include_top=False)
# for i,layer in enumerate(base_model.layers):
#     print(i,layer.name)

# model = Model(include_top=True, weights=None)
# for i,layer in enumerate(model.layers):
#     print(i,layer.name)
#
# # The default input image size for the Xception model is 299x299
# base_model = tf.keras.applications.xception.Xception(
#                 include_top=False,
#                 weights="imagenet",
#                 input_shape=(299,299,3)
# )
# # 冻结该模型
# base_model.trainable = False
#
# # 载入训练和测试数据集
# # 准备数据
# TRAIN_DATASET_PATH = r"D:\nn_tf\cats_vs_dogs\train" #训练数据集路径
# TEST_DATASET_PATH  = r"D:\nn_tf\cats_vs_dogs\test"  #测试数据集路径
# batch_size = 32
# image_size = (299,299)
# # 训练数据集
# train_dataset = keras.preprocessing.image_dataset_from_directory(
#     TRAIN_DATASET_PATH,
#     validation_split=0.2,
#     image_size=image_size,
#     seed=1337,
#     subset='training',
#     batch_size=batch_size
# )
# # 验证数据集
# val_dataset = keras.preprocessing.image_dataset_from_directory(
#     TRAIN_DATASET_PATH,
#     validation_split=0.2,
#     subset="validation",
#     seed=1337,
#     image_size=image_size,
#     batch_size=batch_size,
# )
# # 测试数据集
# test_dataset = keras.preprocessing.image_dataset_from_directory(
#     TEST_DATASET_PATH,
#     image_size=image_size,
#     shuffle=False,
#     batch_size=batch_size
# )
#
# # 查看数据形状
# for image, label in train_dataset.take(1):
#     print(image.shape)
#
#
# # 设计数据增强算法
# data_augmentation = tf.keras.Sequential([
#   layers.RandomFlip("horizontal_and_vertical"),
#   layers.RandomRotation(0.2),
# ])
#
# # 将数据增强作用到训练数据集上
# train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
# train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#
# # 查看数据增强后的数据形状
# for image, label in train_dataset.take(1):
#     print(image.shape)
#
#
# # 添加Xception model的预处理层
# xception_preprocess = keras.applications.xception.preprocess_input
# # 创建带有预处理的新模型
# inputs = keras.Input(shape=(299,299,3), name="image_input")
# x = xception_preprocess(inputs)
# x = base_model(x, training=False)
# x = keras.layers.GlobalAveragePooling2D()(x) #将特征值展平成向量
# outputs = keras.layers.Dense(1, name="precisions")(x)
# model_with_prepocess = keras.Model(inputs, outputs, name="model_with_prepocess")
#
# # 编译模型
# model_with_prepocess.compile(optimizer=keras.optimizers.Adam(),
#               loss=keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=[keras.metrics.BinaryAccuracy()])
# # 训练模型
# model_with_prepocess.fit(train_dataset, epochs=10, validation_data=val_dataset)
#
# # 在测试数据集上测试模型
# loss, acc = model_with_prepocess.evaluate(test_dataset)
# print(f"Loss is {loss}; Accuracy is {acc} in Test Dataset")