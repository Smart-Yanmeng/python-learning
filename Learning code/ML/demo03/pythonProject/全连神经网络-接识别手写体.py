from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt


'''
通过在输入和输出之间堆叠多个全连接层的网络称为多层感知机，
有时会被通俗的称之为`香草`神经网络(即原始神经网络)
我们将训练模型预测 `MNIST` 数据集中的数字标签，
`MNIST` 数据集是十分常用的数据集，
数据集由来自 `250` 个不同人手写的数字构成，
其中训练集包含 `60000` 张图片，
测试集包含 `10000` 张图片，
每个图片都有其标签，
图片大小为 `28*28`。

关键步骤概括
+ 展平输入数据集，使用 `reshape` 方法将每个像素视为一个输入层的节点变量
+ 对标签值进行独热编码，使用 `np_utils` 中的 `to_categorical` 方法将标签转换为独热向量
+ 使用 `Sequential` 堆叠网络层来构建具有隐藏层的神经网络
+ 使用 `model.compile` 方法对神经网络进行了编译，以最大程度地减少多分类交叉熵损失+ 使用 `model.fit` 方法根据训练数据集拟合模型+ 提取了存储在 `history` 中的所有 `epoch` 的训练和测试的损失和准确率+ 使用 `model.predict` 方法输出测试数据集中图片对应每个类别的概率+ 遍历了测试数据集中的所有图像，根据概率值最高索引确定图片类别
+ 最后，计算了准确率(预测类别与图像的实际类别相匹配的个数)

'''

# 导入相关的包和数据集，并可视化数据集以了解数据情况
# 导入相关的 `Keras` 方法和 `MNIST` 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
plt.subplot(221)
plt.imshow(x_train[0], cmap='gray')
plt.subplot(222)
plt.imshow(x_train[1], cmap='gray')
plt.subplot(223)
plt.imshow(x_test[0], cmap='gray')
plt.subplot(224)
plt.imshow(x_test[1], cmap='gray')
plt.show()

'''
展平 `28 x 28` 图像，以便将输入变换为一维的 784 个像素值，
并将其馈送至 `Dense` 层中。
此外，需要将标签变换为独热编码。
此步骤是数据集准备过程中的关键：
`x_train` 数组具有 `x_train.shape[0]` 个数据点(图像)，
每个图像中都有 `x_train.shape[1]` 行和 `x_train.shape[2]` 列， 
我们将其形状变换为具有 `x_train.shape[0]` 个数据，
每个数据具有 `x_train.shape [1] * x_train.shape[2]` 个值的数组**
'''
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(-1, num_pixels).astype('float32')
x_test = x_test.reshape(-1, num_pixels).astype('float32')

'''
将标签数据编码为独热向量：
'''
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

'''
用具有 1000 个节点的隐藏层构建神经网络：
输入具有 `28×28=784` 个值，
这些值与隐藏层中的 `1000` 个节点单元相连，
指定激活函数为 `ReLU`。
最后，隐藏层连接到具有 `num_classes=10` 个值的输出 (有十个可能的图像标签)，
因此 `to_categorical` 方法创建的独热向量有 `10` 列)，
在输出的之前使用 `softmax` 激活函数，以便获得图像的类别概率。
'''
model = Sequential()
model.add(Dense(1000, input_dim=num_pixels, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# 上述模型架构信息可视化如下所示：
model.summary()
'''
输出概要如下：
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 1000)              785000    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                10010     
=================================================================
Total params: 795,010
Trainable params: 795,010
Non-trainable params: 0
_________________________________________________________________

在上述体系结构中，**第一层的参数数量为 `785000`，因为 `784` 个输入单元连接到 `1000` 个隐藏层单元，
因此在隐藏层中包括 `784 * 1000` 权重值加 `1000` 个偏置值，总共 `785000` 个参数**。
类似地，输出层有10个输出，分别连接到 `1000` 个隐藏层，
从而产生 `1000 * 10` 个权重和 `10` 个偏置(总共 `10010` 个参数)。
输出层有 `10` 个节点单位，因为输出中有 `10` 个可能的标签，
输出层为我们提供了**给定输入图像的属于每个类别的概率值**，
例如第一节点单元表示图像属于 0 的概率，第二个单元表示图像属于 1 的概率，以此类推。
'''

'''编译模型
因为目标值是包含多个类别的独热编码矢量，所以损失函数是多分类交叉熵损失。
此外，我们使用 `Adam` 优化器来最小化损失函数，
在训练模型时，监测准确率 (`accuracy`，可以简写为 `acc`) 指标。
'''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

'''
拟合模型
上述代码中，我们指定了模型要拟合的输入 (`x_train`) 和输出 (`y_train`)；
指定测试数据集的输入和输出，模型将不会使用测试数据集来训练权重，
但是，它可以用于观察训练数据集和测试数据集之间的损失值和准确率有何不同。
设定批大小 (`batch size`) 64 每个梯度要更新的样本数，以更新权重
设定回合 (`epoch`) 数 50
'''
history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=10,
                    batch_size=64,
                    verbose=1)

'''
提取不同epoch的训练和测试损失以及准确率指标：
'''

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']  # validate loss datset 验证集（测试集上的损失函数）
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
epochs = range(1, len(val_loss_values) + 1)

plt.subplot(211)
plt.plot(epochs, loss_values, marker='x', label='Traing loss')  # x 训练集损失函数
plt.plot(epochs, val_loss_values, marker='o', label='Test loss')  # y 测试集损失函数
plt.title('Training and test loss')  # 标题
plt.xlabel('Epochs')  # x标签
plt.ylabel('Loss')  # y标签
plt.legend()  # 图例

plt.subplot(212)
plt.plot(epochs, acc_values, marker='x', label='Training accuracy')  # x 训练集准确率
plt.plot(epochs, val_acc_values, marker='o', label='Test accuracy')  # y测试集准确率
plt.title('Training and test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

'''
此外，我们也可以手动计算最终模型在测试集上的准确率：
使用模型的 `predict` 方法计算给定输入(此处为 `x_test` )的预测输出值。
然后，我们循环所有测试集的预测结果，使用 `argmax` 计算具有最高概率值的索引。
同时，对测试数据集的真实标签值执行相同的操作。在测试数据集的预测值和真实值中，
最高概率值的索引相同表示预测正确，
在测试数据集中正确预测的数量除以测试数据集的数据总量即为模型的准确率。

'''
preds = model.predict(x_test)
correct = 0
for i in range(len(x_test)):
    pred = np.argmax(preds[i], axis=0)
    act = np.argmax(y_test[i], axis=0)
    if (pred == act):
        correct += 1
    else:
        continue
accuracy = correct / len(x_test)
print('Test accuracy: {:.4f}%'.format(accuracy*100))
