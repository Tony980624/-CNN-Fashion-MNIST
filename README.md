# Fashion-MNIST-classification with CNN

![cover](https://github.com/Tony980624/Fashion-MNIST-classification/blob/main/file01/dataset-cover.png)

Fashion-MNIST 是 Zalando 文章图片的数据集，包含一个训练集和一个测试集。训练集有 60,000 个样本，测试集有 10,000 个样本。每个样本是一个 28x28 的灰度图像，并与 10 个类别中的一个标签相关联。

#  CNN 关键步骤简介

Convolutional Neural Networks use convolutional kernels that slide over the pixel matrix, performing dot products to generate feature maps.

![convo](https://github.com/Tony980624/Fashion-MNIST-classification/blob/main/file01/convolutional.gif)

有了特征图后，Pooling来缩小特征图（一般是取window内的最大值）

![pool](https://github.com/Tony980624/Fashion-MNIST-classification/blob/main/file01/Pooling.gif)

# 一个虚构例子讲述CNN训练的过程

随机初始化卷积核 K =:

$$
\left[
\begin{matrix}
    0.2 & -0.3 & 0.5 \\\\
    -0.1 & 0.4 & -0.2 \\\\
    0.3 & -0.5 & 0.1
\end{matrix} 
\right]
\tag{1}
$$

随机全连接层权重 W = 

$$
\begin{pmatrix}
0.2 & -0.4 & 0.1 & 0.5 
\end{pmatrix}
$$

假设我们输入的图像的像素灰度矩阵 I = :

$$
\left[
\begin{matrix}
    0.6 & 0.2 & 0.1 & 0. 5 \\\\
    0.9 & 0.3 & 0.7 & 0.4\\\\
    0.8 & 0.5 & 0.2 & 0.6\\\\
    0.1 & 0.4 & 0.9 & 0.7
\end{matrix} 
\right]
\tag{1}
$$

假设数据的真实标签是  : 1

通过对四个区域进行卷积操作后，得到feature map:

$$
\left[
\begin{matrix}
    -0.09 & 0.54   \\\\
    0.44 & -0.3 
\end{matrix} 
\right]
\tag{1}
$$

接下来激活ReLu函数，把位负数的值变为0， Flatted Map = :

$$
\left[
\begin{matrix}
    0 & 0.54   \\\\
    0.44 & 0 
\end{matrix} 
\right]
\tag{1}
$$

把特征map展平位一维向量(省去了pooling操作，因为维度不高，不需要降维):

$$
\begin{pmatrix}
0 & 0.54 & 0.44 & 0 
\end{pmatrix}
$$

全连接层计算 z = w*Flatted map = (0.2×0)+(−0.4×0.54)+(0.1×0.44)+(0.5×0)+0=−0.216+0.044 = −0.172:

输出层计算 y = :

$$y = \frac{1}{1+e^{-z}} = \frac{1}{1+e^{0.172}}$$ $\approx$ 0.4571

带入损失函数计算损失(真实标签为1) :

$$
\text{Loss} = [y \log(\hat{y}) + (1-y) \log(1-\hat{y})] \approx 0.78 
$$

接下来就是根据损失算出梯度来更新卷积核以及全连接层

# 训练模型




