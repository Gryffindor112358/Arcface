### 主要执行的函数train_net(args)

##### 首先配置训练过程中的主要参数

1. CUDA的设备编号
2. batch_size，image_channel, per_batch_size等基本参数

##### 判断预训练模型

1. 如果没有预训练模型且网络的名字是spherenet, 则加载spherenet网络的相关参数
2. 如果有则加载预训练模型得相关参数

##### 判断损失函数

1. 如果是triplet，则加载triplet的相关参数
2. 如果不是，加载FaceImageIter（这是啥）

##### 又判断网络名字（这里跟上一个网络的名字的关系还搞不明白）

1. 如果名字是fresnet或者是fmobilefacenet，resnet的style是gaussian（？？？）
2. 如果不是，resnet的style是uniform（？？？）

##### 然后又设置了一些参数

1. optimizer的一系列，learning_rate, momentom, wd
2. mx.callback.Speedometer的一系列，batch_size, frequent

##### 在val.targets中用对name循环，对每个name

1. 将二进制的data赋值给data_set
2. 将data_set append到ver_list中，把name append到ver_name_list中



