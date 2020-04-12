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


### 操作流程

1.把训练集face_emore放在目录datasets下
2.训练：在recognition目录下先copy一个sample_config.py命名为config.py,并编辑里面的参数。运行train.py,具体命令如CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r100 --loss arcface --dataset emore（有训练好的模型可以直接下载Model-Zoo），有一种损失函数Combined Margin，取得的整体效果会好一些。（训练好的模型不知道是不是自动保存在models目录下或是哪里。。）
3.验证模型：在src/eval目录下运行verification.py来验证模型。（测试集不需要我们准备，貌似可以连接到LFW等数据库来进行验证）（不知道怎么找到已经训练好的模型）
4.Feature Embedding（不确定是不是特征的提取，有可能是类似的作用，在Readme里貌似也找不到其他能够提取特征的东西了。。）：把训练好的模型放在models目录下，运行deploy目录下的test.py.（test.py里可以设置你想提取特征的图片，但是图片需要进行校正对齐过成112x112，校正对齐可用MTCNN）
