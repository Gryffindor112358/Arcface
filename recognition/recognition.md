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

1. 下载训练集并放在datasets目录下，每个训练集包含6个文件，
face_emore/
    train.idx
    train.rec
    property
    lfw.bin
    cfp_fp.bin
    agedb_30.bin
前三个是训练集后三个是验证集
2. 训练：在recognition目录下
先执行：
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
先copy一个sample_config.py命名为config.py,并编辑里面的参数，比如训练集的路径之类的。
训练有三种（或者是三个阶段？）
（1）Train ArcFace with LResNet100E-IR
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r100 --loss arcface --dataset emore
（2）Train ConsianFace with LResNet50E-LR
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r50 --loss cosface --dataset emore
(3)Train Softmax with LMobileNet-GAP
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network m1 --loss softmax --dataset emore
(4)Fine-turn the above Softmax model with Triplet loss
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network m1 --loss triplet --lr 0.005 --pretrained ./models/m1-softmax-emore,1
（有训练好的预训练模型可以直接下载Model-Zoo），有一种损失函数Combined Margin，取得的整体效果会好一些。（训练好的模型不知道是不是自动保存在models目录下或是哪里。。）
3. 验证模型：在src/eval目录下运行verification.py来验证模型。（测试集不需要我们准备，貌似可以连接到LFW等数据库来进行验证）（不知道怎么找到已经训练好的模型）
命令：
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network r100 --loss combined --dataset emore

4. 在Evalution/megaface中测试模型的精确度，所有对齐的图像都已经提供了。
5. Feature Embedding
在deploy文件夹中。输入的照片首先经过centre cropped，然后用MTCNN的RNet+ONet进行进一步对齐。
步骤：（1）准备一个预训练模型，
      （2）把模型放在models目录下，
      （3）运行deploy/test.py
（不确定是不是特征的提取，有可能是类似的作用，在Readme里貌似也找不到其他能够提取特征的东西了。。）：把训练好的模型放在models目录下，运行deploy目录下的test.py.（test.py里可以设置你想提取特征的图片，但是图片需要进行校正对齐过成112x112，校正对齐可用MTCNN）
