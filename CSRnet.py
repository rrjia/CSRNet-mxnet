import os
import random
import sys
import numpy as np
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import mxnet as mx
from mxnet import init
from mxnet.gluon import utils as gutils

data_train_gt = '/home/rrjia2/data/car_counting/gt/'
data_train_im = '/home/rrjia2/data/car_counting/masked_image/'
data_train_index = '/home/rrjia2/data/car_counting/train.txt'
data_test_gt = '/home/rrjia2/data/car_counting/gt/'
data_test_im = '/home/rrjia2/data/car_counting/masked_image/'
data_test_index = '/home/rrjia2/data/car_counting/test.txt'
learning_rate = 0.01

# data_train_gt = 'D:\data\TRANCOS_v3\images\\gt\\'
# data_train_im = 'D:\data\TRANCOS_v3\images\\masked_image\\'
# data_train_index = 'D:\data\TRANCOS_v3\images\\train.txt'


class MultiLayerDilation(nn.HybridBlock):
    def __init__(self, num_dilation, **kwargs):
        super(MultiLayerDilation, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(channels=512, kernel_size=3, dilation=num_dilation, padding=num_dilation)
        self.conv2 = nn.Conv2D(channels=512, kernel_size=3, dilation=num_dilation, padding=num_dilation)
        self.conv3 = nn.Conv2D(channels=512, kernel_size=3, dilation=num_dilation, padding=num_dilation)
        self.conv4 = nn.Conv2D(channels=256, kernel_size=3, dilation=num_dilation, padding=num_dilation)
        self.conv5 = nn.Conv2D(channels=128, kernel_size=3, dilation=num_dilation, padding=num_dilation)
        self.conv6 = nn.Conv2D(channels=64, kernel_size=3, dilation=num_dilation, padding=num_dilation)

        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
        self.bn3 = nn.BatchNorm()
        self.bn4 = nn.BatchNorm()
        self.bn5 = nn.BatchNorm()
        self.bn6 = nn.BatchNorm()

    def hybrid_forward(self, F, X):
        Y1 = F.relu(self.bn1(self.conv1(X)))
        Y2 = F.relu(self.bn2(self.conv2(Y1)))
        Y3 = F.relu(self.bn3(self.conv3(Y2)))
        Y4 = F.relu(self.bn4(self.conv4(Y3)))
        Y5 = F.relu(self.bn5(self.conv5(Y4)))
        Y6 = F.relu(self.bn6(self.conv6(Y5)))
        return Y6


class FocalLoss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label):
        loss = F.mean(F.sqrt(F.sum(F.power(F.subtract(output, label), 2), axis=[1, 2, 3])))
        # loss = F.sqrt(F.sum(F.power(F.subtract(output, label), 2)))
        return loss


class MultiScalaDilation(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(MultiScalaDilation, self).__init__(**kwargs)
        self.dilation1 = MultiLayerDilation(num_dilation=1)
        self.dilation2 = MultiLayerDilation(num_dilation=2)
        self.dilation3 = MultiLayerDilation(num_dilation=3)
        self.dilation4 = MultiLayerDilation(num_dilation=4)

    def hybrid_forward(self, F, X):
        Y1 = self.dilation1(X)
        Y2 = self.dilation2(X)
        Y3 = self.dilation3(X)
        Y4 = self.dilation4(X)
        return F.concat(Y1, Y2, Y3, Y4)


def data_iter(batch_size, dir_name, im_dir, gt_dir, ctx):
    dir_file = open(dir_name)
    lines = dir_file.readlines()
    indexs = list(range(len(lines)))
    random.shuffle(indexs)
    for i in range(0, len(indexs), batch_size):
        samples = np.array(indexs[i: min(i + batch_size, len(indexs))])

        if batch_size > 1:

            xs, ys = [], []
            for index in samples:
                # 获取路径
                file_name = lines[index]
                im_name, gt_name = file_name.split(' ')
                gt_name = gt_name.split('\n')[0]

                # 训练数据(图片)
                batch_xs = mx.image.imread(im_dir + im_name).astype('float32')
                batch_xs = batch_xs.transpose((2, 0, 1))

                # 训练数据 标签(密度图)
                batch_ys = nd.array(np.load(gt_dir + gt_name)).astype('float32')
                batch_ys = batch_ys.reshape([-1, batch_ys.shape[0], batch_ys.shape[1]])

                xs.append(batch_xs)
                ys.append(batch_ys)

            nd_xs = nd.stack(xs[0], xs[1])
            for j in range(2, len(xs)):
                nd_xs = nd.concat(nd_xs, nd.expand_dims(xs[j], 0), dim=0)

            nd_ys = nd.stack(ys[0], ys[1])
            for j in range(2, len(ys)):
                nd_ys = nd.concat(nd_ys, nd.expand_dims(ys[j], 0), dim=0)
            # print(nd_xs.shape, nd_ys.shape)
            yield nd_xs, nd_ys
        else:
            file_name = lines[samples[0]]
            im_name, gt_name = file_name.split(' ')
            gt_name = gt_name.split('\n')[0]

            # 训练数据(图片)
            batch_xs = mx.image.imread(im_dir + im_name).astype('float32')
            batch_xs = nd.expand_dims(batch_xs.transpose((2, 0, 1)), 0)

            # 训练数据 标签(密度图)
            batch_ys = nd.array(np.load(gt_dir + gt_name)).astype('float32')
            batch_ys = nd.expand_dims(batch_ys.reshape([-1, batch_ys.shape[0], batch_ys.shape[1]]), 0)
            yield batch_xs, batch_ys


def get_model():
    preTrain_net = gluon.model_zoo.vision.vgg16_bn(pretrained=True, root='./ckp')
    # preTrain_net = gluon.model_zoo.vision.vgg16_bn(pretrained=True, root='D:\model')
    net = nn.HybridSequential()
    with net .name_scope():
        for layer in preTrain_net.features[:33]:
            net.add(layer)

    net.add(
        MultiScalaDilation(),
        nn.Conv2D(kernel_size=(1, 1), channels=1, activation='sigmoid')
    )
    return net


def train_model():
    batch_size = 10
    num_epochs = 10
    ctx = [mx.gpu(i) for i in range(1)]

    net = get_model()
    net.initialize(ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = FocalLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': 0.001})

    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter(batch_size, data_train_index, data_train_im, data_train_gt, ctx):
            # print('x  shape is ', X.shape)
            # print('y  shape is ', y.shape)
            gpu_Xs = gutils.split_and_load(X, ctx)
            gpu_ys = gutils.split_and_load(y, ctx)
            with autograd.record():
                ls = [loss(net(gpu_X), gpu_y) for gpu_X, gpu_y in zip(gpu_Xs, gpu_ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            # print(ls[0].asscalar())
            print("epoch %d, loss: %f" % (epoch, ls[0].asscalar()))
        print("epoch %d, loss: %f" % (epoch, ls[0].asscalar()))
        net.save_params('ckp/CSRNet-%d.params' % epoch)


def eval_model():
    ctx = [mx.gpu(i) for i in range(1)]
    test_batch_size = 1

    net = get_model()
    net.initialize(ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    net.load_params('ckp/CSRNet-5.params', ctx=ctx)
    loss = FocalLoss()
    # for layer in net:
    #     print(layer)
    mae_arr = []
    for X, y in data_iter(test_batch_size, data_test_index, data_test_im, data_train_gt, ctx):
        X = X.copyto(mx.gpu(0))
        y = y.copyto(mx.gpu(0))
        # print('x  shape is ', X.shape)
        # print('y  shape is ', y.shape)
        predict = net(X)
        mae = nd.subtract(nd.sum(predict), nd.sum(y))
        mae_arr.append(abs(mae.asscalar()))
        ls = loss(predict, y)
        print('label car num is : {}, predict car num is : {}'.format(nd.sum(y).asscalar(), nd.sum(predict).asscalar()))
        print("predict loss: %f  mae: %f  " % (ls.asscalar(), abs(mae.asscalar())))
    print('average mae is : {}'.format(np.mean(np.array(mae_arr))))


def predict_dir_image_car_num(img_dir):
    ctx = [mx.gpu(i) for i in range(1)]

    net = get_model()
    net.initialize(ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    net.load_params('ckp/CSRNet-5.params', ctx=ctx)

    for _, _, file_names in os.walk(img_dir):
        for image_file in file_names:
            batch_xs = mx.image.imread(os.path.join(img_dir, image_file)).astype('float32')
            batch_xs = nd.expand_dims(batch_xs.transpose((2, 0, 1)), 0)
            batch_xs = batch_xs.copyto(mx.gpu(0))
            predict = net(batch_xs)
            print('image file {} predict car num is : {}'.format(image_file, nd.sum(predict).asscalar()))


if __name__ == '__main__':
    # train_model()
    eval_model()
    # predict_dir_image_car_num('/home/rrjia2/data/car_counting/traffic')