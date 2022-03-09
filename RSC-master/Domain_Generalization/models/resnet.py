from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import torch
from torch import nn as nn
from torch.autograd import Variable
import numpy.random as npr
import numpy as np
import torch.nn.functional as F
import random
import math

'''
resnett 认为深层的网络可以提取出更加丰富的语义信息。随着网络的加深一般会让分辨率降低而让通道数增加
resnet18 在第2,3,4,5个stage中，在每个stage中使用的基本模块数目是[2,2,2,2]
'''
class ResNet(nn.Module):
    def __init__(self, block, layers, jigsaw_classes=1000, classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])        #对应resnet中第2,3,4,5的stage
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.jigsaw_classifier = nn.Linear(512 * block.expansion, jigsaw_classes)
        self.class_classifier = nn.Linear(512 * block.expansion, classes)
        #self.domain_classifier = nn.Linear(512 * block.expansion, domains)
        self.pecent = 1/3    #冻结1/3的梯度表示

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        '''

        :param block:  基本模块选择谁（基本模块包括BasicBlock 和 Bottleneck）
        :param planes:  这是每个stage中，与每个block的输出通道相关的参数
        :param blocks:  2
        :param stride:
        :return:
        '''
        downsample = None      #定义了一个下采样模块
        '''
        只要stride>1 或者 输入和输出通道数目不一样，就可以断定残差模块产生的feature map相比于原来的分辨率降低了，此时需要进行下采样
        BasicBlock(或Bottleneck类）中的属性expansion，用于指定下一个BasicBlock的输入通道是多少
        
        '''
        if stride != 1 or self.inplanes != planes * block.expansion:   #前面没有实例化BasicBlock,所以不能使用实例属性，而是直接使用了类属性expansion
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))   #这才是生成了BasicBlock的实例  #self.inplanes等于上一个stage的输出通道数
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):   #resnet18中blocks=2
            layers.append(block(self.inplanes, planes))   #每一个block的输出通道数

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x, gt=None, flag=None, epoch=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        '''RSC的核心部分'''
        flag = False
        if flag:
            print('*'*20)  # 判断确实没进入rsc

            interval = 10
            if epoch % interval == 0:
                self.pecent = 3.0 / 10 + (epoch / interval) * 2.0 / 10  #每过10个epoch就把冻结的比例加上0.2   用在了样本上

            self.eval()
            x_new = x.clone().detach()  #clone()返回值是一个中间变量，支持梯度回溯。detach()操作后tensor与原始的tensor共享数据内存，它改变后原始的数据也相应的改变
            x_new = Variable(x_new.data, requires_grad=True)  #变量   可以反向传播  pytorch都是由tensor计算的，而tensor里面的参数是variable的形式
            x_new_view = self.avgpool(x_new)
            x_new_view = x_new_view.view(x_new_view.size(0), -1)   #view返回的tensor和原来的tensor还是共享data的，但还是一个新的tensor，两者的内存地址并不一致
            output = self.class_classifier(x_new_view)
            class_num = output.shape[1]   #预测的数量 7     output.shape=[128,7]  代表着128个样本 7个类别
            index = gt    #groundtruth  真实标签
            num_rois = x_new.shape[0]  #128  输出的尺寸 128*512*7*7      128=batchsize*2  传入的数据做了一次concat
            num_channel = x_new.shape[1]  #512
            H = x_new.shape[2]   #7
            HW = x_new.shape[2] * x_new.shape[3]  #49

            one_hot = torch.zeros((1), dtype=torch.float32).cuda() #
            one_hot = Variable(one_hot, requires_grad=False)
            sp_i = torch.ones([2, num_rois]).long()  #[2*128]
            sp_i[0, :] = torch.arange(num_rois)      #从0-127
            sp_i[1, :] = index                       #和索引对应的真实值 标签
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()  #稀疏张量
            one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)
            one_hot = torch.sum(output * one_hot_sparse)   #预测值和真实值得乘积
            self.zero_grad()  #梯度置0
            one_hot.backward()  #求导

            grads_val = x_new.grad.clone().detach()
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)  #按列求平均值  从通道的维度上
            channel_mean = grad_channel_mean
            grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)

            spatial_mean = torch.sum(x_new * grad_channel_mean, 1)
            spatial_mean = spatial_mean.view(num_rois, HW)  #128,49
            self.zero_grad()

            choose_one = random.randint(0, 9)   #选择空间和通道的概率  大概各1/2的概率
            if choose_one <= 4:
                # ---------------------------- spatial -----------------------
                spatial_drop_num = math.ceil(HW * 1 / 3.0)    #ceil函数返回数字的上整数   49/3向上取整 17

                ''' ********** 改进001 *********'''
                # put_fe = 3   #放开的前几名特征数量
                # upper_boundary = torch.sort(spatial_mean, dim=1, descending=True)[0][:, put_fe]  #001
                # upper_boundary = upper_boundary.view(num_rois, 1).expand(num_rois, 49)      #001

                th18_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num]  #+put_fe 按行排序  取第17大的值  [128]   +put_fe新改的001
                th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, 49)      #[128 49]

                mask_all_cuda = torch.where(spatial_mean > th18_mask_value, torch.zeros(spatial_mean.shape).cuda(),
                                            torch.ones(spatial_mean.shape).cuda())
                # mask_all_cuda = torch.where(spatial_mean > upper_boundary, torch.ones(spatial_mean.shape).cuda(),    #001  把排名最高的put_fe再放出来
                #                             mask_all_cuda)

                mask_all = mask_all_cuda.reshape(num_rois, H, H).view(num_rois, 1, H, H)
            else:
                # -------------------------- channel ----------------------------
                vector_thresh_percent = math.ceil(num_channel * 1 / 3.2)

                '''********** 改进001 *********'''
                # put_fe = 27
                # upper_boundary = torch.sort(channel_mean, dim=1, descending=True)[0][:, put_fe]    #001
                # upper_boundary = upper_boundary.view(num_rois, 1).expand(num_rois, num_channel)    #001

                vector_thresh_value = torch.sort(channel_mean, dim=1, descending=True)[0][:, vector_thresh_percent]  #改+put_fe
                vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, num_channel)
                vector = torch.where(channel_mean > vector_thresh_value, torch.zeros(channel_mean.shape).cuda(),
                                     torch.ones(channel_mean.shape).cuda())
                # vector = torch.where(channel_mean > upper_boundary, torch.ones(channel_mean.shape).cuda(),  #001
                #                      vector)

                mask_all = vector.view(num_rois, num_channel, 1, 1)

            # ----------------------------------- batch ----------------------------------------
            cls_prob_before = F.softmax(output, dim=1)
            x_new_view_after = x_new * mask_all
            x_new_view_after = self.avgpool(x_new_view_after)
            x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
            x_new_view_after = self.class_classifier(x_new_view_after)
            cls_prob_after = F.softmax(x_new_view_after, dim=1)

            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
            after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
            change_vector = before_vector - after_vector - 0.0001
            change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
            th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][int(round(float(num_rois) * self.pecent))]
            drop_index_fg = change_vector.gt(th_fg_value).long()
            ignore_index_fg = 1 - drop_index_fg
            not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0]
            mask_all[not_01_ignore_index_fg.long(), :] = 1

            self.train()
            mask_all = Variable(mask_all, requires_grad=True)
            x = x * mask_all    #公式2  特征表示z乘以掩码向量m

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.class_classifier(x)


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)  #并没有把BasicBlock类实例化
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
