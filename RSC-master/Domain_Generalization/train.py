import argparse

import torch
#from IPython.core.debugger import set_trace
from torch import nn
#from torch.nn import functional as F
from data import data_helper
## from IPython.core.debugger import set_trace
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
import numpy as np
from models.resnet import resnet18, resnet50
import datetime
import time as time1
import os

# 新加一个 tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./path/to/log')



def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")  #受内存限制 改为32
    parser.add_argument("--image_size", type=int, default=222, help="Image size")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")  #默认20
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="resnet18")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool, help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")

    return parser.parse_args()

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        if args.network == 'resnet18':
            model = resnet18(pretrained=True, classes=args.n_classes)
        elif args.network == 'resnet50':
            model = resnet50(pretrained=True, classes=args.n_classes)
        else:
            model = resnet18(pretrained=True, classes=args.n_classes)
        self.model = model.to(device)
        # print(self.model)
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
        self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (
        len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all,
                                                                 nesterov=args.nesterov)
        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None

    def _do_epoch(self, epoch=None):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        print('-'*60)

        for it, ((data, jig_l, class_l), d_idx) in enumerate(self.source_loader):
            data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            self.optimizer.zero_grad()

            data_flip = torch.flip(data, (3,)).detach().clone()  #按照维度对输入进行翻转
            data = torch.cat((data, data_flip))
            class_l = torch.cat((class_l, class_l))

            class_logit = self.model(data, class_l, True, epoch)  #进行前向传播  forward   第三个参数True代表要进行RSC操作
            class_loss = criterion(class_logit, class_l)
            _, cls_pred = class_logit.max(dim=1)
            loss = class_loss

            loss.backward()
            self.optimizer.step()

            self.logger.log(it, len(self.source_loader),
                            {"class": class_loss.item()},
                            {"class": torch.sum(cls_pred == class_l.data).item(), }, data.shape[0])
            # writer.add_scalar('train/loss', class_loss.item(), it)       #损失值的图像

            del loss, class_loss, class_logit

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():  #test_loaders里面是验证集和测试集（target）的数据
                total = len(loader.dataset)
                class_correct = self.do_test(loader)
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"class": class_acc})
                self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):  #返回有几个预测的准确   预测准确的个数
        class_correct = 0
        for it, ((data, nouse, class_l), _) in enumerate(loader):
            data, nouse, class_l = data.to(self.device), nouse.to(self.device), class_l.to(self.device)

            class_logit = self.model(data, class_l, False)
            _, cls_pred = class_logit.max(dim=1)

            class_correct += torch.sum(cls_pred == class_l.data)

        return class_correct


    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()

            self.logger.new_epoch(self.scheduler.get_lr())
            for n, v in enumerate(self.scheduler.get_lr()):  #其实里面一直只有一个值
                writer.add_scalar('Learning rate', v, self.current_epoch)  #画学习率的图
            self._do_epoch(self.current_epoch)
        writer.close() #新加的

        val_res = self.results["val"]

        test_res = self.results["test"]
        idx_best = val_res.argmax()
        idx_best_test = test_res.argmax()
        print("Best val %g, corresponding test(验证集最好时的测试结果) %g 验证集最好的best epoch: %g- best test: %g, 测试集最好的best epoch: %g" % (val_res.max(), test_res[idx_best], idx_best+1, test_res.max(), idx_best_test+1))
        print('验证准确率\n', val_res, '\n\n')
        print('测试准确率\n', test_res)

        localtime = time1.localtime(time1.time())
        time = time1.strftime('%Y%m%d-%H.%M.%S', time1.localtime(time1.time()))
        da = str(datetime.datetime.today())
        filename = 'TXT_no_RSC\\naive\\' + str(self.args.target) + '_'+ str(time) + '.txt'
        print(filename)
        file = open(filename, mode='w')
        file.write('best test' + str(test_res.max())+'   '+' local in'+str(idx_best_test+1)+'epoch'+'\n')
        file.write('Best val' + str(val_res.max)+'  '+'  local in'+str(idx_best+1)+'  corresponding test acc'+str(test_res[idx_best])+'\n\n')
        file.write('val acc\n'+str(val_res))
        file.write('\ntest acc\n'+str(test_res))


        self.logger.save_best(test_res[idx_best], test_res.max())  #存进来的是固定值
        return self.logger, self.model



def main():
    args = get_args()
    # args.source = ['art_painting', 'cartoon', 'sketch']
    # args.target = 'photo'
    #args.source = ['art_painting', 'cartoon', 'photo']
    #args.target = 'sketch'
    args.source = ['art_painting', 'photo', 'sketch']
    args.target = 'cartoon'
    # args.source = ['photo', 'cartoon', 'sketch']
    # args.target = 'art_painting'
    # --------------------------------------------
    print("Target domain: {}".format(args.target))
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()


if __name__ == "__main__":

    for i in range(1):
        torch.backends.cudnn.benchmark = True  #设置为True，会使得cuDNN来衡量自己库里面的多个卷积算法的速度，然后选择其中最快的那个卷积算法
        main()
