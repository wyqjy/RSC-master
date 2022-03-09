'''
多个数据集联合读取
https://zhuanlan.zhihu.com/p/222772996

这个文件有写好的包，可以不用重写
JiGen里需要多写一个函数
'''

import bisect
import warnings

from torch.utils.data import Dataset

# This is a small variant of the ConcatDataset class, which also returns dataset index
#from data.JigsawLoader import JigsawTestDatasetMultiple


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod   #修饰器  静态方法
    def cumsum(sequence):  #计算多个源域的样本数目，并将数目做成一个累加值的列表r   其实相当于r[i]=r[i-1]+l
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    # def isMulti(self): #可能用不到  在JiGen里用到的，这里没有
    #     return isinstance(self.datasets[0], JigsawTestDatasetMultiple)  #判断是否是一个已知的类型

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()  #调用父类构造方法
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)    #传进来的是Dataset的类 转换成list
        self.cumulative_sizes = self.cumsum(self.datasets)  #累加的样本数目值list

    def __len__(self):
        return self.cumulative_sizes[-1]   #累加的list最后一个元素就是总共样本的数目

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)   #二分查找 找到要用哪个数据集
        if dataset_idx == 0:   #找一个数据集里面具体第几个样本
            sample_idx = idx   #第一个数据集 idx就是本身
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx    #返回样本，和第几个数据集的索引

    @property      #函数前加上@property，使得该函数可直接调用，封装起来
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes
