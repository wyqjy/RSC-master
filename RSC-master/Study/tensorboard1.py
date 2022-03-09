#!/usr/bin/env python
# encoding: utf-8
'''
@Author: Yuqi
@Contact: www2048g@126.com
@File: tensorboard1.py
@Time: 2022/2/28 18:45
'''

import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
for epoch in range(100):
    writer.add_scalar('scala/test', np.random.rand(), epoch)
    # writer.add_scalar('scalar/scalars_test',{'xsinx': epoch * np.sin(epoch), 'xcosx': epoch*np.cos(epoch)}, epoch)
    # 第二个writer会报错，提示NotImplementedError: Got <class 'dict'>, but expected numpy array or torch tensor.

writer.close()