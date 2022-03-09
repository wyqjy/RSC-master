#!/usr/bin/env python
# encoding: utf-8
'''
@Author: Yuqi
@Contact: www2048g@126.com
@File: where.py
@Time: 2022/3/1 10:26
'''

import torch

a = torch.arange(6)
c = torch.ones(6)
b = torch.where(a<c, -1, a)
print(a)
print(c)
print(b)