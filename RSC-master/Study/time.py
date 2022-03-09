#!/usr/bin/env python
# encoding: utf-8
'''
@Author: Yuqi
@Contact: www2048g@126.com
@File: time.py
@Time: 2022/3/2 10:46
'''

import time
localtime = time.localtime(time.time())
time = time.strftime('%Y%m%d-%H.%M.%S', time.localtime(time.time()))
print(str(time))