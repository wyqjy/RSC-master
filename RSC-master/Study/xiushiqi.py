from os.path import join, dirname
import torch

class Test:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def fun(a,b,c):   #静态方法， 没有self参数
        print(a,b,c)

    def fun1(self):
        print(self.y)

def testxiushiqi():
    A = Test(4,5)
    Test.fun(1, 2, 3)   #类名.函数名
    A.fun(7, 8, 9)      #对象名.函数名

print(dirname(__file__))

torch.cuda.is_available()