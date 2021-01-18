#!/usr/bin/python
# -*- coding: UTF-8 -*-
import multiprocessing
import pickle
from multiprocessing import Pool

import cv2


class Test:
    def callme(self):
        # print("ca cv2UMat",cv2UMat)
        # print(cv2UMat.get())
        return "11 "

    def callback(self, result):
        print(result)

    def start(self):
        p = Pool()
        m = multiprocessing.Manager()
        # quque = m.Queue()
        image = cv2.UMat(cv2.imread('/Users/denghaizhu/Downloads/name2.png'))
        info=pickle.dumps(image.get())
        print(info)
        # quque.put(self.image)
        r = p.apply_async(self.callme)
        # p.apply_async(self.callme, args=("t2",), callback=self.callback)
        # p.apply_async(self.callme, args=("t3",), callback=self.callback)
        # p.apply_async(self.callme, args=("t4",), callback=self.callback)
        print(r.get())
        p.close()
        p.join()


if __name__ == '__main__':
    t = Test()
    t.start()
