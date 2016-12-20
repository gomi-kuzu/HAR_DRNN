#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
クロスバリゼーション用
"""

import os
import fnmatch
import numpy as np
import random
from operator import add

def split_seq(seq, s_size):
    size = len(seq)/s_size
    return [seq[i:i+size] for i in range(0, len(seq), size)]

def mkf(test_num):
    f = os.listdir("./HascToolDataPrj/SampleData_non_sequence/")
    hako_x = [[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]]]

    train_list = []
    test_list =[]
    train_label = []
    test_label = []
    f.sort()

    for t,i in enumerate(f):

        ff = os.listdir("./HascToolDataPrj/SampleData_non_sequence/{}".format(i))
        ff.sort()

        for v in ff:

            fff = fnmatch.filter(os.listdir("./HascToolDataPrj/SampleData_non_sequence/{0}/{1}".format(i,v)),"*.csv" )
            fff.sort()
            ffff = split_seq(fff,5)
            for k,l in enumerate(ffff):
                for e in l:
                    data = np.loadtxt("./HascToolDataPrj/SampleData_non_sequence/{0}/{1}/{2}".format(i,v,e),delimiter=",",dtype=np.float32)
                    data = np.delete(data,np.s_[::4], 1)
                    hako_x[t][k].append(data)


    hako2 = [[],[],[],[],[],[]]

    for y,h in enumerate(hako_x):

        hako2[y] = h

    for i,v in enumerate(hako2):
        te_v = v[test_num]
        del v[test_num]

        for l in te_v:
            test_list.append(l)
            test_label.append(i)
        for p in v:
            for q in p:
                train_list.append(q)
                train_label.append(i)

    #シャッフル
    ran_tr = range(len(train_list))
    ran_te = range(len(test_list))

    random.shuffle(ran_tr)
    random.shuffle(ran_te)

    r_train_list = []
    r_test_list = []
    r_train_label = []
    r_test_label = []

    for t in ran_tr:
        r_train_list.append(train_list[t])
        r_train_label.append(train_label[t])

    for g in ran_te:
        r_test_list.append(test_list[g])
        r_test_label.append(test_label[g])

    return r_train_list, r_test_list, r_train_label, r_test_label

if __name__=="__main__":
    a, b, c, d = mkf(4)
    print(a,a[0][0])