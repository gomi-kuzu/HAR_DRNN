#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""

"""

from __future__ import print_function
import argparse
import sys
import time

import numpy as np

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import network
import matplotlib.pyplot as plt

import mkd
import random
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--n_epoch', '-e', default=100, type=int,
                    help='max_epoch')
parser.add_argument('--n_units', '-u', default=60, type=int,
                    help='n_units')
parser.add_argument('--batch', '-b', default=20, type=int,
                    help='batchsize')
parser.add_argument('--b_len', '-bl', default=30, type=int,
                    help='bprop_len')
parser.add_argument('--s_len', '-s', default=1200, type=int,
                    help='signal_len')
parser.add_argument('--cross_n', '-cn', default=4, type=int,
                    help='Cross-validation No.')
parser.add_argument('--gr_cli', '-gc', default=5, type=int,
                    help='grad_clip')
parser.add_argument('--drop_rate', '-dr', default=0.5, type=float,
                    help='drop out tate')
args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

n_epoch = args.n_epoch   # number of epochs
n_units = args.n_units  # number of units per layer
batchsize = args.batch   # minibatch size
bprop_len = args.b_len   # length of truncated BPTT
grad_clip = args.gr_cli    # gradient norm threshold to clip
len_sign = args.s_len#信号ぶったぎり
print("epoch:{} \nunit:{} \nbatch:{} \ntruncated time:{} \nsig length:{} \ncliping coef:{} \n".format(n_epoch,n_units,batchsize,bprop_len,len_sign,grad_clip))

day = datetime.datetime.today()#ファイル名とか用に日付取得
dayname = "{0}_{1}_{2}_{3}_{4}".format(day.year ,day.month, day.day ,day.hour,day.minute)
os.makedirs("result/{}".format(dayname))
os.mkdir("result/{}/model".format(dayname))

# Prepare dataset
x_train, x_test, t_train, t_test= mkd.mkf(args.cross_n)

jamp = len(x_train)/batchsize
if  not len(x_train)%batchsize == 0:
    jamp = len(x_train)//batchsize + 1#１エポックあたりこの回数回るよ（トレインデータを全部舐めまわすイメージ）

net = network.DRNN(3, n_units, 6)
model = L.Classifier(net)
model.compute_accuracy = True
for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

## Setup optimizer

optimizer = optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))# 勾配爆発を抑制

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_hdf5(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_hdf5(args.resume, optimizer)


f = open('result/{}/rezalt.txt'.format(dayname), 'w') # 学習記録用ファイルを書き込みモードで開く
f.write("scr{} e{} u{} batch{} bp{} sig{} cn{} gc{} dr{}\n".format(__file__,n_epoch,n_units,batchsize,bprop_len,len_sign,args.cross_n,grad_clip,args.drop_rate))

## Learning loop
start_at = time.time()#学習時間はかるために
cur_at = start_at
cur_log_perp = xp.zeros(())
cur_log_perp_e =xp.zeros(())
len_t = 0
len_t_e = 0
accum_loss = 0
sum_accuracy = 0
t_loss = 0
t_loss_e = 0
acu = 0#正答率計算用
acu_e = 0
loss_i_e = 0
time_hako = []
#図にするとき用とかの保存用箱
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for n in range(n_epoch):

    t_loss = 0
    t_loss_e = 0
    acu = 0
    acu_e = 0
    print("epoch {}".format(n))
    e_state_t = time.time()

    ran_tr = range(len(x_train))#訓練データシャッフル
    r_train_list = []
    r_train_label = []
    for t in ran_tr:
        r_train_list.append(x_train[t])
        r_train_label.append(t_train[t])
    x_train = r_train_list
    t_train = r_train_label

    for b,p in enumerate(range(jamp)):
        print("e{}_b{}".format(n,b))
        lool = []
        for jj in range(batchsize):
            lool.append((p*batchsize+jj)%len(x_train))
        accum_loss = 0
        z =[]
        for l in lool:#信号ぶっこむ最初の時刻をランダムで決めるその際はみ出さんようにする処理
            z.append(x_train[l].shape[0])
        len_t = min(z)
        l_max = len_t - (len_sign+1)
        r_l = random.randint(0,l_max)
        aaa = range(r_l,r_l+len_sign)
        sum_accuracy = 0

        for e ,v in enumerate(aaa):
            tra_x = []
            tes_t = []
            for q in lool:
                tra_x.append(x_train[q][v])
                tes_t.append(t_train[q])
            x = chainer.Variable(xp.asarray(tra_x, dtype=np.float32))
            t = chainer.Variable(xp.asarray(tes_t,dtype=np.int32))

            loss_i = model(x, t)
            accum_loss += loss_i

            cur_log_perp += loss_i.data
            sum_accuracy += float(model.accuracy.data)


            if (e + 1) % bprop_len == 0:  # Run truncated BPTT
                model.zerograds()
                accum_loss.backward()
                accum_loss.unchain_backward()  # truncate
                accum_loss = 0
                optimizer.update()
            #sys.stdout.flush()

        late_a = sum_accuracy/len(aaa)
        acu += late_a
        m_loss = float(cur_log_perp)/len(aaa)
        t_loss += m_loss
        cur_log_perp = xp.zeros(())

    now = time.time()
    time_hako.append(now - e_state_t)

    ##こっからtest
    model2 = model.copy()  # to use different state
    model2.predictor.reset_state()  # initialize stateこれしとかんとバッチサイズ変えられん
    model2.predictor.train = False#ドロップアウトをオフに

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model2.to_gpu()

    print("\nevaluate!")

    z =[]
    for l in x_test:
        z.append(l.shape[0])
    len_t = min(z)

    sum_accuracy = 0

    aaaa = range(len_t)

    for v in aaaa:
        test =[]
        for q in x_test:
            test.append(q[v])
        x = chainer.Variable(xp.asarray(test, dtype=np.float32))
        t = chainer.Variable(xp.asarray(t_test,dtype=np.int32))

        loss_i_e = model2(x, t)
        acu_e += float(model2.accuracy.data)
        t_loss_e += float(loss_i_e.data)

        sys.stdout.flush()

    rez1 = "train mean loss={0} epoch train time={1} accuracy={2}".format(t_loss/jamp, now - e_state_t, acu/jamp)
    rez2 = "test mean loss={0} accuracy={1}".format(t_loss_e/len(aaaa),acu_e/len(aaaa))
    print(rez1)
    print(rez2)
    f.write("epoch {}  ".format(n) + rez1 + " e " + rez2 + "\n")
    train_loss.append(t_loss/jamp)
    train_acc.append(acu/jamp)
    test_loss.append(t_loss_e/len(aaaa))
    test_acc.append(acu_e/len(aaaa))
    print("now max train acc = {} now max test acc = {}".format(max(train_acc),max(test_acc)))

    if n > -1:
        os.mkdir("result/{}/model/epoch{}".format(dayname,n))
        # Save the model and the optimizer
        print('save the model')
        serializers.save_hdf5('result/{}/model/epoch{}/HAR.model'.format(dayname,n), model)
        print('save the optimizer')
        serializers.save_hdf5('result/{}/model/epoch{}/HAR.state'.format(dayname,n), optimizer)

    #正答率、誤差関数をcsvで保存
    fig_data = zip(train_acc,test_acc,train_loss,test_loss)
    s_f_d = np.array(fig_data)
    np.savetxt("result/{}/rez{}.csv".format(dayname,dayname),s_f_d,delimiter=",")

    # 精度と誤差をグラフ描画
    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    ax1.set_ylim([0,1])
    ax2.set_ylim([0,1])
    ax1.plot(range(len(train_acc)), train_acc)
    ax2.plot(range(len(test_acc)), test_acc)
    ax3.plot(range(len(train_loss)),train_loss)
    ax4.plot(range(len(test_loss)),test_loss)
    ax1.set_title("validity_train_acc")
    ax2.set_title("validity_test_acc")
    ax3.set_title("validity_train_loss")
    ax4.set_title("validity_test_loss")
    plt.savefig("result/{}/output_{}.png".format(dayname,dayname))

end_time = time.time()
f.write("total_time = {}({}h) \n".format(end_time - start_at,(end_time - start_at)/3600) + "total_train_time = {}({}h) \n".format(sum(time_hako),(sum(time_hako))/3600) + "max_train_acc = {}\n".format(max(train_acc)) +"max_test_acc = {}".format(max(test_acc)))
f.close() # ファイルを閉じる

#正答率、誤差関数をcsvで保存
fig_data = zip(train_acc,test_acc,train_loss,test_loss)
s_f_d = np.array(fig_data)
np.savetxt("result/{}/rez{}.csv".format(dayname,dayname),s_f_d,delimiter=",")

# 精度と誤差をグラフ描画
fig = plt.figure(figsize=(16,12))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.set_ylim([0,1])
ax2.set_ylim([0,1])
ax1.plot(range(len(train_acc)), train_acc)
ax2.plot(range(len(test_acc)), test_acc)
ax3.plot(range(len(train_loss)),train_loss)
ax4.plot(range(len(test_loss)),test_loss)
ax1.set_title("validity_train_acc")
ax2.set_title("validity_test_acc")
ax3.set_title("validity_train_loss")
ax4.set_title("validity_test_loss")
plt.savefig("result/{}/output_{}.png".format(dayname,dayname))
print("finish!")
