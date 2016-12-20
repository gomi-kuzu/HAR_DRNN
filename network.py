# -*- coding:utf-8 -*-
"""
DRNNの構築
"""
import chainer
import chainer.functions as F
import chainer.links as L


class DRNN(chainer.Chain):

    def __init__(self, in_s, n_units,out_s, train=True):
        super(DRNN, self).__init__(
            l1=L.LSTM(in_s, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.LSTM(n_units, n_units),
            l4=L.Linear(n_units, out_s),
        )
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()

    def __call__(self, x):
        h1 = self.l1(x)
        h2 = self.l2(F.dropout(h1, train=self.train))
        h3 = self.l3(F.dropout(h2, train=self.train))
        y = self.l4(F.dropout(h3, train=self.train))
        return y
