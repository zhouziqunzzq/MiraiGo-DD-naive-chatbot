#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_naive_predictor.py
# @Author: harry
# @Date  : 2/28/21 5:15 AM
# @Desc  : Test naive predictor

from naive_chatbot.predictor import NaivePredictor


def test():
    p = NaivePredictor(
        msgs_save_path='./model/msgs',
        times_save_path='./model/times',
        dictionary_save_path='./model/dictionary',
        lsi_save_path='./model/lsi',
        sim_index_save_path='./model/sim_index',
    )
    pred_msgs = p.predict_one(
        '好家伙', n_prediction=5, time_offset_seconds=300, sim_cutoff=0.0, verbose=True,
    )
    print(pred_msgs)


if __name__ == '__main__':
    test()
