#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_grpc_client.py
# @Author: harry
# @Date  : 2/28/21 6:40 AM
# @Desc  : Test grpc client

import grpc
import argparse

from naive_chatbot.predictor_pb2 import *
from naive_chatbot.predictor_pb2_grpc import *


def test(port: int = 20233):
    channel = grpc.insecure_channel(f'localhost:{port}')
    stub = ChatPredictorStub(channel)

    req = PredictRequest()
    req.msg = '早上好！'
    req.n_prediction = 5
    req.time_offset_seconds = 300
    req.sim_cutoff = 0.0
    pred_rsts = stub.PredictOne(req)
    pred_rsts = pred_rsts.result
    for rst in pred_rsts:
        print(rst.msg)
        print(rst.sim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int, default=20233)
    args = parser.parse_args()
    test(args.port)
