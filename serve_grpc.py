#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : serve_grpc.py
# @Author: harry
# @Date  : 2/28/21 6:31 AM
# @Desc  : Serve NaivePredictor over grpc

import grpc
import argparse

from naive_chatbot.predictor import NaivePredictor, NaivePredictorServicer
from naive_chatbot import predictor_pb2_grpc
from concurrent.futures import ThreadPoolExecutor


def serve(port: int = 20233):
    p = NaivePredictor(
        msgs_save_path='./model/msgs',
        times_save_path='./model/times',
        dictionary_save_path='./model/dictionary',
        lsi_save_path='./model/lsi',
        sim_index_save_path='./model/sim_index',
    )

    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    predictor_pb2_grpc.add_ChatPredictorServicer_to_server(
        NaivePredictorServicer(p), server,
    )
    server.add_insecure_port(f'[::]:{port}')
    print(f'starting server on port {port}...')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int, default=20233)
    args = parser.parse_args()
    serve(args.port)
