#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : predictor.py
# @Author: harry
# @Date  : 2/28/21 4:41 AM
# @Desc  : Predict using trained LSI model and similarity index

import json
import os
import re
import zhconv
import pickle

from typing import Dict, List, Tuple, Any
from dateutil.parser import parse
from jieba import posseg as pseg
from gensim import corpora, models, similarities
from naive_chatbot.data_preprocess import tokenization


class NaivePredictor(object):
    @staticmethod
    def _load_model(
            msgs_save_path: str = 'msgs',
            times_save_path: str = 'times',
            dictionary_save_path: str = 'dictionary',
            lsi_save_path: str = 'lsi',
            sim_index_save_path: str = 'sim_index',
    ) -> Tuple[List[str], List[int], corpora.Dictionary, models.LsiModel, similarities.Similarity]:
        with open(msgs_save_path, 'rb') as f:
            msgs = pickle.load(f)
        with open(times_save_path, 'rb') as f:
            times = pickle.load(f)
        dictionary = corpora.Dictionary.load(dictionary_save_path)
        lsi = models.LsiModel.load(lsi_save_path)
        sim_index = similarities.Similarity.load(sim_index_save_path)
        return msgs, times, dictionary, lsi, sim_index

    def __init__(
            self,
            msgs_save_path: str = 'msgs',
            times_save_path: str = 'times',
            dictionary_save_path: str = 'dictionary',
            lsi_save_path: str = 'lsi',
            sim_index_save_path: str = 'sim_index',
    ):
        super(NaivePredictor, self).__init__()
        msgs, times, dictionary, lsi, sim_index = NaivePredictor._load_model(
            msgs_save_path, times_save_path, dictionary_save_path, lsi_save_path, sim_index_save_path
        )
        self.msgs = msgs
        self.times = times
        self.dictionary = dictionary
        self.lsi = lsi
        self.sim_index = sim_index

    def predict_one(
            self, msg: str,
            n_prediction: int = 5,
            time_offset_seconds: int = 120,
            sim_cutoff: float = 0.0,
            verbose: bool = False
    ) -> List[Tuple[str, float]]:
        msg = zhconv.convert(msg, 'zh-cn')
        new_tokens = tokenization(msg)
        if verbose:
            print(f'new_tokens: {new_tokens}')
        new_vec = self.dictionary.doc2bow(new_tokens)
        if len(new_vec) == 0:
            if verbose:
                print('new_vec is empty, returning empty prediction result')
            return []
        if verbose:
            print(f'new_vec: {new_vec}')
            for v in new_vec:
                print(f'{v} - {self.dictionary[v[0]]}')
        sims = self.sim_index[self.lsi[new_vec]]
        res = list(enumerate(sims))
        sorted_res = sorted(res, key=lambda x: x[1], reverse=True)

        pred_rst = []
        sorted_rst_idx = 0
        while sorted_rst_idx < len(sorted_res) and len(pred_rst) < n_prediction:
            t_msg_idx, t_msg_sim = sorted_res[sorted_rst_idx]
            sorted_rst_idx += 1

            if t_msg_idx == len(self.msgs) - 1:  # reach the end of msgs list
                continue

            t_msg = self.msgs[t_msg_idx]
            t_msg_time = self.times[t_msg_idx]
            pred_msg = self.msgs[t_msg_idx + 1]
            pred_msg_time = self.times[t_msg_idx + 1]
            if verbose:
                print(f't_msg: {t_msg}')
                print(f't_msg_sim: {t_msg_sim}')
                print(f't_msg_time: {t_msg_time}')
                print(f'pred_msg: {pred_msg}')
                print(f'pred_msg_time: {pred_msg_time}')

            if t_msg_sim < sim_cutoff or pred_msg_time - t_msg_time > time_offset_seconds:
                if verbose:
                    print(f'skipped because of sim_cutoff or time_offset_seconds')
                    print(f'==================')
                continue
            else:
                if verbose:
                    print(f'pred_msg appended')
                    print(f'==================')
                pred_rst.append((pred_msg, t_msg_sim))

        return pred_rst
