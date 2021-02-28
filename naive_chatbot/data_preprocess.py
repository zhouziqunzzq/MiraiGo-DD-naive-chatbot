#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_preprocess.py
# @Author: harry
# @Date  : 2/28/21 2:36 AM
# @Desc  : Read in logs and output chat corpus

import json
import os
import re
import zhconv
import pickle

from typing import Dict, List, Tuple, Any, Optional
from dateutil.parser import parse
from jieba import posseg as pseg
from gensim import corpora, models, similarities


def get_log_paths(base_path: str = './logs') -> List[str]:
    rst = []
    for root, dirs, files in os.walk(base_path, topdown=True):
        for name in files:
            cur_file_path = os.path.abspath(os.path.join(root, name))
            _, ext = os.path.splitext(cur_file_path)
            if ext == '.log':
                rst.append(cur_file_path)
    return rst


def process_one(
        log_file_path: str, filter_group_id: Optional[List[int]] = None
) -> Tuple[List[str], List[int]]:
    msgs, times = [], []

    image_tag_pattern = re.compile(r'\[Image:[\w{}\-]+\.[\w{}\-]+]')
    reply_tag_pattern = re.compile(r'\[Reply:[\d]+]')
    at_tag_pattern = re.compile(r'@[\w]+')

    with open(log_file_path, 'r') as f:
        for line in f:
            log_line = json.loads(line)
            # filter specific group id
            if filter_group_id is not None \
                    and 'GroupCode' in log_line \
                    and log_line['GroupCode'] not in filter_group_id:
                continue
            # filter out only group msg
            if 'module' in log_line and log_line['module'] == 'internal.logging' \
                    and 'from' in log_line and log_line['from'] == 'GroupMessage':
                msg = log_line['msg']
                time = int(parse(log_line['time']).timestamp())
                # filter [Image], [Reply], @
                msg = re.sub(image_tag_pattern, '', msg)
                msg = re.sub(reply_tag_pattern, '', msg)
                msg = re.sub(at_tag_pattern, '', msg)
                msg = msg.strip()
                if len(msg) == 0:
                    continue
                # convert to zh-ch
                msg = zhconv.convert(msg, 'zh-cn')
                # print(f'{time}: {msg}')
                msgs.append(msg)
                times.append(time)

    return msgs, times


def process_all(
        log_file_paths: List[str],
        filter_group_id: Optional[List[int]] = None,
) -> Tuple[List[str], List[int]]:
    msgs, times = [], []
    for p in log_file_paths:
        m, t = process_one(p, filter_group_id)
        msgs.extend(m)
        times.extend(t)
    return msgs, times


def tokenization(content: str) -> List[str]:
    """
    {标点符号、连词、助词、副词、介词、时语素、‘的’、数词、方位词、代词}
    {'x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r'}
    去除文章中特定词性的词
    :content str
    :return list[str]
    """
    # stop_flags = {'x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r'}
    stop_flags = {'x', 'uj'}
    stop_words = {'\r\n', '\n', '\xa0'}
    words = pseg.cut(content)
    return [word for word, flag in words if flag not in stop_flags and word not in stop_words]


def build_dictionary(tokens_list: List[List[str]]) -> corpora.Dictionary:
    return corpora.Dictionary(tokens_list)


def do_preprocess(
        logs_path: str,
        msgs_save_path: str = 'msgs',
        times_save_path: str = 'times',
        dictionary_save_path: str = 'dictionary',
        lsi_save_path: str = 'lsi',
        sim_index_save_path: str = 'sim_index',
        filter_group_id: Optional[List[int]] = None,
):
    # get log file list
    log_file_paths = get_log_paths(logs_path)
    print(f'#log files: {len(log_file_paths)}')

    # read in all logs and get filtered msgs & times, save them to pickle file
    msgs, times = process_all(log_file_paths, filter_group_id)
    assert len(msgs) == len(times)
    print(f'#filtered_msgs: {len(msgs)}')
    with open(msgs_save_path, 'wb') as f:
        pickle.dump(msgs, f)
    with open(times_save_path, 'wb') as f:
        pickle.dump(times, f)

    # perform tokenization
    print('tokenizing...')
    msgs = [tokenization(msg) for msg in msgs]
    print('building and saving dictionary...')
    dictionary = build_dictionary(msgs)
    dictionary.save(dictionary_save_path)
    print(f'#words in dictionary: {len(dictionary)}')

    # tokenize corpus
    print('tokenizing corpus...')
    corpus = [dictionary.doc2bow(msg) for msg in msgs]

    # build lsi model
    print('building and saving LSI model...')
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=500)
    lsi.save(lsi_save_path)

    # build similarity index
    print('building and saving similarity index...')
    index = similarities.Similarity(sim_index_save_path, lsi[corpus], num_features=lsi.num_topics)
    index.save(sim_index_save_path)


if __name__ == '__main__':
    do_preprocess(logs_path='../logs')
