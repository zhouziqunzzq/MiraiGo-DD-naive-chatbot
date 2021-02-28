#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : preprocess_data.py
# @Author: harry
# @Date  : 2/28/21 5:14 AM
# @Desc  : Preprocess data and build LSI model

from naive_chatbot.data_preprocess import *


def main():
    do_preprocess(
        logs_path='./logs',
        msgs_save_path='./model/msgs',
        times_save_path='./model/times',
        dictionary_save_path='./model/dictionary',
        lsi_save_path='./model/lsi',
        sim_index_save_path='./model/sim_index',
        filter_group_id=[825502693],
    )


if __name__ == '__main__':
    main()
