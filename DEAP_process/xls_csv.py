#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020.2.2 16:55
# @Author  : Tony
# @Site    : 
# @File    : xls_csv.py
# @Software: PyCharm

import pandas as pd

def xlsx_to_csv_pd():
	data_xls = pd.read_excel("participant_ratings.xls", index_col=0)
	data_xls.to_csv("participant_ratings.csv", encoding='utf-8')

if __name__ == '__main__':
	xlsx_to_csv_pd()
	print("Done!")