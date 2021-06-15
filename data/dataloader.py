from builtins import *

import numpy as np
import pandas as pd
import random

class DataLoader():
	def __init__(self,data_path):
		self.data_path = data_path

	def load_adj(self):
		df_adj = pd.read_csv(open(self.data_path) + '/adj.tsv', sep='\t', dtype={0:np.int32, 1:np.int32})
		return df_adj

	def load_latest_session(self):
		ret = []
		for line in open(self.data_path + '/latest_sessions.txt'):
			chunks = line.strip().split(',')
			ret.append(chunks)
		return ret

	def load_map(self, name='user'):
		if name == 'user':
			file_path = self.data_path + '/user_id_map.tsv'
		elif name == 'item':
			file_path = self.data_path + '/item_id_map.tsv'
		else:
			raise NotImplementedError
		id_map = {}
		for line in open(file_path):
			k, v = line.strip().split('\t')
			id_map[k] = str(v)
		return id_map

	def load_data(self):
		adj = self.load_adj()
		latest_sessions = self.load_latest_session()
		user_id_map = self.load_map('user')
		item_id_map = self.load_map('item')
		train = pd.read_csv(open(self.data_path + '/train.tsv'), sep='\t', dtype={0:np.int32, 1:np.int32, 3:np.float32})
		valid = pd.read_csv(open(self.data_path + '/valid.tsv'), sep='\t', dtype={0:np.int32, 1:np.int32, 3:np.float32})
		test = pd.read_csv(open(self.data_path + '/test.tsv'), sep='\t', dtype={0:np.int32, 1:np.int32, 3:np.float32})
		return [adj, latest_sessions, user_id_map, item_id_map, train, valid, test]



if __name__ == '__main__':
	# test data loading.
	data_path = 'path/to/data/'
	data = DataLoader(data_path).load_data()



