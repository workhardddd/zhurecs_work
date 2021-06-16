from builtins import *

from dataloader import DataLoader
import pandas as pd
import numpy as np
import tensorflow as tf

class sessionpadding():
	'''
	different sessions have different lengths, model needs same length for next thing;
	use keras to padding items in every session.
	datadf:dataloader.py data[4],data[5],data[6];

	'''

	def __init__(self,datadf,maxlen=20):
		self.datadf = datadf
		self.maxlen = maxlen

	def padding(self):
		self.padding_dict = self.datadf.sort_values(by=['TimeId']).groupby('SessionId')['ItemId'].apply(list).to_dict()
		list_v = [i for i in self.padding_dict.values()]
		list_k = [i for i in self.padding_dict.keys()]
		x_data = []
		y_data = []
		for i in list_v:
			x = i[1:]
			y = i[:-1]
			x_data.append(x)
			y_data.append(y)
		# use -1 padding values
		x_list_v_afterpadding = tf.keras.preprocessing.sequence.pad_sequences(sequences=x_data,maxlen=self.maxlen,dtype='int32',padding='post',truncating='post',value=-1).tolist()
		y_list_v_afterpadding = tf.keras.preprocessing.sequence.pad_sequences(sequences=y_data,maxlen=self.maxlen,dtype='int32',padding='post',truncating='post',value=-1).tolist()
		self.padding_x = dict(zip(list_k,x_list_v_afterpadding))
		self.padding_y = dict(zip(list_k,y_list_v_afterpadding))
		return self.padding_x,self.padding_y

class construct_adj():
	def __init__(self,datadf,num_nodes,max_degree,adj_info,latest_peruser_time):
		self.datadf = datadf
		self.num_nodes = num_nodes
		self.max_degree = max_degree
		self.adj_info = adj_info
		self.latest_peruser_time = latest_peruser_time
		self.visible_time = self.user_visiable_time()
		self.adj,self.deg = construct_adj()

	def construct_adj(self):
		'''
		Construct adj table used during training/validing/testing.
		'''
		adj = self.num_nodes*np.ones((self.num_nodes+1, self.max_degree), dtype=np.int32)
		deg = np.zeros((self.num_nodes,))
		missed = 0
		for nodeid in self.datadf.UserId.unique():
			neighbors = np.array([neighbor for neighbor in
		                      self.adj_info.loc[self.adj_info['Follower']==nodeid].Followee.unique()], dtype=np.int32)
			deg[nodeid] = len(neighbors)
			if len(neighbors) == 0:
				missed += 1
				continue
			if len(neighbors) > self.max_degree:
				neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
			elif len(neighbors) < self.max_degree:
				neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
			adj[nodeid, :] = neighbors
		return adj, deg

	def user_visiable_time(self):
		visible_time = []
		for l in self.latest_peruser_time:
			timeid = max(loc for loc, val in enumerate(l) if val == 'NULL') + 1
			visible_time.append(timeid)
		return visible_time

	def _remove_infoless(self):
		data = self.datadf.loc[self.deg[self.datadf['UserId']]!=0]
		saved_session_ids = []
		for sessid in self.datadf.SessionId.unique():
			userid,timeid = sessid.split('_')
			userid,timeid = int(userid),int(timeid)
			count1 = 0
			for neightbor in self.adj[userid, :]:
				if self.visible_time[neightbor] <= timeid and self.deg[neightbor] > 0:
					count2 = 0
					for second_neighbor in self.adj[neightbor, :]:
						if self.visible_time[second_neighbor] <= timeid:
							break
						count2 += 1
					if count2 < self.max_degree:
						break
				count1 += 1
			if count1 < self.max_degree:
				saved_session_ids.append(sessid)
		return saved_session_ids




if __name__=='__main__':
	#data_path = 'path/to/data'
	data_path = '/home/ubuntu454/software_zhu/zhurecs_work/data/DoubanMovie/'
	data = DataLoader(data_path).load_data()
	adj_info = data[0]
	latest_per_user_by_time = data[1]
	user_id_map = data[2]
	item_id_map = data[3]
	train_df = data[4]
	valid_df = data[5]
	test_df = data[6]
	train_padding_x,train_padding_y = sessionpadding(train_df).padding()
	#adj,deg = construct_adj(datadf=train_df,num_nodes=len(user_id_map),max_degree=50,adj_info=adj_info)
