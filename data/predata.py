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

class construct_graph():
	def __init__(self,datadf,num_nodes,max_degree,adj_info,latest_peruser_time):
		self.datadf = datadf
		self.num_nodes = num_nodes
		self.max_degree = max_degree
		self.adj_info = adj_info
		self.latest_peruser_time = latest_peruser_time
		self.visible_time = self.user_visiable_time()
		self.adj,self.deg = self.construct_adj()

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
				lentmp = self.max_degree - len(neighbors)
				neighborstmp = np.random.choice(neighbors,lentmp,replace=True)
				neighbors = np.concatenate((neighbors,neighborstmp))
			adj[nodeid, :] = neighbors
		return adj, deg

	def user_visiable_time(self):
		visible_time = []
		for l in self.latest_peruser_time:
			timeid = max(loc for loc, val in enumerate(l) if val == 'NULL') + 1
			visible_time.append(timeid)
			assert timeid > 0 and timeid < len(l), 'Wrong when create visible time {}'.format(timeid)
		return visible_time

	def remove_infoless(self):
		data = self.datadf.loc[self.deg[self.datadf['UserId']]!=0]
		saved_session_ids = []
		for sessid in data.SessionId.unique():
			userid,timeid = sessid.split('_')
			userid,timeid = int(userid),int(timeid)
			count1 = 0
			for neightbor in self.adj[userid, :]:
				if self.visible_time[neightbor] <= timeid and self.deg[neightbor] > 0:
					count1 += 1

			if count1 < self.max_degree and count1>0:
				saved_session_ids.append(sessid)
		return saved_session_ids

	def make_layer_1(self,num_samples_1,nodeids,timeids):
		adj_lists_1 = []
		num_samples_1 = num_samples_1
		nodeids = nodeids
		timeids = timeids
		for idx in range(len(nodeids)):
			node = nodeids[idx]
			timeid = timeids[idx]
			adj = self.adj[node,:]
			neighbors = []
			for neighbor in adj:
				if self.visible_time[neighbor] <= timeid and self.deg[neighbor] > 0:
					# for second_neighbor in self.adj[neighbor]:
					# 	if self.visible_time[second_neighbor] <= timeid:
					# 		for second_neighbor in self.adj[neighbor]:
					# 			if self.visible_time[second_neighbor] <= timeid:
					# 				neighbors.append(neighbor)
					# 				break
					neighbors.append(neighbor)

			assert len(neighbors) > 0
			if len(neighbors) < num_samples_1:
				neighbors = np.random.choice(neighbors, num_samples_1, replace=True)
			elif len(neighbors) >= num_samples_1:
				neighbors = np.random.choice(neighbors, num_samples_1, replace=False)
			adj_lists_1.append(neighbors)
		return np.array(adj_lists_1,dtype = np.int32)

	def support_node(self):
		self.sessionids = self.remove_infoless()
		nodeids = [int(sessionid.split('_')[0]) for sessionid in self.sessionids]
		timeids = [int(sessionid.split('_')[1]) for sessionid in self.sessionids]
		self.layer1 = self.make_layer_1(num_samples_1=10,nodeids=nodeids,timeids=timeids)
		support_layer1 = []
		idx = 0
		for sessionid in self.sessionids:
			# userid = int(sessionid.split('_')[0])
			timeid = int(sessionid.split('_')[1])
			layer1 = self.layer1[idx]
			per_layer = []
			for support_node in layer1:
				support_session_id = str(self.latest_peruser_time[support_node][timeid])
				# support_session_id = str(support_node) + '_' + support_session_id
				per_layer.append(support_session_id)
			support_layer1.append(np.array(per_layer))
		return np.array(support_layer1)









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

	train_layer1 = construct_graph(datadf=train_df,num_nodes=len(user_id_map),
	                     max_degree=50,adj_info=adj_info,
	                     latest_peruser_time=latest_per_user_by_time).support_node()

	print(1)