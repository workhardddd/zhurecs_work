from builtins import *

from dataloader import DataLoader
import pandas as pd
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


if __name__=='__main__':
	data_path = 'path/to/data'
	data = DataLoader(data_path).load_data()
	train_df = data[4]
	valid_df = data[5]
	test_df = data[6]
	train_padding_x,train_padding_y = sessionpadding(train_df).padding()
