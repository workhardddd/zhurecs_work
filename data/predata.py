from builtins import *

from dataloader import DataLoader
import pandas as pd
import tensorflow as tf
data_path = 'path/to/data'
data = DataLoader(data_path).load_data()
train_df = data[4]
valid_df = data[5]
test_df = data[6]


class sessionpadding():
	'''
	different sessions have different lengths, model needs same length for next thing;
	use keras to padding items in every session.
	datadf:dataloader.py data[4],data[5],data[6];

	'''

	def __init__(self,datadf):
		self.datadf = datadf

	def padding(self):
		self.padding_dict = self.datadf.sort_values(by=['TimeId']).groupby('SessionId')['ItemId'].apply(list).to_dict()
		list_v = [i for i in self.padding_dict.values()]
		list_k = [i for i in self.padding_dict.keys()]
		list_v_afterpadding = tf.keras.preprocessing.sequence.pad_sequences(sequences=list_v,maxlen=20,dtype='int32',padding='post',truncating='post',value=0).tolist()
		self.padding_dict = dict(zip(list_k,list_v_afterpadding))
		return self.padding_dict
