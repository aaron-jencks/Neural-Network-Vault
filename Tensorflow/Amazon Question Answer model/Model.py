
# Tensorflow related
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import pandas as pd

# Keras
from tensorflow import keras
import keras.layers as layers
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K

# General imports
import gzip
import datetime
import QAStructure
import WordEmbeddings

# Multi-core processing
from multiprocessing import Pool, Array


print("Tensorflow Version: " + tf.__version__)

# Global variables, needed for multi-core processing
db = []
vocab = []

# Parses the database
def parse(path): 
	g = gzip.open(path, 'r') 
	for l in g: 
		yield eval(l)
		
def applyVocab(d):
	global db
	global vocab
	vocab.append([d['question'], d['answer']])
	# Formats the data into objects
	db.append(QAStructure.QAObject(d['questionType'], d['asin'], d['question'], d['answer']))
		
# The main subroutine to be run in the program
def main():
	global db
	global vocab
	vocab_map_size = 100000
	
	# Opens the database file
	print("Parsing data file")
	
	"""
	with Pool() as pool:
		pool.imap(applyVocab, parse('databases\qa_Appliances.json.gz'))
	"""
	
	for d in parse('databases\qa_Appliances.json.gz'):
		applyVocab(d)
	
	print("Created the database")
	#print(vocab[:5])
	
	# Sets up the training data
	
	print('Setting up the training data')
	
	df_train = QAStructure.get_dataframe(vocab[:int(len(vocab) / 2)])
	df_train.head()
	
	#print("Training data: \n{}".format(df_train))
	
	train_data = (np.array(df_train['text'].tolist(), dtype = object)[:, np.newaxis], np.asarray(pd.get_dummies(df_train.label), dtype = np.int8))
	
	print('Training data:')
	print(train_data[0][:5])
	print(train_data[1][:5])
	
	category_counts = len(df_train.label.cat.categories)
	
	del df_train
	
	# Sets up the testing data
	
	print('Setting up the testing data')
	
	df_test = QAStructure.get_dataframe(vocab[int(len(vocab) / 2):])
	df_test.head()
	
	#print("Testing data: \n{}".format(df_test))
	
	test_data = (np.array(df_test['text'].tolist(), dtype = object)[:, np.newaxis], np.asarray(pd.get_dummies(df_test.label), dtype = np.int8))
	
	del df_test
	
	print("Training entries: {}, labels: {}".format(len(train_data[0]), len(train_data[1])))
	print("Testing entries: {}, labels: {}".format(len(test_data[0]), len(test_data[1])))
	
	print("Creating word embeddings")
	print("Data size: ", len(db))
	
	#print('Testing encoder')
	#print(WordEmbeddings.UniversalEmbedding(train_data[0])[0])
	
	with tf.Session() as session:
	
		# This section must be run before accessing the universal sentence encoder
		print('Setting up tensorflow')
		
		run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
	
		session.run([tf.global_variables_initializer(), tf.tables_initializer()], options=run_options)
		
		print('Creating model')
		
		input_layer = layers.Input(shape=(1,), dtype=tf.string)
		
		with tf.device('/cpu:0'): # because the embedding is not valid with gpu processing apparently
		
			embed_size = WordEmbeddings.GetEmbeddingSize()
	
			embeddings = layers.Lambda(WordEmbeddings.UniversalEmbedding, output_shape=(embed_size,))(input_layer)
		
		hidden = layers.Dense(256, activation='relu')(embeddings)
		pred = layers.Dense(category_counts, activation='softmax')(hidden)
		
		model = Model(inputs=[input_layer], outputs=pred)
		model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		
		print('Saving model to file')
		plot_model(model, to_file = "model.png", show_shapes = True)
		
		print('Training')
		
		K.set_session(session)
		history = model.fit(train_data[0], train_data[1], validation_data = test_data, epochs = 10, batch_size = 32)
		model.save_weights('./model.h5')
	
	
	
if __name__ == "__main__":
    import sys
    main()