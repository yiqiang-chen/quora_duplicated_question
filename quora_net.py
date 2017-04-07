import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential,Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.engine.topology import Merge
from keras.layers import TimeDistributed, Lambda, Input
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
from keras.models import model_from_json

#creat a branch of siamese network
def create_network(word_index_len, embedding_matrix):
	seq = Sequential()
	#pre-learned word embedding
	seq.add(Embedding(word_index_len + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
	seq.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))
	
	seq.add(Dense(300))
	seq.add(PReLU())
	seq.add(Dropout(0.2))
	seq.add(BatchNormalization())

	seq.add(Dense(300))
	seq.add(PReLU())
	seq.add(Dropout(0.2))
	seq.add(BatchNormalization())

	seq.add(Dense(300))
	
	return seq

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred)+(1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
	
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_accuracy(y_true, y_pred):
	y_class = K.switch(y_pred>1, 1, 0)
	return K.mean(K.equal(y_true, y_class), axis=-1)

if __name__ == "__main__":

	data = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')
	y = data.is_duplicate.values


	#pre-processing
	tk = text.Tokenizer(nb_words=200000)

	max_len = 40
	tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))
	x1 = tk.texts_to_sequences(data.question1.values)
	x1 = sequence.pad_sequences(x1, maxlen=max_len)

	x2 = tk.texts_to_sequences(data.question2.values.astype(str))
	x2 = sequence.pad_sequences(x2, maxlen=max_len)

	#print np.shape(x2)

	word_index = tk.word_index
	


	#load word embedding
	embeddings_index = {}
	f = open('data/glove.840B.300d.txt')
	for line in tqdm(f):
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()

	print('Found %s word vectors.' % len(embeddings_index))

	embedding_matrix = np.zeros((len(word_index) + 1, 300))
	for word, i in tqdm(word_index.items()):
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector


	#siamese net
	input_a = Input(shape=(40,))
	input_b = Input(shape=(40,))
	network = create_network(len(word_index), embedding_matrix)
	output_a = network(input_a)
	output_b = network(input_b)

	#cos_distance = merge([output_a, output_b], mode='cos', dot_axes=1) 
	#cos_distance = Reshape((1,))(cos_distance)
	#cos_similarity = Lambda(lambda x: 1-x)(cos_distance)

	distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([output_a, output_b])

	model = Model([input_a, input_b], distance)

	model.compile(loss=contrastive_loss, optimizer='adam', metrics=['accuracy'])
	checkpoint = ModelCheckpoint('our_weights.h5', monitor='val_loss', save_best_only=True, verbose=2)

	#model.fit([x1, x2], y=y, batch_size=384, nb_epoch=20,verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])

	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
