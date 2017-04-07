import pandas as pd
import numpy as np

from keras.preprocessing import sequence, text
from keras.models import model_from_json


test_file = './data/test.csv'


data = pd.read_csv(test_file)

#pre-processing
tk = text.Tokenizer(nb_words=200000)

max_len = 40
tk.fit_on_texts(list(data.question1.values.astype(str)) + list(data.question2.values.astype(str)))
x1 = tk.texts_to_sequences(data.question1.values.astype(str))
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tk.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("our_weights.h5")
print("Loaded model from disk")

y_pred = loaded_model.predict([x1,x2], batch_size=300, verbose=0)

print np.shape(y_pred)
df = pd.DataFrame(y_pred, columns=["colummn"])
df.to_csv('testset_label.csv', index=True)
