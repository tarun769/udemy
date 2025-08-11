from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np

### sentences
sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]

voc_size = 10000

### One Hot Representation
one_hot_repr=[one_hot(words,voc_size)for words in sent]

# make len of each sent equal by prefixing 0
sent_length=8
embedded_docs=pad_sequences(one_hot_repr,padding='pre',maxlen=sent_length)

'''
[[  0   0   0   0  48 186 253 213]
 [  0   0   0   64  48 186 253  99]]
'''
## feature representation
dim=10

model=Sequential()
model.add(Embedding(voc_size,dim))
model.compile('adam','mse')
_ = model(np.array(embedded_docs))
print(model.summary())





