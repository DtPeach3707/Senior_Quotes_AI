import tensorflow as tf
from tensorflow import concat
from tensorflow.keras.layers import Input, Dense, GRU, TimeDistributed, Dropout
from tensorflow.keras.models import Model
import numpy as np


def stack_ragged(tensors):  # So only middle dimension can have shape None, and for combining ragged tensors
    values = concat(tensors, axis=0)
    lens = tf.stack([tf.shape(t, out_type=tf.int64)[0] for t in tensors])
    return tf.RaggedTensor.from_row_lengths(values, lens)


class SeniorQuoteGenerator:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.network = self.build_network()
        self.network.compile(optimizer='adam', loss='categorical_crossentropy')

    def build_network(self):
        inputs = Input(shape=(None, self.input_dim), ragged=True)
        result = TimeDistributed(Dense(300, activation='relu'))(inputs)
        result = GRU(400, reset_after=False)(result)  # Tfjs doesn't support reset_after GRU layers
        for filters in [400, 300, 300]:
            result = Dropout(0.4)(result)
            result = Dense(filters, activation='relu')(result)
        result = Dense(self.input_dim, activation='softmax')(result)

        modell = Model(inputs=inputs, outputs=result, name='ConvNet')
        print(modell.summary())
        return modell

    def generate_question(self, alphanum, numalpha, threshold=0.02, thresholded=True, max_char=200):
        q_start = [0.0] * self.input_dim
        q_start[int(self.input_dim) - 1] = 1.0
        question = [q_start]
        finished = False
        char_count = 1
        while not finished:
            char_predict = self.network.predict(np.array([question]))
            next_char_lis = [0.0] * self.input_dim
            if thresholded:
                thresholded_probs = []
                prob_tot = 0
                ind = 0
                indexes = []
                for prob in char_predict[0]:
                    if prob > threshold:
                        prob_tot += prob
                        thresholded_probs.append(prob)
                        indexes.append(ind)
                        ind += 1
                ind = 0
                next_char = np.random.choice(indexes, p=np.array(thresholded_probs)/prob_tot)
            else:
                next_char = np.random.choice(np.arange(0, self.input_dim), p=char_predict[0])
            next_char_lis[next_char] = 1.0
            question.append(next_char_lis)
            if next_char == alphanum['|']:
                finished = True
            char_count += 1
            if char_count == max_char:
                finished = True
        text_q = ''
        for q in question:
          text_q += numalpha[np.argmax(q)]
        print(text_q)
