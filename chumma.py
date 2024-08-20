from tensorflow.keras.models import load_model, Model
import mne
import tensorflow as tf
import numpy as np

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.units = units
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, inputs):
        score = self.V(tf.nn.tanh(self.W(inputs)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)

        return context_vector

def detect(file):
    model=load_model("model1.h5", custom_objects={'SelfAttention': SelfAttention})
    print(file)
    def preprocess(file):
        # Define function to load and preprocess EDF file
        raw = mne.io.read_raw_edf(file)
        raw.load_data()
        raw.set_eeg_reference()
        raw.filter(l_freq=0.5, h_freq=40)
        # data = raw.get_data()
        epochs=mne.make_fixed_length_epochs(raw,duration=5,overlap=1)
        #data = np.transpose(data, (0, 2, 1))
        # data.filter(l_freq=0.5, h_freq=40)
        array=epochs.get_data()
        print(array.shape)
        return array

    # Define a function to pad sequences to a fixed length
    def pad_s(sequences, max_length):
        padded_sequences = []
        for seq in sequences:
            pad_width = max_length - seq.shape[1]
            if pad_width < 0:
                # Sequence is longer than the maximum length, truncate it
                padded_seq = seq[:max_length]
            else:
                # Pad the sequence
                padded_seq = np.pad(seq, ((0,0),(0, pad_width), (0, 0)), mode='constant')
            padded_sequences.append(padded_seq)
        return padded_sequences

    test_x=preprocess(file)
    # Define the maximum length of sequences
    max_length = 41
    test_x=pad_s([test_x],41)[0]

    from tensorflow.keras.preprocessing.sequence import pad_sequences
    max_length =1280
    def pad_se(sequences, max_length):
        padded_sequences = []
        for seq in sequences:
            pad_width = max_length - seq.shape[2]
            padded_seq = np.pad(seq, ((0,0),(0,0),(0, pad_width)), mode='constant')
            padded_sequences.append(padded_seq)
        return padded_sequences
    test_x1=pad_se([test_x],1280)[0]

    test_x1=np.moveaxis(test_x1,1,2)
    predictions=model.predict(test_x1)

    threshold = 0.005
    binary_predictions = (predictions >= threshold).astype(int)

    a=0
    b=0
    for i in binary_predictions:
        if i == [1]:
            a+=1
        else:
            b+=1
    if a<b:
        return "NEGATIVE"
    else:
        return "POSITIVE"