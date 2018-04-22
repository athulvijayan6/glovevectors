import os
import sys
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from word2vec.MagnesiumModel import MagnesiumModel

AI_HOME = os.environ['AI_HOME']
AI_DATA = os.environ['AI_DATA']
sys.path.append(os.path.join(AI_HOME, 'language.learn'))
from datasets.SkipGramMattMahoney import SkipGramMattMahoney

data_dir = os.path.join(AI_DATA, 'mattmahoney')
train_dir = os.path.join(AI_DATA, 'word2vec', 'model')
batch_size = 128
num_skips = 1
skip_window = 2


def test_input():
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    graph = tf.Graph()
    neutron = SkipGramMattMahoney(data_dir, graph)
    neutron.download_and_convert_skipgram(num_skips=num_skips, skip_window=skip_window)
    reversed_dictionary = neutron.load_reversed_vocabulary()
    with tf.Session(graph=graph) as session:
        with graph.as_default():
            # Start queues to fetch data
            inputs, targets = neutron.load_batch(batch_size=batch_size, is_training=True)
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=session, coord=coord)
            session.run(tf.global_variables_initializer())
            try:
                inputs, targets = session.run([inputs, targets])
                for i, t in zip(inputs, targets):
                    print(i, t)
                    print(reversed_dictionary[i], reversed_dictionary[t])
            except OutOfRangeError as e:
                print("out of range")


def train():
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    graph = tf.Graph()
    neutron = SkipGramMattMahoney(data_dir, graph)
    # neutron.download_and_convert_skipgram(num_skips=num_skips, skip_window=skip_window)
    with tf.Session(graph=graph) as session:
        with graph.as_default():
            magnesium = MagnesiumModel(neutron=neutron,
                                       session=session,
                                       graph=graph,
                                       train_dir=train_dir)
            magnesium.train(batch_size)


if __name__ == "__main__":
    # test_input()
    train()
