import tensorflow as tf
import numpy as np

from NeonTrainer import NeonTrainer

layers = tf.contrib.layers
metrics = tf.metrics
arg_scope = tf.contrib.framework.arg_scope


class MagnesiumModel(NeonTrainer):

    def __init__(self, neutron, session, graph=tf.Graph(), train_dir='/tmp/neon/'):
        super(MagnesiumModel, self).__init__(neutron, session, graph, train_dir)
        self.embedding_size = 200
        self.embeddings = None
        self.num_sampled = 64

    def load_batch(self, batch_size, is_training, num_threads):
        return self.neutron.load_batch(batch_size, is_training, num_threads)

    def model(self, input_data, num_classes, is_training):
        with self.graph.as_default():
            # create placeholders for feeding
            self.embeddings = tf.Variable(tf.random_uniform(
                [self.neutron.vocabulary_size, self.embedding_size],
                -1.0, 1.0), name="emeddings")
            embeddings = tf.nn.embedding_lookup(self.embeddings, input_data)
            nce_weights = tf.Variable(tf.truncated_normal(
                [self.neutron.vocabulary_size, self.embedding_size],
                stddev=1.0 / np.sqrt(self.embedding_size)), name="nce_weights")
            nce_biases = tf.Variable(tf.zeros(self.neutron.vocabulary_size), name="nce_biases")
            tf.add_to_collection('nce_params', nce_weights)
            tf.add_to_collection('nce_params', nce_biases)
            return embeddings

    def losses(self, targets, logits):
        nce_weights, nce_biases = tf.get_collection('nce_params')
        nce_loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                                 biases=nce_biases,
                                                 inputs=logits,
                                                 labels=targets,
                                                 num_sampled=self.num_sampled,
                                                 num_classes=self.neutron.vocabulary_size))
        return nce_loss

    def evaluate(self, checkpoint_dir, checkpoint_name=None, batch_size=32):
        pass

    def visualize(self):
        pass
