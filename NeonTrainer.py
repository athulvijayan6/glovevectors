import os
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
import abc


class NeonTrainer(object):
    """
        This example implements a stable training and evaluating framework for Deep Learning.
    This is a production friendly implementation for reuse and deployment.

    This is implemented to train a net with multiple GPU towers and CPU coordinating the training process.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 neutron,
                 session=None,
                 graph=tf.Graph(),
                 train_dir='/tmp/neon/train'):
        self.neutron = neutron
        self.session = session
        self.graph = graph
        self.train_dir = train_dir
        self.num_gpus = 1
        self.TOWER_NAME = 'universetower_neon'

        self.INITIAL_LEARNING_RATE = 1e-2
        self.NUM_EPOCHS_PER_DECAY = 350.0
        self.LEARNING_RATE_DECAY_FACTOR = 0.1
        self.DECAY_STEPS = 5000

        if not os.path.isdir(self.train_dir):
            os.makedirs(self.train_dir)

    @abc.abstractmethod
    def load_batch(self, batch_size, is_training, num_threads):
        """
        abstract method for loading a batch of data
        :param batch_size:
        :param is_training:
        :param num_threads:
        :return:
        """
        return

    @abc.abstractmethod
    def model(self, input_data, num_classes, is_training):
        """
        abstract method for model definition
        :param input_data:
        :param num_classes:
        :param is_training:
        :return:
        """
        return

    @abc.abstractmethod
    def losses(self, targets, logits):
        """
        abstract method for defining loss function
        :param targets:
        :param logits:
        :return:
        """
        return

    @abc.abstractmethod
    def evaluate(self, checkpoint_dir, batch_size, checkpoint_name=None, ):
        """
        abstract method for evaluating model
        :param checkpoint_dir:
        :param checkpoint_name:
        :param batch_size:
        :return:
        """
        return

    def train(self, batch_size, max_steps=10000, restore_path=None):
        assert batch_size % self.num_gpus == 0
        # Use cpu0 as the coordinator device.
        # CPU will act as a master and distribute training tasks to the slave GPUs
        with self.graph.as_default(), tf.device('/cpu:0'):
            # Each batch is split int num_gpus and create mini-batches.
            # Each GPU is given these mini-batches to compute gradient
            # The gradients from each GPU is collected by master CPU to update the weights
            # GPUs get synchronized at end of each batch. (or a set of mini-batches)
            global_step = tf.get_variable('global_step',
                                          [],
                                          initializer=tf.constant_initializer(0),
                                          trainable=False)
            learning_rate = tf.train.exponential_decay(self.INITIAL_LEARNING_RATE,
                                                       global_step=global_step,
                                                       decay_steps=self.DECAY_STEPS,
                                                       decay_rate=self.LEARNING_RATE_DECAY_FACTOR,
                                                       staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # Distribute training across multiple gpus
            # give a batch of data to each GPU for computing gradients; cpu collects the gradient and makes updates
            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                minibatch_size = int(batch_size / self.num_gpus)
                for i in range(self.num_gpus):
                    with tf.device('/cpu:%d' % i), tf.name_scope("%s_%d" % (self.TOWER_NAME, i)) as scope:
                        inputs, targets = self.load_batch(minibatch_size, is_training=True, num_threads=1)
                        logits = self.model(inputs,
                                            num_classes=self.neutron.vocabulary_size,
                                            is_training=True)
                        targets = tf.reshape(targets, [batch_size, 1])
                        loss = self.losses(targets, logits)
                        tf.add_to_collection('losses', loss)
                        losses = tf.get_collection('losses', scope=scope)
                        total_loss = tf.add_n(losses, name='total_loss')
                        tf.get_variable_scope().reuse_variables()

                        grads = optimizer.compute_gradients(total_loss)
                        tower_grads.append(grads)
            # synchronize across gpu to collect grads
            average_grads = []
            for grads_and_vars in zip(*tower_grads):
                grads_tmp = []
                for g, _ in grads_and_vars:
                    expanded_grad = tf.expand_dims(g, axis=0)
                    grads_tmp.append(expanded_grad)
                average_grad = tf.concat(axis=0, values=grads_tmp)
                average_grad = tf.reduce_mean(average_grad, axis=0)
                # take variable from first tower
                var = grads_and_vars[0][1]
                average_grads.append((average_grad, var))
            optimizer_step = optimizer.apply_gradients(average_grads, global_step=global_step)
            # define training op
            train_op = tf.group(optimizer_step)

            # add some summaries
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            summaries.append(tf.summary.scalar('learning_rate', learning_rate))
            summaries.append(tf.summary.scalar('total_loss', total_loss))
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))
            summary_op = tf.summary.merge(summaries)
            summary_writer = tf.summary.FileWriter(self.train_dir, self.graph)
            global_var_init = tf.global_variables_initializer()
            local_var_init = tf.local_variables_initializer()

            saver = tf.train.Saver(tf.trainable_variables())
            if restore_path:
                assert tf.gfile.Exists(restore_path)
                saver.restore(self.session, restore_path)
                print('pre-trained model restored from ' + restore_path)

            # start training
            coord = tf.train.Coordinator()
            self.session.run(global_var_init)
            self.session.run(local_var_init)
            threads = tf.train.start_queue_runners()
            print('training started at: %s' % datetime.now())
            try:
                for step in range(max_steps):
                    if coord.should_stop():
                        break
                    step_start_time = time.time()
                    _, step_loss, _global_step = self.session.run([train_op, total_loss, global_step])
                    duration = time.time() - step_start_time
                    assert not np.isnan(step_loss), "Optimization diverged at step: %d" % step
                    if step % 1 == 0:
                        # print status
                        examples_per_second = batch_size / duration
                        trainer_msg = ("Training %s: step %d, \
                        loss = %.3f (%.1f examples/seconds; %.3f seconds/batch)")
                        print(trainer_msg % (datetime.now(), step, step_loss, examples_per_second, duration))
                    if step % 100:
                        summary_str = self.session.run(summary_op)
                        summary_writer.add_summary(summary_str, global_step=_global_step)
                    if (step % 1000) or (step == max_steps - 1):
                        checkpoint_path = os.path.join(self.train_dir, 'latest_model.ckpt')
                        saver.save(self.session, checkpoint_path, global_step=global_step)
            except tf.errors.OutOfRangeError:
                print("Training finished as the data is exhausted")
            finally:
                print("Training ended. Do eval or inspect the summaries for model analysis")
                coord.request_stop()
            coord.join(threads=threads)

    def restore(self, path):
        pass

    def analyze_training(self):
        pass

    # TODO
    def deploy(self):
        pass

if __name__ == "__main__":
    print("abstract class. Please inherit the class to create your model")
