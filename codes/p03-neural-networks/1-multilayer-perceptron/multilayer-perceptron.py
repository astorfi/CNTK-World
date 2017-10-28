# from __future__ import print_function
# import numpy as np
# import cntk as C
# from cntk.learners import sgd, learning_rate_schedule, UnitType
# from cntk.logging import ProgressPrinter
# from cntk.layers import Dense, Sequential
#
# def generate_random_data(sample_size, feature_dim, num_classes):
#      # Create synthetic data using NumPy.
#      Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)
#
#      # Make sure that the data is separable
#      X = (np.random.randn(sample_size, feature_dim) + 3) * (Y + 1)
#      X = X.astype(np.float32)
#      # converting class 0 into the vector "1 0 0",
#      # class 1 into vector "0 1 0", ...
#      class_ind = [Y == class_number for class_number in range(num_classes)]
#      Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
#      return X, Y
#
# def ffnet():
#     inputs = 2
#     outputs = 2
#     layers = 2
#     hidden_dimension = 50
#
#     # input variables denoting the features and label data
#     features = C.input_variable((inputs), np.float32)
#     label = C.input_variable((outputs), np.float32)
#
#     # Instantiate the feedforward classification model
#     my_model = Sequential ([
#                     Dense(hidden_dimension, activation=C.sigmoid),
#                     Dense(outputs)])
#     z = my_model(features)
#
#     ce = C.cross_entropy_with_softmax(z, label)
#     pe = C.classification_error(z, label)
#
#     # Instantiate the trainer object to drive the model training
#     lr_per_minibatch = learning_rate_schedule(0.125, UnitType.minibatch)
#     progress_printer = ProgressPrinter(0)
#     trainer = C.Trainer(z, (ce, pe), [sgd(z.parameters, lr=lr_per_minibatch)], [progress_printer])
#
#     # Get minibatches of training data and perform model training
#     minibatch_size = 25
#     num_minibatches_to_train = 1024
#
#     aggregate_loss = 0.0
#     for i in range(num_minibatches_to_train):
#         train_features, labels = generate_random_data(minibatch_size, inputs, outputs)
#         # Specify the mapping of input variables in the model to actual minibatch data to be trained with
#         trainer.train_minibatch({features : train_features, label : labels})
#         sample_count = trainer.previous_minibatch_sample_count
#         aggregate_loss += trainer.previous_minibatch_loss_average * sample_count
#
#     last_avg_error = aggregate_loss / trainer.total_number_of_samples_seen
#
#     test_features, test_labels = generate_random_data(minibatch_size, inputs, outputs)
#     avg_error = trainer.test_minibatch({features : test_features, label : test_labels})
#     print(' error rate on an unseen minibatch: {}'.format(avg_error))
#     return last_avg_error, avg_error
#
# np.random.seed(98052)
# ffnet()
#
#
# sys.exit()

# from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from sklearn.datasets import fetch_mldata
import random
import cntk as C
import cntk.tests.test_utils
from sklearn.preprocessing import OneHotEncoder
# cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
# C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components
import argparse

class Batch_Reader(object):

    def __init__(self, data , label):
        self.data = data
        self.label = label
        self.num_sample = data.shape[0]

    def next_batch(self, batch_size):
        index = random.sample(range(self.num_sample), batch_size)
        return self.data[index,:].astype(float),self.label[index,:].astype(float)

#################
### parameter ###
#################

# Define the parser
parser = argparse.ArgumentParser()
parser.add_argument('-num', '--num_samples', type=int,
                    default=500, help='number of samples')
parser.add_argument('-lr', '--initial_learning_rate', type=float,
                    default=0.001, help='initial learning rate')
parser.add_argument('-num_t', '--num_samples_to_train', type=int,
                    default=400, help='number of samples for training')
parser.add_argument('-num_it', '--num_iterations', type=int,
                    default=400, help='number of iterations')
parser.add_argument('-bs', '--batch_size', type=int,
                    default=25, help='mini batch size')

args = parser.parse_args()


# Network has only one output which is its prediction.
input_dim = 784
num_output_classes = 10
num_minibatches_to_train = int(args.num_samples_to_train / args.batch_size)




num_hidden_layers = 2
hidden_layers_dim = 400

input = C.input_variable(input_dim)
label = C.input_variable(num_output_classes)

def create_model(features):
    with C.layers.default_options(init = C.layers.glorot_uniform(), activation = C.ops.relu):
            h = features
            for _ in range(num_hidden_layers):
                h = C.layers.Dense(hidden_layers_dim)(h)
            r = C.layers.Dense(num_output_classes, activation = None)(h)
            return r

z = create_model(input)

loss = C.cross_entropy_with_softmax(z, label)

label_error = C.classification_error(z, label)

# Instantiate the trainer object to drive the model training
learning_rate = 0.2
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
learner = C.sgd(z.parameters, lr_schedule)
trainer = C.Trainer(z, (loss, label_error), [learner])


# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=5):
    if len(a) < w:
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))

    return mb, training_loss, eval_error


# Initialize the parameters for the trainer
minibatch_size = 64
num_samples_per_sweep = 60000
num_sweeps_to_train_with = 10
num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size


mnist = fetch_mldata('MNIST original', data_home=os.path.dirname(os.path.abspath(__file__)))

train_data = mnist.data[:60000,:]
train_label = mnist.target[:60000]
enc = OneHotEncoder()
enc.fit(train_label[:,None])
onehotlabels = enc.transform(train_label[:,None]).toarray()

train_reader = Batch_Reader(train_data,onehotlabels)




# # Map the data streams to the input and labels.
# input_map = {
#     label  : reader_train.streams.labels,
#     input  : reader_train.streams.features
# }

# Run the trainer on and perform model training
training_progress_output_freq = 500

plotdata = {"batchsize":[], "loss":[], "error":[]}

for i in range(0, int(num_minibatches_to_train)):

    # Read a mini batch from the training data file
    batch_data, batch_label = train_reader.next_batch(batch_size=32)

    arguments = {input: batch_data, label: batch_label}
    trainer.train_minibatch(arguments=arguments)
    batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)

    if not (loss == "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)


sys.exit()





