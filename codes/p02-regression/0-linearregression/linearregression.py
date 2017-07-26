#
# from sklearn import datasets
# from sklearn.model_selection import cross_val_predict
# from sklearn import linear_model
# import matplotlib.pyplot as plt
#
# lr = linear_model.LinearRegression()
# boston = datasets.load_boston()
# y = boston.target
#
# # cross_val_predict returns an array of the same size as `y` where each entry
# # is a prediction obtained by cross validation:
# predicted = cross_val_predict(lr, boston.data, y, cv=10)
#
# fig, ax = plt.subplots()
# ax.scatter(y, predicted)
# ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# plt.show()
#
# sys.exit()

# # Import the relevant components
# from __future__ import print_function
# import numpy as np
# import sys
# import os
# from cntk import *
# import matplotlib.pyplot as plt
from sklearn import datasets
#
# import cntk as C
#
#
# # Define the network
# input_dim = 2
# num_output_classes = 2
#
# # Ensure we always get the same amount of randomness
# np.random.seed(0)
#
#
# # Helper function to generate a random data sample
# def generate_random_data_sample(sample_size, feature_dim, num_classes):
#     # Create synthetic data using NumPy.
#     Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)
#
#     # Make sure that the data is separable
#     X = (np.random.randn(sample_size, feature_dim) + 3) * (Y + 1)
#
#     # Specify the data type to match the input variable used later in the tutorial
#     # (default type is double)
#     X = X.astype(np.float32)
#
#     # converting class 0 into the vector "1 0 0",
#     # class 1 into vector "0 1 0", ...
#     class_ind = [Y == class_number for class_number in range(num_classes)]
#     Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
#     return X, Y
#
#
#
# # Create the input variables denoting the features and the label data. Note: the input
# # does not need additional info on number of observations (Samples) since CNTK creates only
# # the network topology first
# mysamplesize = 32
# features, labels = generate_random_data_sample(mysamplesize, input_dim, num_output_classes)
#
boston = datasets.load_boston()
labels = boston.target[:,None]
features = boston.data[:,0:1]

import numpy as np
import matplotlib.pyplot as plt
import cntk
from cntk import Trainer, learning_rate_schedule, UnitType
from cntk.learners import sgd
from cntk.ops import *
# from cntk.internal.utils import get_train_eval_criterion, get_train_loss
# from cntk.train.trainer import previous_minibatch_loss_average
# from cntk.train.trainer import previous_minibatch_evaluation_average
# from cntk.train.trainer import previous_minibatch_loss_average

previous_minibatch_loss_average = Trainer.previous_minibatch_loss_average
previous_minibatch_evaluation_average = Trainer.previous_minibatch_evaluation_average

from cntk.layers import default_options, Dense

# Uncomment the next line to take advantage of NVidia GPUs
# cntk.device.set_default_device(cntk.device.gpu(0))

# # Create the feature and the label data. For our toy example the features/labels represent a simple line passing through the origin
# sample_size = 500
# features = np.arange(0, sample_size, dtype=np.float32).reshape((sample_size, 1))
# labels = np.arange(0, sample_size, dtype=np.float32).reshape((sample_size, 1))

plt.scatter(features[:, 0:1], labels, c='r')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# Define the network as a single node with no activation function
input_dim = 1
num_outputs = 1
input = cntk.input_variable(input_dim)
label = cntk.input_variable(num_outputs)
z = Dense(num_outputs)(input)

# Setup loss and evaluation functions
loss = cntk.squared_error(z, label)
eval_error = cntk.squared_error(z, label)

# Instantiate the trainer object to drive the model training
learning_rate = 0.00001
lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
learner = sgd(z.parameters, lr_schedule)
trainer = Trainer(z, (loss, eval_error), [learner])

# Initialize the parameters for the trainer
minibatch_size = 25
num_samples_to_train = 400
num_minibatches_to_train = int(num_samples_to_train / minibatch_size)

# And train the network. I should probably shuffle the dataset before training , but it's a toy example and it works as-is
for no_iter in range(0, 2):
    for i in range(0, num_minibatches_to_train):
        train_features = features[(i * minibatch_size):(i * minibatch_size + minibatch_size), :]
        train_labels = labels[(i * minibatch_size):(i * minibatch_size + minibatch_size), :]
        trainer.train_minibatch({input: train_features, label: train_labels})
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        print("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}".format(i, training_loss, eval_error))

        # Run the trained model on our test data
test_features = features[400:500, :]
test_labels = labels[400:500, :]

test_eval_result = trainer.test_minibatch({input: test_features, label: test_labels})
print("Test Data Evaluation Error: {0:.2f}".format(test_eval_result))

# Print out our weight and bias
print("Our model trained parameters of: ", z.W.value, z.b.value)

# And test some random data well outside our training data set
out_of_sample_data = np.array([[-600], [9000], [5500], [7500]], dtype=np.float32)
result = z.eval({input: out_of_sample_data})
print("Out of sample test data: ", out_of_sample_data)
print("Returned values: ", result[:, 0])

plt.scatter(out_of_sample_data[:, 0], result[:, 0], c='r')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
