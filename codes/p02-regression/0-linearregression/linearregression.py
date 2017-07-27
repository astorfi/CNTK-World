import numpy as np
import matplotlib.pyplot as plt
import cntk
from cntk import Trainer, learning_rate_schedule, UnitType
from cntk.learners import sgd
from cntk.ops import *
from cntk.layers import default_options, Dense

#################
### parameter ###
#################

# data parameters
num_samples = 500

# Network parameters
input_dim = 1
num_outputs = 1

# Training parameters
initial_learning_rate = 0.001
minibatch_size = 25
num_samples_to_train = 400
num_minibatches_to_train = int(num_samples_to_train / minibatch_size)
num_iterations = 400



###################################
##### Arbitrary Data Creation #####
###################################

# random data
features = np.linspace(-1, 1, num_samples)
predictions = 2 * features + np.random.randn(*features.shape) * 0.5

# Plotting the scatter plot
plt.scatter(features, predictions, c='r')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# adding one dimension for further processing. Input must be formatted as (batch_size,1).
features = features[:,None]
predictions = predictions[:,None]


###################
##### Network #####
###################

# Output is a single node with a linear operation.
input = cntk.input_variable(input_dim)
label = cntk.input_variable(num_outputs)
pred = Dense(num_outputs)(input)

##################
###### Loss ######
##################

# Defining loss function and evaluation metric
loss = cntk.squared_error(pred, label)
eval_fun = cntk.squared_error(pred, label)


######################
###### Training ######
######################

# Instantiate the trainer object to drive the model training
learning_rate = learning_rate_schedule(initial_learning_rate, UnitType.minibatch)
optimizer_op = sgd(pred.parameters, learning_rate)
train_op = Trainer(pred, (loss, eval_fun), [optimizer_op])

for step in range(0, num_iterations):
    for batch_num in range(0, num_minibatches_to_train):
        batch_features = features[(batch_num * minibatch_size):(batch_num * minibatch_size + minibatch_size), :]
        batch_labels = predictions[(batch_num * minibatch_size):(batch_num * minibatch_size + minibatch_size), :]
        train_op.train_minibatch({input: batch_features, label: batch_labels})
        training_loss = train_op.previous_minibatch_loss_average
        eval_value = train_op.previous_minibatch_evaluation_average
        print("Minibatch: {0}, Loss: {1:.2f}".format(batch_num, training_loss))


##############################
###### Model Evaluation ######
##############################

# Test data
test_features = features[num_samples_to_train:num_samples, :]
test_labels = predictions[num_samples_to_train:num_samples, :]

# Train data
train_features = features[0:num_samples_to_train, :]
train_labels = predictions[0:num_samples_to_train, :]

# Print out our weight and bias
print("Trained parameters are: w= {0:.2f}, b={1:.2f}".format(pred.W.value[0][0], pred.b.value[0]))


##################
###### Plot ######
##################

# Evaluation of training set
plt.scatter(train_features[:,0], train_labels[:,0], c='b')
X = train_features[:,0]
Y = pred.W.value[0] * train_features[:,0] + pred.b.value[0]
plt.plot(X, Y, 'r')
plt.xlabel("Feature")
plt.ylabel("Predicted")
plt.show()

# Evaluation on test set
plt.scatter(test_features[:,0], test_labels[:,0], c='b')
X = test_features[:,0]
Y = pred.W.value[0] * test_features[:,0] + pred.b.value[0]
plt.plot(X, Y, 'r')
plt.xlabel("Feature")
plt.ylabel("Predicted")
plt.show()



