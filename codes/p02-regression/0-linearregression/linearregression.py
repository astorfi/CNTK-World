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
Num_samples = 500

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
features = np.linspace(-1, 1, Num_samples)
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

test_features = features[400:500, :]
test_labels = predictions[400:500, :]

train_features = features[0:400, :]
train_labels = predictions[0:400, :]

test_eval_result = train_op.test_minibatch({input: test_features, label: test_labels})
print("Test Data Evaluation Error: {0:.2f}".format(test_eval_result))

# Print out our weight and bias
print("Our model trained parameters of: ", pred.W.value, pred.b.value)

# And test some random data well outside our training data set
out_of_sample_data = np.array(test_features, dtype=np.float32)
result = pred.eval({input: out_of_sample_data})
print("Out of sample test data: ", out_of_sample_data)
print("Returned values: ", result[:, 0])

# plt.scatter(test_labels[:, 0], result[:, 0], c='r')
plt.scatter(test_features[:,0], test_labels[:,0], c='b')
X = test_features[:,0]
print(pred.W.value[0].shape)
Y = pred.W.value[0] * test_features[:,0] + pred.b.value[0]
plt.plot(X, Y, 'r')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# plt.scatter(test_labels[:, 0], result[:, 0], c='r')
plt.scatter(train_features[:,0], train_labels[:,0], c='b')
X = train_features[:,0]
print(pred.W.value[0].shape)
Y = pred.W.value[0] * train_features[:,0] + pred.b.value[0]
plt.plot(X, Y, 'r')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
