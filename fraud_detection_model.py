from typing import Optional, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import time
import pyjokes
import warnings

warnings.filterwarnings("ignore")

credit_info = pd.read_csv("creditcard.csv")

# Shuffle data
credit_info_shuffled = credit_info.sample(frac=1)
# one-hot encode the target variable
credit_info_encoded = pd.get_dummies(credit_info_shuffled, columns=["Class"])

# Splitting the data into train and test
training_size = int(0.8 * len(credit_info_encoded))
Training_set = credit_info_encoded.iloc[:training_size, :]
Test_set = credit_info_encoded.iloc[training_size:, :]
# Separate out the input variables and the target variables
X_train = Training_set.iloc[:, :-2]
y_train = Training_set.iloc[:, -2:]
X_test = Test_set.iloc[:, :-2]
y_test = Test_set.iloc[:, -2:]
# min-max normalize the data using MinMaxScaler
normalizer = MinMaxScaler()
X_train_scaled = normalizer.fit_transform(X_train)
X_test_scaled = normalizer.transform(X_test)
# Adding weighting to the fraud class to tackle the imbalanced data problem
fraud_ratio = credit_info["Class"].value_counts()[1] / len(credit_info)
fraud_weighting = 1 / fraud_ratio
y_train.iloc[:, 1] = y_train.iloc[:, 1] * fraud_weighting * 1.5

# Convert the data frame to numpy array
X_train_np = np.asarray(X_train_scaled, "float32")
y_train_np = np.asarray(y_train, "float32")
X_test_np = np.asarray(X_test_scaled, "float32")
y_test_np = np.asarray(y_test, "float32")

############ BUILDING THE COMPUTATIONAL GRAPH ########################
# Allow me to construct a tensor of 30 elements corresponding to the Inputs column number
Inputs_dimensions = X_train_np.shape[1]
Outputs_dimensions = y_train_np.shape[1]
hidden_layer_1_cells = 150
hidden_layer_2_cells = 200

# Disable eager execution
tf.compat.v1.disable_eager_execution()

X_train_node = tf.compat.v1.placeholder("float32", [None, Inputs_dimensions], name="X_train")
y_train_node = tf.compat.v1.placeholder("float32", [None, Outputs_dimensions], name="y_train")
X_test_node = tf.compat.v1.constant(X_test_np, name="X_test")
y_test_node = tf.compat.v1.constant(y_test_np, name="y_test")

# weights and biases for the Input layer
weights_node_1 = tf.compat.v1.Variable(
    tf.compat.v1.random.normal([Inputs_dimensions, hidden_layer_1_cells], name="weight_1"))
biases_node_1 = tf.compat.v1.Variable(tf.compat.v1.random.normal([hidden_layer_1_cells], name="biases_1"))
# weights and biases for the 1st hidden layer
weights_node_2 = tf.compat.v1.Variable(
    tf.compat.v1.random.normal([hidden_layer_1_cells, hidden_layer_2_cells], name="weight_2"))
biases_node_2 = tf.compat.v1.Variable(tf.compat.v1.random.normal([hidden_layer_2_cells], name="biases_2"))
# weights and biases for the 2nd Hidden layer
weights_node_3 = tf.compat.v1.Variable(
    tf.compat.v1.random.normal([hidden_layer_2_cells, Outputs_dimensions], name="weight_3"))
biases_node_3 = tf.compat.v1.Variable(tf.compat.v1.random.normal([Outputs_dimensions], name="biases_3"))


# Build the network function to connection the layers
def network_function(input_tensor):
    hidden_layer_1 = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(input_tensor, weights_node_1) + biases_node_1)
    hidden_layer_2 = tf.compat.v1.nn.dropout(
        tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(hidden_layer_1, weights_node_2) + biases_node_2),
        rate=0.2)
    output_layer = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(hidden_layer_2, weights_node_3) + biases_node_3)
    return output_layer


y_train_prediction = network_function(X_train_node)
y_test_prediction = network_function(X_test_node)

# Compute the loss function
cross_entropy_loss = tf.compat.v1.losses.softmax_cross_entropy(y_train_node, y_train_prediction)
# Create an optimizer to minimize the loss
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy_loss)


# Calculate the accuracy of the prediction
def accuracy_calculate(actual_amount, predicted_amount):
    actual = np.argmax(actual_amount, axis=1)
    predicted = np.argmax(predicted_amount, axis=1)
    return 100 * np.sum(np.equal(predicted, actual)) / predicted.shape[0]


# Training the model
epochs = 5000
checkpoint = "./sigmoid_weights.ckpt"

with tf.compat.v1.Session() as session:
    tf.compat.v1.global_variables_initializer().run()
    precision_trajectory = []
    accuracy_trajectory = []
    recall_trajectory = []
    best_precision = 0
    best_recall = 0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        # I don't really need to look at the first output (optimizer)
        _, cross_entropy_score = session.run([optimizer, cross_entropy_loss],
                                             feed_dict={X_train_node: X_train_np,
                                                        y_train_node: y_train_np})

        # print out cross entropy score every 5 epochs
        if epoch % 10 == 0:
            end_time = time.time()
            time_taken = end_time - start_time

            # Evaluate the accuracy and precision score for each epoch to decide whether to save that epoch
            final_y_test = y_test_node.eval()
            final_y_test_prediction = y_test_prediction.eval()
            final_fraud_y_test = final_y_test[final_y_test[:, 1] == 1]
            final_fraud_y_test_prediction = final_y_test_prediction[final_y_test[:, 1] == 1]

            final_accuracy = accuracy_calculate(final_y_test, final_y_test_prediction)
            final_precision = accuracy_calculate(final_fraud_y_test, final_fraud_y_test_prediction)
            final_recall = 100 * np.sum(np.equal(final_fraud_y_test[:,1], final_fraud_y_test_prediction[:,1])) / \
                           final_fraud_y_test.shape[0]
            accuracy_trajectory.append(final_accuracy)
            precision_trajectory.append(final_precision)
            recall_trajectory.append(final_recall)

            if final_precision > best_precision or final_recall > best_recall:
                best_precision = final_precision
                best_recall = final_recall
                saver = tf.compat.v1.train.Saver()
                saver.save(session, checkpoint)

        if epoch % 100 == 0:
            print("Epoch: {}".format(epoch),
                  "         Current Loss Score: {0:.4f}".format(cross_entropy_score),
                  "         Time needed to run 10 epochs: {} seconds".format(round(time_taken, 2)))

            print("Current Overall Accuracy Rate: {0:.2f}%".format(final_accuracy))
            print("Current Fraud Prediction Accuracy Rate (Precision): {0:.2f}%".format(final_precision))
            print("Current Fraud Detection Rate (Recall): {0:.2f}%".format(final_recall))
            print(pyjokes.get_joke())

    print("Best Accuracy: {0: .4f}%".format(final_accuracy))
    print("Best Precision: {0: .2f}%".format(final_precision))
    print("Best Recall; {0: .2f}%".format(final_recall))
