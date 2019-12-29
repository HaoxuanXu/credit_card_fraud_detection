import numpy as np
import pandas as pd
import tensorflow as tf
import time
import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")

credit_info = pd.read_csv("creditcard.csv")

# Shuffle data
credit_info_shuffled = credit_info.sample(frac=1)
# one-hot encode the target variable
credit_info_encoded = pd.get_dummies(credit_info_shuffled, columns=["Class"])
# min-max normalize the data
credit_info_normalized = (credit_info_encoded - credit_info_encoded.min()) / (
        credit_info_encoded.max() - credit_info_encoded.min())
# Separating the inputs and the target values
Inputs = credit_info_normalized.iloc[:, :-2]
Targets = credit_info_normalized.iloc[:, -2:]
# Converting Inputs and Target to numpy arrays
Inputs_array = np.asarray(Inputs.values, "float32")
Targets_array = np.asarray(Targets.values, "float32")
# Splitting the data set into training set and testing set
training_size = int(0.8 * len(Inputs_array))
(X_train, y_train) = (Inputs_array[:training_size], Targets_array[:training_size])
(X_test, y_test) = (Inputs_array[training_size:], Targets_array[training_size:])
# Adding weighting to the fraud class to tackle the imbalanced data problem
fraud_ratio = credit_info["Class"].value_counts()[1] / len(credit_info)
fraud_weighting = 1 / fraud_ratio
y_train[:, 1] = y_train[:, 1] * fraud_weighting

############ BUILDING THE COMPUTATIONAL GRAPH ########################
# Allow me to construct a tensor of 30 elements corresponding to the Inputs column number
Inputs_dimensions = Inputs_array.shape[1]
Outputs_dimensions = Targets_array.shape[1]
hidden_layer_1_cells = 100
hidden_layer_2_cells = 150

X_train_node = tf.compat.v1.placeholder("float32", [None, Inputs_dimensions], name="X_train")
y_train_node = tf.compat.v1.placeholder("float32", [None, Outputs_dimensions], name="y_train")
X_test_node = tf.compat.v1.constant(X_test, name="X_test")
y_test_node = tf.compat.v1.constant(y_test, name="y_test")

# weights and biases for the 1st hidden layer
weights_node_1 = tf.compat.v1.Variable(tf.compat.v1.zeros([Inputs_dimensions, hidden_layer_1_cells], name="weight_1"))
biases_node_1 = tf.compat.v1.Variable(tf.compat.v1.zeros([hidden_layer_1_cells], name="biases_1"))
# weights and biases for the 2nd hidden layer
weights_node_2 = tf.compat.v1.Variable(tf.compat.v1.zeros([hidden_layer_1_cells, hidden_layer_2_cells], name="weight_2"))
biases_node_2 = tf.compat.v1.Variable(tf.compat.v1.zeros([hidden_layer_2_cells], name="biases_2"))
# weights and biases for the 3rd hidden layer
weights_node_3 = tf.compat.v1.Variable(tf.compat.v1.zeros([hidden_layer_2_cells, Outputs_dimensions], name="weight_3"))
biases_node_3 = tf.compat.v1.Variable(tf.compat.v1.zeros([Outputs_dimensions], name="biases_3"))


# Build the network function to connection the layers
def network_function(input_tensor):
    hidden_layer_1 = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(input_tensor, weights_node_1) + biases_node_1)
    hidden_layer_2 = tf.compat.v1.nn.dropout(
        tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(hidden_layer_1, weights_node_2) + biases_node_2),
        rate=0.15)
    output_layer = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(hidden_layer_2, weights_node_3) + biases_node_3)
    return output_layer


y_train_prediction = network_function(X_train_node)
y_test_prediction = network_function(X_test_node)

# Compute the loss function
cross_entropy_loss = tf.compat.v1.losses.softmax_cross_entropy(y_train_node, y_train_prediction)
# Create an optimizer to minimize the loss
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0005).minimize(cross_entropy_loss)


# Calculate the accuracy of the prediction
def accuracy_calculate(actual_amount, predicted_amount):
    actual = np.argmax(actual_amount, axis=1)
    predicted = np.argmax(predicted_amount, axis=1)
    return 100 * np.sum(np.equal(predicted, actual)) / predicted.shape[0]


# Training the model
epochs = 10000
checkpoint = "./sigmoid_v2_weights.ckpt"

with tf.compat.v1.Session() as session:
    epochs_with_no_improvement = 0
    cross_entropy_record = []
    epochs_checking_threshold = 1000
    tf.compat.v1.global_variables_initializer().run()
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        # I don't really need to look at the first output (optimizer)
        _, cross_entropy_score = session.run([optimizer, cross_entropy_loss],
                                             feed_dict={X_train_node: X_train,
                                                        y_train_node: y_train})
        # print out cross entropy score every 5 epochs
        if epoch % 10 == 0:
            cross_entropy_record.append(cross_entropy_score)
            end_time = time.time()
            time_taken = end_time - start_time
            print("Epoch: {}".format(epoch),
                  "         Current Loss Score: {0:.4f}".format(cross_entropy_score),
                  "         Time needed to run 10 epochs: {} seconds".format(round(time_taken, 2))

                  )
            final_y_test = y_test_node.eval()
            final_y_test_prediction = y_test_prediction.eval()
            final_accuracy = accuracy_calculate(final_y_test, final_y_test_prediction)
            print("Current Overall Accuracy Rate: {0:.2f}%".format(final_accuracy))
            final_fraud_y_test = final_y_test[final_y_test[:,1] == 1]
            final_fraud_y_test_prediction = final_y_test_prediction[final_y_test[:,1] == 1]
            final_fraud_accuracy = accuracy_calculate(final_fraud_y_test, final_fraud_y_test_prediction)
            print("Current Fraud Prediction Accuracy Rate: {0:.2f}%".format(final_fraud_accuracy))

        if epoch % 100 == 0:
            print("I will save this weight file now!")
            saver = tf.compat.v1.train.Saver()
            saver.save(session, checkpoint)
            if cross_entropy_score < cross_entropy_record.min():
                print("Good training so far! Keep going!")
                epochs_with_no_improvement = 0
            else:
                print("Houston, we have a problem!!")
                epochs_with_no_improvement += 100
            if epochs_with_no_improvement == epochs_checking_threshold:
                print("Training Terminated. No more improvements can be made")
                break
        break





