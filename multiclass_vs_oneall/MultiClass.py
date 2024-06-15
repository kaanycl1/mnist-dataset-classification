############################################################################################
#               Implementation of MultiClass Perceptron.                                   #
############################################################################################
#Kaan Yücel - 150210318

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
import time


def visualizer(loss, accuracy, n_epochs):
    """
    Returns the plot of Training/Validation Loss and Accuracy.
    :param loss: A list defaultdict with 2 keys "train" and "val".
    :param accuracy: A list defaultdict with 2 keys "train" and "val".
    :param n_epochs: Number of Epochs during training.
    :return:
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    x = np.arange(0, n_epochs, 1)
    axs[0].plot(x, loss['train'], 'b')
    axs[0].plot(x, loss['val'], 'r')
    axs[1].plot(x, accuracy['train'], 'b')
    axs[1].plot(x, accuracy['val'], 'r')
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss value")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy value (in %)")

    axs[0].legend(['Training loss', 'Validation loss'])
    axs[1].legend(['Training accuracy', 'Validation accuracy'])


class OneVsAll:
    def __init__(self, x_train, y_train, x_test, y_test, alpha, beta, mb, n_class, F, n_epochs, info):
        """
        This is an implementation from scratch of Multi Class Perceptron using One vs All strategy,
        and Momentum with SGD optimizer.

        :param x_train: Vectorized training data.
        :param y_train: Label training vector.
        :param x_test: Vectorized testing data.
        :param y_test: Label test vector.
        :param alpha: The learning rate.
        :param beta: Momentum parameter.
        :param mb: Mini-batch size.
        :param n_class: Number of classes.
        :param F: Number of features.
        :param n_epochs: Number of Epochs.
        :param info: 1 to show training loss & accuracy over epochs, 0 otherwise.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.alpha = alpha
        self.beta = beta
        self.mb = mb
        self.n_class = n_class
        self.F = F
        self.n_epochs = n_epochs
        self.info = info

    def relabel(self, label):
        """
        This function takes a class, and relabels the training label vector into a binary class,
        it's used to apply One vs All strategy.

        :param label: The class to relabel.
        :return: A new binary label vector.
        """

        y = self.y_train.tolist()
        n = len(y)
        y_new = [1 if y[i] == label else 0 for i in range(n)]

        return np.array(y_new).reshape(-1, 1)

    def momentum(self, y_relab):
        """
        This function is an implementation of the momentum with SGD optimization algorithm, and it's
        used to find the optimal weight vector of the perceptron algorithm.
        :param y_relab: A binary label vector.
        :return: A weight vector, and history of loss/accuracy over epochs.
        """

        # Initialize weights and velocity vectors
        W = np.zeros((self.F + 1, 1))
        V = np.zeros((self.F + 1, 1))

        # Store loss & accuracy values for plotting
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        # Split into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(self.x_train, y_relab, test_size=0.1, random_state=42)
        n_train = len(x_train)
        n_val = len(x_val)

        ##############################################################################
        #                             YOUR CODE                                      #
        ##############################################################################

        for _ in range(self.n_epochs):

            start = time.time()
            train_loss = 0
            # Compute the loss and gradient over mini-batches of the training set
            for i in range(0, n_train - self.mb + 1, self.mb):
                x_batch = x_train[i:i + self.mb]
                y_batch = y_train[i:i + self.mb]
                x_batch = np.hstack((np.ones((len(x_batch), 1)), x_batch))  # for bias terms
                raw_predictions = np.dot(x_batch, W)
                # Apply sigmoid activation function
                sigmoid_scores = 1 / (1 + np.exp(-raw_predictions))
                pure_error = sigmoid_scores - y_batch
                # cross entropy error
                data_loss = -np.mean(y_batch * np.log(sigmoid_scores) + (1 - y_batch) * np.log(1 - sigmoid_scores))
                # Train loss
                train_loss += data_loss/len(x_batch)
                # gradient
                grad = np.dot(x_batch.T, pure_error) / len(x_batch)

                # Update velocity
                V = self.beta * V + self.alpha * grad

                # Update weights
                W -= V
            # Computing the training accuracy
            scores_train = np.dot(np.hstack((np.ones((n_train, 1)), x_train)), W)
            y_pred_train = (scores_train > 0).astype(int)  # convert probabilities to binary predictions
            train_acc = np.mean(y_pred_train == y_train) * 100

            # Computing the loss & accuracy over the validation set
            scores_val = np.dot(np.hstack((np.ones((len(x_val), 1)), x_val)), W)
            sigmoid_scores_val = 1 / (1 + np.exp(-scores_val))
            val_loss = -np.mean(y_val * np.log(sigmoid_scores_val) + (1 - y_val) * np.log(1 - sigmoid_scores_val))
            y_pred_val = (scores_val > 0).astype(int)
            val_acc = np.mean(y_pred_val == y_val) * 100

            ##############################################################################
            #                             END OF YOUR CODE                               #
            ##############################################################################

            end = time.time()
            duration = round(end - start, 2)

            if self.info: print("Epoch: {} | Duration: {}s | Train loss: {} |"
                                " Train accuracy: {}% | Validation loss: {} | "
                                "Validation accuracy: {}%".format(_, duration,
                                                                  round(train_loss, 5), train_acc, round(val_loss, 5),
                                                                  val_acc))

            # Append training & validation accuracy and loss values to a list for plotting
            loss['train'].append(train_loss)
            loss['val'].append(val_loss)
            accuracy['train'].append(train_acc)
            accuracy['val'].append(val_acc)

        return W, loss, accuracy

    def train(self):
        """
        This function trains the model using One-vs-All strategy, and returns a weight
        matrix, to be used during inference.
        :return: A weight matrix of size (F+1, n_class), where F is the number of features,
        and n_class is the number of classes to predict.
        """

        weights = []
        loss, accuracy = 0, 0

        ##############################################################################
        #                             YOUR CODE                                      #
        ##############################################################################

        for i in range(1, self.n_class + 1):
            print("-" * 50 + " Processing class {} ".format(i) + "-" * 50 + "\n")

            y_relab = self.relabel(i - 1)

            W, loss, accuracy = self.momentum(y_relab)
            weights.append(W)
        # Get the weights matrix as a numpy array
        weights = np.array(weights).reshape(10, 41).T
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return weights, loss, accuracy

    def test(self, weights):
        """
        This function is used to test the model over new testing data samples, using
        the weights matrix obtained after training.
        :param weights: A weight matrix of size (F+1, n_class), where F is the number of features,
        and n_class is the number of classes to predict.
        :return:
        """

        ##############################################################################
        #                             YOUR CODE                                      #
        ##############################################################################


        # Computing the predicted values
        scores = np.dot(np.hstack((np.ones((len(self.x_test), 1)), self.x_test)), weights)

        y_hat = np.argmax(scores, axis=1).reshape(-1)

        # Computing the test accuracy
        test_acc = np.mean(y_hat == self.y_test) * 100

        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        print("-" * 50 + "\n Test accuracy is {}%".format(test_acc))
