import numpy
import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
from operator import truediv


class Neural_Network:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        #set number of nodes in each input hidden output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #link weight matrices wih and who
        # weights inside the arrays are w_i_j where link is from node i to node j in the next layer
        #w11 w21
        #w12 w22

        print('Input Nodes: ', self.inodes, ' Hidden Nodes: ', self.hnodes, ' Output Nodes: ', self.onodes)

        #self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        #self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)

        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))


        #print('Matrix 1: \n', self.wih)
        #print('Matrix 2: \n', self.who)

        self.lr = learningrate

        #activation function
        self.activation_function = lambda x: scipy.special.expit(x)


    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        #error
        output_errors = targets - final_outputs
        #backward pass
        hidden_errors = np.dot(self.who.T, output_errors)
        #update weight
        self.who += self.lr * np.dot(output_errors * final_outputs * (1.0 - final_outputs), np.transpose(hidden_outputs))

        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), np.transpose(inputs))


    def query(self, inputs_list):

        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

        pass


#n = Neural_Network(inputnodes=3, hiddennodes=10, outputnodes=8, learningrate=0.2)

#a = n.query([0.1, 0.2, 0.5])
#print(a)


training_data_file = open('train.csv','r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#print(len(training_data_list))

example = training_data_list[0]
#print(example)

all_values_of_example_in_string = example.split(',')
#print(all_values_of_example_in_string)

all_values_of_example = np.asfarray(all_values_of_example_in_string)
#print(all_values_of_example)

label = all_values_of_example[0]
#print(label)

picture = all_values_of_example[1:]

#print(picture.shape)


plt.imshow(picture.reshape(28,28), cmap='Greys', interpolation='None')
plt.show()


#training

epochs = 100
output_nodes = 10
n = Neural_Network(inputnodes=784, hiddennodes=200, outputnodes=10, learningrate=0.1)
#print(n.who)

for e in range(epochs):

    print('Epoch: ', e + 1)
    i = 0
    for record in training_data_list:
        #print("item: ", i , " of ", len(training_data_list))

        all_values = record.split(',')
        inputs = ((numpy.asfarray(all_values[1:])) / 255.0 * 0.99) + 0.01

        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)
        #i+= 1

#print(n.who)

test_data_file = open('test.csv','r')
test_data_list = test_data_file.readlines()
test_data_file.close()

record = test_data_list[0]
all_values = record.split(',')

picture = numpy.asfarray(all_values[1:])
plt.imshow(picture.reshape(28,28), cmap='Greys', interpolation='None')
plt.show()

inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
#print(inputs)

outputs = n.query(inputs)
#print(outputs)

scorecard = []
overallScore = []
M=10
N=10
confusion_matrix_test = [ [0] * N for _ in range(M)]
confusion_matrix_train = [ [0] * N for _ in range(M)]

for record in training_data_list:
    all_values = record.split(',')

    correct_label = int(all_values[0])

    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    outputs = n.query(inputs)

    label = np.argmax(outputs)

    if(label == correct_label):
        scorecard.append(1)
        overallScore.append(1)

    else:
        scorecard.append(0)
        overallScore.append(0)

    confusion_matrix_train[correct_label][label] += 1

print(confusion_matrix_train)


tp = np.diag(confusion_matrix_train)
prec = list(map(truediv, tp, np.sum(confusion_matrix_train, axis=0)))
rec = list(map(truediv, tp, np.sum(confusion_matrix_train, axis=1)))
print('Precision TRAIN: {}\nRecall TRAIN: {}'.format(prec, rec))
print('Precision avg: {}\nRecall avg: {}'.format((sum(prec)/len(prec)), (sum(rec)/len(rec))))

f_score_train = []

for i in range(10):
    f_score_train.append(2*((prec[i]*rec[i])/(prec[i]+rec[i])))

print("Fscore TRAIN: {}".format(f_score_train))

scorecard_array = np.asarray(scorecard)
print('TRAIN performance: ', scorecard_array.sum() / scorecard_array.size)

scorecard = []

for record in test_data_list:
    all_values = record.split(',')

    correct_label = int(all_values[0])

    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    outputs = n.query(inputs)

    label = np.argmax(outputs)

    if(label == correct_label):
        scorecard.append(1)
        overallScore.append(1)


    else:
        scorecard.append(0)
        overallScore.append(0)

    confusion_matrix_test[correct_label][label] += 1


print(confusion_matrix_test)

tp = np.diag(confusion_matrix_test)
prec = list(map(truediv, tp, np.sum(confusion_matrix_test, axis=0)))
rec = list(map(truediv, tp, np.sum(confusion_matrix_test, axis=1)))
print('Precision TEST: {}\nRecall TEST: {}'.format(prec, rec))
print('Precision avg: {}\nRecall avg: {}'.format((sum(prec)/len(prec)), (sum(rec)/len(rec))))

f_score_test = []

for i in range(10):
    f_score_test.append(2*((prec[i]*rec[i])/(prec[i]+rec[i])))

print("Fscore TEST: {}".format(f_score_test))

scorecard_array = np.asarray(scorecard)
print('test performance: ', scorecard_array.sum() / scorecard_array.size)

overallScore_array = np.asarray(overallScore)
print('overall performance: ', overallScore_array.sum() / overallScore_array.size)
