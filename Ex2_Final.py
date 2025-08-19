import matplotlib.pyplot as plt
import numpy as np
#import time
import random


# Define variables
numHiddenLayerOneNeurons = 0
numHiddenLayerTwoNeurons = 0
numInputNeurons = 0
numOutputNeurons = 0
learningRate = 0.0
momentum = 0.0
maxIterations = 0
trainFile = ""
testFile = ""

#open and read parameters file
with open('parameters.txt', 'r') as file:
    for line in file:
        parts = line.split()
        if len(parts) == 2:
            key, value = parts
            if key == 'numHiddenLayerOneNeurons':
                numHiddenLayerOneNeurons = int(value)
            elif key == 'numHiddenLayerTwoNeurons':
                numHiddenLayerTwoNeurons = int(value)
            elif key == 'numInputNeurons':
                numInputNeurons = int(value)
            elif key == 'numOutputNeurons':
                numOutputNeurons = int(value)
            elif key == 'learningRate':
                learningRate = float(value)
            elif key == 'momentum':
                momentum = float(value)
            elif key == 'maxIterations':
                maxIterations = int(value)
            elif key == 'trainFile':
                trainFile = value
            elif key == 'testFile':
                testFile = value

#load data
train_Data = []
test_Data = []
targetOutput = []

#normalize inputs
def normalize(data, min_val=0, max_val=15):
    return (np.array(data) - min_val) / (max_val - min_val)

#at first read all the data and seperate target output
with open(trainFile, 'r') as file:
    for line in file:
        parts = line.split(',')
        letter = parts[0].strip()
        input_values = [int(value) for value in parts[1:16]]
        normalized_inputs = normalize(input_values)
        target_index = ord(letter) - ord('A')
        target_output = [1 if i == target_index else 0 for i in range(26)]
        targetOutput.append(target_output)
        train_Data.append(normalized_inputs)

#combine data and targets for shuffling
combined = list(zip(train_Data, targetOutput))

#shuffle the combined data
random.shuffle(combined)

#unzip the shuffled data
train_Data[:], targetOutput[:] = zip(*combined)

#find index to split 70% training and 30% testing
split_index = int(len(train_Data) * 0.7)

# Split the data to train and test
trainData = train_Data[:split_index]
testData = train_Data[split_index:]

trainData = np.array(trainData)
targetOutput = np.array(targetOutput)

#initialize weights
limit = 0.1         #all are random values between the limit
weightsLayerOne = np.random.uniform(-limit, limit, (numHiddenLayerOneNeurons, numInputNeurons + 1))
weightsLayerTwo = np.random.uniform(-limit, limit, (numHiddenLayerTwoNeurons, numHiddenLayerOneNeurons + 1)) if numHiddenLayerTwoNeurons > 0 else np.array([])
weightsLayerOutput = np.random.uniform(-limit, limit, (numOutputNeurons, numHiddenLayerTwoNeurons + 1)) if numHiddenLayerTwoNeurons > 0 else np.random.uniform(-limit, limit, (numOutputNeurons, numHiddenLayerOneNeurons + 1))

#helper arrays to store weihts
previous_weightsLayerOne = np.zeros_like(weightsLayerOne)
previous_weightsLayerTwo = np.zeros_like(weightsLayerTwo) if numHiddenLayerTwoNeurons > 0 else np.array([])
previous_weightsLayerOutput = np.zeros_like(weightsLayerOutput)

train_error_epochs = []
test_error_epochs = []

#sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#sigmoid derivative function
def sigmoid_derivative(y):
    return y * (1 - y)

#calculate error from target and actual output
def calculate_error(target, actual_out):
    return 0.5 * np.sum((target - actual_out) ** 2)

#classify the answer for success rates
def classifiedCorrect(target, actualOutput):
    return np.argmax(target) == np.argmax(actualOutput)

def train():
    global weightsLayerOne, weightsLayerTwo, weightsLayerOutput, previous_weightsLayerOne, previous_weightsLayerTwo, previous_weightsLayerOutput
    
    for epoch in range(maxIterations):
        #start_time = time.time()
        train_error = 0.0
        train_success = 0.0
        test_error = 0.0
        test_success = 0.0
        
        for dataNum in range(len(trainData)):
            inputs = np.append(trainData[dataNum], 1)  #add bias to initial input
            
            # Forward pass
            hidden_layer1_output = sigmoid(np.dot(weightsLayerOne, inputs))     #hidden layer 1 output
            
            #check if i have two or one hidden layers
            if numHiddenLayerTwoNeurons > 0:
                hidden_layer1_output = np.append(hidden_layer1_output, 1)  #add bias to hidden layer1 output/hidden layer 2 input
                hidden_layer2_output = sigmoid(np.dot(weightsLayerTwo, hidden_layer1_output))
                hidden_layer2_output = np.append(hidden_layer2_output, 1)  #add bias to hidden layer2 output/output layer input
                output_layer_output = sigmoid(np.dot(weightsLayerOutput, hidden_layer2_output))
            else:
                hidden_layer1_output = np.append(hidden_layer1_output, 1)  #add bias to hidden layer1 output/output layer input
                output_layer_output = sigmoid(np.dot(weightsLayerOutput, hidden_layer1_output))
            
            # Backward pass
            output_layer_error = (targetOutput[dataNum] - output_layer_output) * sigmoid_derivative(output_layer_output)    #calculate output layer delta error
            
            #check if i have two or one hidden layers
            if numHiddenLayerTwoNeurons > 0:
                hidden_layer2_error = np.dot(weightsLayerOutput[:, :-1].T, output_layer_error) * sigmoid_derivative(hidden_layer2_output[:-1])  #calculate layer 2 delta error
                hidden_layer1_error = np.dot(weightsLayerTwo[:, :-1].T, hidden_layer2_error) * sigmoid_derivative(hidden_layer1_output[:-1])    #calculate layer 1 delta error
            else:
                hidden_layer1_error = np.dot(weightsLayerOutput[:, :-1].T, output_layer_error) * sigmoid_derivative(hidden_layer1_output[:-1])  #calculate layer 1 delta error
            
            #update output layer weights based on the current and previous weights
            new_weightsLayerOutput = weightsLayerOutput + learningRate * np.outer(output_layer_error, hidden_layer2_output if numHiddenLayerTwoNeurons > 0 else hidden_layer1_output) + momentum * (weightsLayerOutput - previous_weightsLayerOutput)
            previous_weightsLayerOutput = np.copy(weightsLayerOutput)
            weightsLayerOutput = new_weightsLayerOutput
            
            #update hidden layer 2 weights based on the current and previous weights(if there is second layer)
            if numHiddenLayerTwoNeurons > 0:
                new_weightsLayerTwo = weightsLayerTwo + learningRate * np.outer(hidden_layer2_error, hidden_layer1_output) + momentum * (weightsLayerTwo - previous_weightsLayerTwo)
                previous_weightsLayerTwo = np.copy(weightsLayerTwo)
                weightsLayerTwo = new_weightsLayerTwo
            
            #update hidden layer 1 weights based on the current and previous weights
            new_weightsLayerOne = weightsLayerOne + learningRate * np.outer(hidden_layer1_error, inputs) + momentum * (weightsLayerOne - previous_weightsLayerOne)
            previous_weightsLayerOne = np.copy(weightsLayerOne)
            weightsLayerOne = new_weightsLayerOne

            #compute training error
            train_error += calculate_error(targetOutput[dataNum], output_layer_output)
            
            #find if the answer is correct or not
            if classifiedCorrect(targetOutput[dataNum], output_layer_output) == True:
                train_success += 1
        
        #save calculations
        train_error_epochs.append([epoch, train_error])
        train_success = train_success / len(trainData)

        #testing
        for dataNum in range(len(testData)):
            inputs = np.append(testData[dataNum], 1)    #add bias to initial input
            # Forward pass
            hidden_layer1_output = sigmoid(np.dot(weightsLayerOne, inputs)) #hidden layer 1 output
            
            if numHiddenLayerTwoNeurons > 0:
                hidden_layer1_output = np.append(hidden_layer1_output, 1)   #add bias to hidden layer1 output/hidden layer 2 input
                hidden_layer2_output = sigmoid(np.dot(weightsLayerTwo, hidden_layer1_output))
                hidden_layer2_output = np.append(hidden_layer2_output, 1)   #add bias to hidden layer2 output/output layer input
                output_layer_output = sigmoid(np.dot(weightsLayerOutput, hidden_layer2_output))
            else:
                hidden_layer1_output = np.append(hidden_layer1_output, 1)   #add bias to hidden layer1 output/output layer input
                output_layer_output = sigmoid(np.dot(weightsLayerOutput, hidden_layer1_output))
            
            test_error += calculate_error(targetOutput[int(dataNum + ((20000)*0.7))], output_layer_output)  #compute testing error

            #find if the answer is correct or not
            if classifiedCorrect(targetOutput[int(dataNum + ((20000)*0.7))], output_layer_output) == True:
                test_success += 1

         #save calculations
        test_error_epochs.append([epoch, test_error])

        test_success = test_success / len(testData)

        #write to file
        file.write(f"{epoch}\t{train_error}\t{test_error}\n")
        file1.write(f"{epoch}\t{train_success}\t{test_success}\n")
        # if(epoch%10 == 0):
        #     print(f"Epoch(train) {epoch}: Error = {train_error}, Success rate = {train_success}, Time = {time.time() - start_time}")
        #     print(f"Epoch(test) {epoch}: Error = {test_error}, Success rate = {test_success}, Time = {time.time() - start_time}")


# def printgraphs(a):
#     # Separate the values into x and y values
#     if(a == 0):
#         x_values = [pair[0] for pair in train_error_epochs]
#         y_values = [pair[1] for pair in train_error_epochs]
#         plt.title('Train Error')
#         length = len(train_error_epochs)

#     if(a == 1):
#         x_values = [pair[0] for pair in test_error_epochs]
#         y_values = [pair[1] for pair in test_error_epochs]    
#         plt.title('Test Error')
#         length = len(test_error_epochs)
    
#     # Plot the graph
#     plt.xticks(range(0, length + 30, 30))
#     plt.plot(x_values, y_values)
#     plt.xlabel('X values')
#     plt.ylabel('Y values')
#     plt.grid(True)
#     plt.show()


#name the files
filename_errors = "errors.txt"
filename_successrate = "successrate.txt"
#filename_errors = f"({learningRate},{momentum})[{numHiddenLayerOneNeurons}][{numHiddenLayerTwoNeurons}]errors.txt"
#filename_successrate = f"({learningRate},{momentum})[{numHiddenLayerOneNeurons}][{numHiddenLayerTwoNeurons}]successrate.txt"


#open file to write errors
file = open(filename_errors, 'w')
#open file to write successrate
file1 = open(filename_successrate, 'w')
# Write headers
file.write('Epoch\tTrain Error\tTest Error\n')
file1.write('Epoch\tTrain Successrate\tTest Successrate\n')
# Start training
train()
# printgraphs(0)
# printgraphs(1)
