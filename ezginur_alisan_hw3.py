import csv
import matplotlib.pyplot as plt
import numpy as np
from math import e

trainInputs = []
trainOutputs = []

testFeatures = []
testOutputs = []

MSETrain = [0,0,0,0]
MSETest = [0,0,0,0]

with open(r'C:\Users\ezgia\OneDrive\Masaüstü\CS454 Homework 3\train.csv', newline='') as csvfile:
    dataReader = csv.reader(csvfile, delimiter=',')

    next(dataReader)  # Skip the first row
    for row in dataReader:
        trainInputs.append(float(row[0]))
        trainOutputs.append(float(row[1]))

with open(r'C:\Users\ezgia\OneDrive\Masaüstü\CS454 Homework 3\test.csv', newline='') as csvfile:
    testReader = csv.reader(csvfile, delimiter=',')

    next(testReader)
    for row in testReader:
        testFeatures.append(float(row[0]))
        testOutputs.append(float(row[1]))

#sigmoid function
def sigmoidFunc(x):
  return 1/(1+np.exp(-x))


def singleLayerPerceptron(learningRate, numberofEpochs, trainInputs, trainOutputs, testFeatures, testOutputs):
    weightSingle = np.random.uniform(-1, 1, 1)
    print(weightSingle)
    bias = np.random.uniform(-1, 1, 1)
    outputForSLP = []
    errorForEpoch = []
    for epoch in range(numberofEpochs):
        errorForIterationSLP = 0
        for i in range(len(trainInputs)):
            input = trainInputs[i]
            weight = weightSingle
            output = input * weight + bias                   # xi * wi + bias
            error = trainOutputs[i] - output                 # yAct - yPred
            errorForIterationSLP += (1 / 2) * (error) ** 2   # Error for the one iteration
            weightSingle += learningRate * error * input     # updating the weights
            bias += learningRate * error                     # updating the bias
            if (epoch == numberofEpochs - 1):
                outputForSLP.append(output)
        errorForEpoch.append(errorForIterationSLP)
        if (epoch) % 100 == 0:
            print("SLP Processed", epoch, "iterations. And the error value is : ", errorForIterationSLP)

    array = np.array(outputForSLP).ravel()

    # Fitting a model for the train data
    plt.title("Single Layer Perceptron on Train Data")
    plt.scatter(trainInputs, trainOutputs, color='blue')

    degree = 1
    coeffs = np.polyfit(trainInputs, array, degree)
    poly = np.poly1d(coeffs)
    x_poly = np.linspace(min(trainInputs), max(trainInputs), 100)
    y_poly = poly(x_poly)
    plt.plot(x_poly, y_poly, color='red')
    plt.show()
    # Calculate the predicted y-values
    y_pred = np.polyval(coeffs, trainInputs)

    # Calculate the MSE
    mse = np.sum((trainOutputs - y_pred) ** 2) / len(trainInputs)
    print('MSE single line:', mse)

    # Fitting a model for the test data
    plt.title("Single Layer Perceptron on Test Data")
    plt.scatter(testFeatures, testOutputs, color='blue')

    x_poly = np.linspace(min(testFeatures), max(testFeatures), 100)
    y_poly = poly(x_poly)
    plt.plot(x_poly, y_poly, color='red')
    plt.show()

    # Inference for the train case for Model 1
    totalErrorSquaredTrain = 0
    for i in range(len(trainInputs)):
        tOutput = trainInputs[i] * weightSingle + bias
        totalErrorSquaredTrain += (tOutput - trainOutputs[i]) ** 2
    MSETrainModel = totalErrorSquaredTrain / len(trainInputs)

    # Inference for the test case for Model 1
    totalErrorSquaredTest = 0
    for i in range(len(testFeatures)):
        tOutput = testFeatures[i] * weightSingle + bias
        totalErrorSquaredTest += (tOutput - testOutputs[i])**2
    MSETestModel = totalErrorSquaredTest / (len(testFeatures))
    # print("Inferenced test outputs are : ", testOutputs)

    print("The MSE for the data in Single Layer Perceptron is: ", MSETrainModel)
    print("The MSE for the test in Single Layer Perceptron is: ", MSETestModel)
    MSETrain[0] = MSETrainModel
    MSETest[0] = MSETestModel




def multiLayerPerceptron(hiddenLayer,learningRate, numberofEpochs, trainInputs, trainOutputs, testFeatures, testOutputs):
    weights = np.random.uniform(-1, 1, hiddenLayer)             # input to hidden weights for condition Multi Layer Perceptron 1
    weightsHidden = np.random.uniform(-1, 1, hiddenLayer)       # hidden to output weights for condition Multi Layer Perceptron 1
    bias = np.random.uniform(-1, 1, hiddenLayer)             # bias for input-hidden
    biasHidden = np.random.uniform(-1, 1, 1)                    # bias for hidden-output
    errorForIteration = []
    lastOutputs = []
    for epoch in range(numberofEpochs):
        totalErrorMLP = 0
        for i in range(len(trainInputs)):
            outputMLP = []
            for k in range(hiddenLayer):
                input = trainInputs[i]
                weight = weights[k]  # weights input-hidden layer
                output = input * weight + bias[k]  # output of the hidden layer k
                activation = sigmoidFunc(output)
                outputMLP.append(activation)  # output for the layers
            result = 0
            for k in range(hiddenLayer):
                result += outputMLP[k] * weightsHidden[k]  # Calculating all Zk * Vk combinations
            result += biasHidden  # Adding hidden-output bias
            error = (trainOutputs[i] - result)  # yActual - yPredicted
            errorRate = (1 / 2) * (trainOutputs[i] - result) ** 2  # Error rate
            totalErrorMLP += errorRate
            if (epoch == numberofEpochs - 1):
                lastOutputs.append(result)
            for k in range(hiddenLayer):
                deltaWeightHidden = learningRate * error * outputMLP[k]
                deltaWeight = learningRate * error * weightsHidden[k] * outputMLP[k] * (1 - outputMLP[k]) * trainInputs[i]
                weightsHidden[k] += deltaWeightHidden  # Updating the hidden-output layer's weights
                weights[k] += deltaWeight  # Updating the input-hidden layer's weights
                bias[k] += learningRate * error * outputMLP[k] * (1 - outputMLP[k])  # Updating the input-hidden bias
            biasHidden += learningRate * error  # Updating the hidden-output bias

        errorForIteration.append(totalErrorMLP)
        if (epoch) % 100 == 0:
            print("MLP with hidden layer:",hiddenLayer," Processed", epoch, "iterations. And the error value is : ", totalErrorMLP)

    array = np.array(lastOutputs).ravel()

    # Fitting a model for the train data
    coeffs = np.polyfit(trainInputs, array, 3)
    p = np.poly1d(coeffs)
    # Calculate the predicted y-values
    y_pred = np.polyval(coeffs, trainInputs)

    # Calculate the MSE
    mse = np.sum((trainOutputs - y_pred) ** 2) / len(trainInputs)
    print('MSE Multi curve:', mse)

    # Plot the data and the polynomial fit
    if (hiddenLayer == 10):
        plt.title("Multi Layer Perceptron with 10 hidden units on Train Data")
    elif (hiddenLayer == 20):
        plt.title("Multi Layer Perceptron with 20 hidden units on Train Data")
    elif (hiddenLayer == 50):
        plt.title("Multi Layer Perceptron with 50 hidden units on Train Data")
    plt.scatter(trainInputs, trainOutputs, color='orange')
    x_poly = np.linspace(min(trainInputs), max(trainInputs), 100)
    y_poly = p(x_poly)
    plt.plot(x_poly, y_poly, color='blue')
    plt.show()

    # Fitting a model for the test data
    if (hiddenLayer == 10):
        plt.title("Multi Layer Perceptron with 10 hidden units on Test Data")
    elif (hiddenLayer == 20):
        plt.title("Multi Layer Perceptron with 20 hidden units on Test Data")
    elif (hiddenLayer == 50):
        plt.title("Multi Layer Perceptron with 50 hidden units on Test Data")
    plt.scatter(testFeatures, testOutputs, color='orange')
    x_poly = np.linspace(min(testFeatures), max(testFeatures), 100)
    y_poly = p(x_poly)
    plt.plot(x_poly, y_poly, color='blue')
    plt.show()

    # Inference for the train data
    totalErrorSquaredTrain = 0
    outputTrain = []
    for i in range(len(trainInputs)):
        outputTrain = []
        for k in range(hiddenLayer):
            input = trainInputs[i]
            weight = weights[k]
            biasX = bias[k]
            tOutput = input * weight + biasX
            act1 = sigmoidFunc(tOutput)
            res = act1 * weightsHidden[k]
            outputTrain.append(res)
        result = 0
        for k in range(hiddenLayer):
            result += outputTrain[k]
        result += biasHidden
        totalErrorSquaredTrain += (result - trainOutputs[i]) ** 2

    MSETrainModel = totalErrorSquaredTrain / len(trainInputs)

    # Inference for the test data
    TotalErrorSquaredTest2 = 0
    for i in range(len(testFeatures)):
        outputTest = []
        for k in range(hiddenLayer):
            tOutput = testFeatures[i] * weights[k] + bias[k]
            act1 = sigmoidFunc(tOutput)
            res = act1 * weightsHidden[k]
            outputTest.append(res)
        result = 0
        for k in range(hiddenLayer):
            result += outputTest[k]
        result += biasHidden
        squaredError = (result - testOutputs[i]) ** 2
        TotalErrorSquaredTest2 += squaredError

    MSETestModel = TotalErrorSquaredTest2 / (len(testFeatures))

    # MSE for both train and test case
    print("The MSE for the data in Multi Layer Perceptron with " ,hiddenLayer," is: ", MSETrainModel)
    print("The MSE for the test in Multi Layer Perceptron with " ,hiddenLayer," is: ", MSETestModel)

    if(hiddenLayer == 10):
        MSETrain[1] = MSETrainModel
        MSETest[1] = MSETestModel
    elif(hiddenLayer == 20):
        MSETrain[2] = MSETrainModel
        MSETest[2] = MSETestModel
    elif(hiddenLayer == 50):
        MSETrain[3] = MSETrainModel
        MSETest[3] = MSETestModel



singleLayerPerceptron(0.025,2000,trainInputs,trainOutputs,testFeatures,testOutputs)
multiLayerPerceptron(10,0.025,2000,trainInputs,trainOutputs,testFeatures,testOutputs)
multiLayerPerceptron(20,0.025,2000,trainInputs,trainOutputs,testFeatures,testOutputs)
multiLayerPerceptron(50,0.025,2000,trainInputs,trainOutputs,testFeatures,testOutputs)

numberOfHiddenUnits = [0, 10, 20, 50]

plt.title("Train MSE")
plt.scatter(numberOfHiddenUnits, MSETrain)
plt.plot(numberOfHiddenUnits, MSETrain)
plt.show()

plt.title("Test MSE")
plt.scatter(numberOfHiddenUnits, MSETest)
plt.plot(numberOfHiddenUnits, MSETest)
plt.show()

plt.title("Mean of Squared Errors for Degrees")
plt.xlabel("Degrees")
plt.ylabel("MSE")
plt.scatter(numberOfHiddenUnits, MSETrain)
plt.plot(numberOfHiddenUnits,MSETrain, marker = 'o', color='red', label = "Errors on Training Set")
plt.scatter(numberOfHiddenUnits, MSETest)
plt.plot(numberOfHiddenUnits,MSETest, marker = 'o', label="Errors on Test Set")
plt.legend()
plt.show()