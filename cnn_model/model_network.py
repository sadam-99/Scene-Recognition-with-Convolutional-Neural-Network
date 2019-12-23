import numpy as np
import pickle 
import sys
from time import *
from cnn_model.model_loss import *
from cnn_model.model_layers import *


# Creating the class for building the layers architecture in the CNN Model
class CNN_NET:
    def __init__(self):
        
        # Modified Lenet Architecture
        # input: 100 x 100 x 3
        # convolution 1: (5x5x6)@ sride =1, padding= 2 -> 100x100x6 {(100-5+2x2)/1+1}
        # maxpooling 2: (2x2)@sride = 1, pool_size= 2 -> 50x50x6 {(100-2)/2+1}
        # convolution 3: (5x5x16)@sride = 1, padding=2 -> 50x50x16 {(50-5+2x2)/1+1}
        # maxpooling 4: (2x2)@sride = 1, pool_size = 2 -> 25x25x16 {(50-2)/2+1}
        # convolution 5: (5x5x72)@sride =1, padding = 2 -> 25x25x72 {(5-5)/1+1}
        # Flattening : (1x45000) => {25*25*72 = 45000 by flatenning}
        # fully connected 6: ((1x45000 -> 1x36) 
        # fully connected 7: (1x36) -> (1x6)
        # softmax layer: (1x6) -> (1x6)
        
        Learning_Rate = 0.01
        self.model_layers = []
        #L: 0
        self.model_layers.append(Convolution_Layer(inputs_channel=3, num_filters=6, kernel_size=5, padding=2, stride=1, learning_rate=Learning_Rate, name='conv1'))
        #Ly: 1
        self.model_layers.append(ReLu_Activation())
        #Ly: 2
        self.model_layers.append(Maxpool_Layer(pool_size=2, stride=2, name='maxpool2'))
        #Ly: 3
        self.model_layers.append(Convolution_Layer(inputs_channel=6, num_filters=16, kernel_size=5, padding=2, stride=1, learning_rate=Learning_Rate, name='conv3'))
        #Ly: 4
        self.model_layers.append(ReLu_Activation())
        #Ly: 5
        self.model_layers.append(Maxpool_Layer(pool_size=2, stride=2, name='maxpool4'))
        #Ly: 6
        self.model_layers.append(Convolution_Layer(inputs_channel=16, num_filters=72, kernel_size=5, padding=2, stride=1, learning_rate=Learning_Rate, name='conv5'))
        #Ly: 7
        self.model_layers.append(ReLu_Activation())
        #Ly: 8
        self.model_layers.append(Flattening_Layer())
        #Ly: 9
        self.model_layers.append(FullyConnected_Layer(num_inputs=45000, num_outputs=36, learning_rate=Learning_Rate, name='fc6'))
        #Ly: 10
        self.model_layers.append(ReLu_Activation())
        #Ly: 11
        self.model_layers.append(FullyConnected_Layer(num_inputs=36, num_outputs=6, learning_rate=Learning_Rate, name='fc7'))
        #Ly: 12
        self.model_layers.append(Softmax_Layer())
        self.layers_numb = len(self.model_layers)
        
    
    # Function for training The model
    def training(self, training_data, training_label, batch_size, epoch, weights_file):
        total_accuracy = 0
        for e in range(epoch):
            for batch_index in range(0, training_data.shape[0], batch_size):
                # batch input for loading the training input data
                if batch_index + batch_size < training_data.shape[0]:
                    train_data = training_data[batch_index:batch_index+batch_size]
                    train_labels = training_label[batch_index:batch_index + batch_size]
                else:
                    train_data = training_data[batch_index:training_data.shape[0]]
                    train_labels = training_label[batch_index:training_label.shape[0]]
                model_loss = 0
                accuracy = 0
                start_time = time()
                for b in range(batch_size):
                    train_img = train_data[b]
                    train_label = train_labels[b]
                    # forward pass
                    for lay in range(self.layers_numb):
                        # print("Working on forward pass for layer no.", l)
                        output = self.model_layers[lay].forward_propagation(train_img)
                        # print("the shape output ater iteration is :",l, output.shape)
                        train_img = output
#                     print("output shape:", output.shape)
                    model_loss += cross_entropy_loss(output, train_label)
                    if np.argmax(output) == np.argmax(train_label):
                        accuracy += 1
                        total_accuracy += 1
                    # print("output is:", output, output.shape, np.argmax(output), np.argmax(y) )
                    # backward pass
                    # print("The model_loss and accuracy is", model_loss, total_accuracy)
                    dy = output
                    for lay in range(self.layers_numb-1, -1, -1):
                        # print("Working on backward pass for layer no.", l)
                        dout = self.model_layers[lay].backward_propagation(dy)
                        dy = dout
                # time
                end_time = time()
                batch_time = end_time-start_time
                remaining_time = (training_data.shape[0]*epoch-batch_index-training_data.shape[0]*e)/batch_size*batch_time
                hrs = int(remaining_time)/3600
                mins = int((remaining_time/60-hrs*60))
                secs = int(remaining_time-mins*60-hrs*3600)
                # Calculating the model loss and accuracy.
                model_loss /= batch_size
                batch_accuracy = float(accuracy)/float(batch_size)
                training_accuracy = float(total_accuracy)/float((batch_index+batch_size)*(e+1))
                print('=== Epoch: {0:d}/{1:d} === Iter:{2:d} === model_loss: {3:.2f} === BAcc: {4:.2f} === TAcc: {5:.2f} === Remain: {6:d} Hrs {7:d} Mins {8:d} Secs ==='.format(e,epoch,batch_index+batch_size,model_loss,batch_accuracy,training_accuracy,int(hrs),int(mins),int(secs)))
        # dump weights and biases into the pickle file after each epoch.
            weights = []
            for i in range(self.layers_numb):
                wt = self.model_layers[i].extract_weights()
                weights.append(wt)
            with open(weights_file, 'ab') as pick_file:
                pickle.dump(weights, pick_file, protocol=pickle.HIGHEST_PROTOCOL)
                
    
    # Function for Testing  The model
    def testing(self, test_data, test_label, test_size):
        total_accuracy = 0
        for i in range(test_size):
            x0 = test_data[i]
            y0 = test_label[i]
            for lay in range(self.layers_numb):
                output = self.model_layers[lay].forward_propagation(x0)
                x0 = output
            # Calculate the testing accuracy
            if np.argmax(output) == np.argmax(y0):
                total_accuracy += 1
        sys.stdout.write("\n")
        print('=== Test Size:{0:d} === Test accuracy:{1:.2f} ==='.format(test_size, float(total_accuracy)/float(test_size)))
        
    
    # Function for Testing The model from the trained weights
    def testing_trained_weights(self, test_data, test_label, test_size, weights_file):
        with open(weights_file, 'rb') as pick_file:
            wt = pickle.load(pick_file)
        
        # Load the trained weights and biases for testing
        self.model_layers[0].feed_weights(wt[0]['conv1.weights'], wt[0]['conv1.bias'])
        self.model_layers[3].feed_weights(wt[3]['conv3.weights'], wt[3]['conv3.bias'])
        self.model_layers[6].feed_weights(wt[6]['conv5.weights'], wt[6]['conv5.bias'])
        self.model_layers[9].feed_weights(wt[9]['fc6.weights'], wt[9]['fc6.bias'])
        self.model_layers[11].feed_weights(wt[11]['fc7.weights'], wt[11]['fc7.bias'])
        # Calculate the testing accuracy
        total_accuracy = 0
        for i in range(test_size):
            x = test_data[i]
            y = test_label[i]
            for l in range(self.layers_numb):
                output = self.model_layers[l].forward_propagation(x)
                x = output
            if np.argmax(output) == np.argmax(y):
                total_accuracy += 1
        sys.stdout.write("\n")
        print('=== Test Size:{0:d} === Test accuracy:{1:.2f} ==='.format(test_size, float(total_accuracy)/float(test_size)))
        
    # Function for Predicting the image with trained weights to which class it belongs       
    def predict_img_trained_weights(self, inputs, weights_file):
        with open(weights_file, 'rb') as pick_file:
            wt = pickle.load(pick_file)
        # Load the trained weights and biases for Prediction
        self.model_layers[0].feed_weights(wt[0]['conv1.weights'], wt[0]['conv1.bias'])
        self.model_layers[3].feed_weights(wt[3]['conv3.weights'], wt[3]['conv3.bias'])
        self.model_layers[6].feed_weights(wt[6]['conv5.weights'], wt[6]['conv5.bias'])
        self.model_layers[9].feed_weights(wt[9]['fc6.weights'], wt[9]['fc6.bias'])
        self.model_layers[11].feed_weights(wt[11]['fc7.weights'], wt[11]['fc7.bias'])
        for l in range(self.layers_numb):
            output = self.model_layers[l].forward_propagation(inputs)
            inputs = output
        Scene_Class = np.argmax(output)
        # Calculate the probabilities of each Class
        probability = output[0, Scene_Class]
        return Scene_Class, probability

