# Scene Recognition Using Convolutional Neural Network(CNN)

        
        ## Classes:
                - Convolution_Layer
                - Maxpool_Layer
                - FullyConnected_Layer
                - Flattening_Layer
                - ReLu_Activation
                - Softmax_Layer

        ## Functions:
                - 'cross_entropy_loss()': For calculating cross entropy loss   
                - 'training()' : Function for training The model
                - 'testing()': Function for testing The model
                - 'testing_trained_weights()': Function for Testing The model from the trained weights
                - 'predict_img_trained_weights()':Function for Predicting the image with trained weights to which     class it belongs 
                - 'forward_propagation()'- Forward Propagation function for each type of layer which are defined in specific Class
                - 'backward_propagation()'- Backward Propagation function for each type of layer which are defined in specific Class
                - 'zero_pad()' - Zero padding function
                - 'extract_weights()'- Function for extracting weights and biases
                - 'feed_weights()'- Function for feeding or filling weights and biases

        ## Variables.
                - 'train_img' : The training images datasets
                - 'training_labels' : The training labels
                - 'test_imgs' : The testing images datasets
                - 'testing_labels' : The testing labels
                - 'Class_num' : The number of Classes
                - 'batch_size': The training batch size which is taken collectively for forward and backward pass.
                - 'Epochs': The total number of iterations we want to train on.
                - 'CNN':  the Convolutional neurak network Class object
                - 'test_imgs_count'= The Number of testing images to be tested on.

## Running the Codes

Run command `python cnn_main.py` on Anaconda Prompt

In the ```cnn_main.py```, We can modify the epoch, batch size, and testing images count to train the CNN model and testing will also be done on testing datasets. The learning rate can be modified in ```model_network.py```. The folder named "training_datasets" contains the training images and The folder named "testing_datasets" contains the testing images! "cnn_model_weights.pkl" file contains the trained weights and biases after the 3 Epochs for our CNN Model.

## Modified LeNet Training

This Project is implemented on  convolutional layers, fully-connected layers and maxpooling layers  also including backpropagation and gradients descent for training  the Convolutional Neural network(Modified LeNet which was created by famous Scientist Yann Lecun) and cross entropy is used to evaluate the loss.


## Results

* learning rate: 0.01
* batch size: 30
* training accuracy: 0.43
* training accuracy: 0.19
* loss = 1.79



