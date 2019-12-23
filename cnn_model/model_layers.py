import numpy as np

# Creating the convolutional layer Class
class Convolution_Layer:

    def __init__(self, inputs_channel, num_filters, kernel_size, padding, stride, learning_rate, name):
        # weight size: (F, C, K, K)
        # bias size: (F) 
        self.Filters = num_filters
        self.Kernels = kernel_size
        self.Channels = inputs_channel

        self.weights = np.zeros((self.Filters, self.Channels, self.Kernels, self.Kernels))
        self.bias = np.zeros((self.Filters, 1))
        # Weights initialization
        for i in range(0,self.Filters):
            self.weights[i,:,:,:] = np.random.normal(loc=0, scale=np.sqrt(1./(self.Channels*self.Kernels*self.Kernels)), size=(self.Channels, self.Kernels, self.Kernels))

        self.pad = padding
        self.ST = stride
        self.LRate = learning_rate
        self.Layer_name = name
        
    # Zero padding function
    def zero_pad(self, inputs, size):
        wid, hei = inputs.shape[0], inputs.shape[1]
        new_wid = 2 * size + wid
        new_hei = 2 * size + hei
        out = np.zeros((new_wid, new_hei))
        out[size:wid+size, size:hei+size] = inputs
        return out
    # Forward Propagation function
    def forward_propagation(self, inputs):
        """
        this forward is doing the convolution and storing the results into the feature maps
        (Calculating the feature maps)
        """
        # input size: (C, W, H)
        # output size: (N, F ,WW, HH)
        Channels = inputs.shape[2]
        Width = inputs.shape[0]+2*self.pad
        Height = inputs.shape[1]+2*self.pad
        self.inputs = np.zeros(( Width, Height, Channels))
        for c in range(inputs.shape[2]):
            self.inputs[:,:, c] = self.zero_pad(inputs[:,:, c], self.pad)
        WW = int((Width - self.Kernels)/self.ST) + 1
        HH = int((Height - self.Kernels)/self.ST) + 1
        # print("width and height are:", WW, HH)
        feature_maps = np.zeros(( WW, HH, self.Filters))
        # print("the shape of the inputs is",self.inputs.shape)
        # print("the shape of the weights is" , self.weights.shape)
        
        # Doing the Convolution
        try:
            for f in range(self.Filters):
                for ch in range(Channels):
                    for w in range(WW):
                        for h in range(HH):
                            feature_maps[w,h,f]=np.sum(self.inputs[w:w + self.Kernels, h:h + self.Kernels, ch]*self.weights[f, ch, :, :])+self.bias[f]
                
                        #print(feature_maps[w,h,f])
        except IndexError:
            print("I m having Index error")
        # print("the feauture map shape is:",feature_maps.shape )
        return feature_maps
    
    # Forward Propagation function
    def backward_propagation(self, dy):

        Width, Height, Channels = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)

        Width, Height, F = dy.shape
        for f in range(F):
            for ch in range(Channels):
                for w in range(Width):
                    for h in range(Height):
                        dw[f,ch,:,:]+=dy[w,h,f]*self.inputs[w:w+self.Kernels,h:h+self.Kernels, ch]
                        dx[w:w+self.Kernels,h:h+self.Kernels, ch]+=dy[w,h,f]*self.weights[f,ch,:,:]

        # print("the shapes for dw, dx, dy are: ",dw.shape, dx.shape, dy.shape)
        for f in range(F):
            db[f] = np.sum(dy[:, :, f])

        self.weights -= self.LRate * dw
        self.bias -= self.LRate * db
#         print("the weights after backptop: ", self.weights)
        return dx
    
    # Function for extracting weights and biases
    def extract_weights(self):
        return {self.Layer_name +'.weights':self.weights, self.Layer_name +'.bias':self.bias}
    
    # Function for feeding or filling weights and biases
    def feed_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
        
# Creating the MAxpooling layer Class
class Maxpool_Layer:

    def __init__(self, pool_size, stride, name):
        self.pool = pool_size
        self.ST = stride
        self.Layer_name = name
    # Forward Propagation function
    def forward_propagation(self, inputs):
        try:
            # print("In the forward pass of maxpooling")
            self.inputs = inputs
            Width, Height, Channels = inputs.shape
            # print("the object types are", type(Channels),type(W),type(H))
            new_width = int((Width - self.pool)/self.ST)+ 1
            new_height = int((Height - self.pool)/self.ST) + 1
            # print("the new width",new_width,new_height)
            
            l1 = int(Width/self.ST)
            l2 = int(Height/self.ST)
 
            # print("the range are ",l1,l2, C)       
            out = np.zeros(( new_width, new_height, Channels))
            # print("In maxpooling, the value of C,W,H,new_width,new_height", C,W,H,new_width,new_height)
            for c in range(Channels):
                for w in range(l1):
                    for h in range(l2):
                        #print("the calculated value is",np.max(self.inputs[c, w*self.ST:w*self.ST+self.pool, h*self.ST:h*self.ST+self.pool]))
                        out[ w, h, c] = np.max(self.inputs[ w*self.ST:w*self.ST+self.pool, h*self.ST:h*self.ST+self.pool, c])
                        #print("the out value is",out[c, w, h])
            # print("the shape after maxpool:",out.shape)
        except:
            print("Error in Maxpooling Layer")
        # print("the shape of max pool:", out.shape)
        return out
    
    # Backward Propagation function
    def backward_propagation(self, dy):
        masking = np.ones_like(self.inputs)*0.25
        return masking*(np.repeat(np.repeat(dy[0:-4, 0:-4 :],2,axis=0),2,axis=1))
    
    # Function for extracting weights and biases
    def extract_weights(self):
        return 
    
# Creating the Fully Connected layer Class   
class FullyConnected_Layer:

    def __init__(self, num_inputs, num_outputs, learning_rate, name):
        self.weights = 0.01*np.random.rand(num_inputs, num_outputs)
        self.bias = np.zeros((num_outputs, 1))
        self.LRate = learning_rate
        self.Layer_name = name
        
    # Forward Propagation function
    def forward_propagation(self, inputs):
        self.inputs = inputs
        return np.dot(self.inputs, self.weights) + self.bias.T
    
    # Backward Propagation function
    def backward_propagation(self, dy):

        if dy.shape[0] == self.inputs.shape[0]:
            dy = dy.T
        dw = dy.dot(self.inputs)
        db = np.sum(dy, axis=1, keepdims=True)
        dx = np.dot(dy.T, self.weights.T)

        self.weights -= self.LRate * dw.T
        self.bias -= self.LRate * db

        return dx
    
    # Function for extracting weights and biases
    def extract_weights(self):
        return {self.Layer_name +'.weights':self.weights, self.Layer_name +'.bias':self.bias}
    
    # Function for feeding or filling weights and biases
    def feed_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
# Creating the Flattening layer Class
class Flattening_Layer:
    def __init__(self):
        pass
    
    # Forward Propagation function
    def forward_propagation(self, inputs):
        self.Width, self.Height, self.Channels, = inputs.shape
        return inputs.reshape(1, self.Channels*self.Width*self.Height)
    
    # Backward Propagation function
    def backward_propagation(self, dy):
        return dy.reshape( self.Width, self.Height, self.Channels)
    
    # Function for extracting weights and biases
    def extract_weights(self):
        return

# Creating the ReLu Activation Class
class ReLu_Activation:
    def __init__(self):
        pass
    
    # Forward Propagation function
    def forward_propagation(self, inputs):
        self.inputs = inputs
        relu = inputs.copy()
        relu[relu < 0] = 0
        return relu
    
    # Backward Propagation function
    def backward_propagation(self, dy):
        dx = dy.copy()
        dx[self.inputs < 0] = 0
        return dx
    
    # Function for extracting weights and biases
    def extract_weights(self):
        return
    
# Creating the Softmax Layer Class
class Softmax_Layer:
    def __init__(self):
        pass
    # Forward Propagation function
    def forward_propagation(self, inputs):
        exp_prob = np.exp(inputs, dtype=np.float)
        self.out = exp_prob/np.sum(exp_prob)
        # print("Completing the forward pass of softmax", self.out.shape)
        # print("Completing the softmax with values", np.unique(self.out).shape)
        return self.out
    
    # Backward Propagation function
    def backward_propagation(self, dy):
        return self.out.T - dy.reshape(dy.shape[1],1)
    
    # Function for extracting weights and biases
    def extract_weights(self):
        return
