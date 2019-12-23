import numpy as np

# calculating cross entropy loss
def cross_entropy_loss(inputs, class_labels):

    output_numb = class_labels.shape[0]
    prob = np.sum(class_labels.reshape(1,output_numb)*inputs)
    entropy_loss = -np.log(prob)
    return entropy_loss