# Converted from cnn.ipynb

# ======================================================================
# # Convolution Neural Network
# ======================================================================

# ======================================================================
# - A Convelutional neural network (convNet/CNN) is deep learning algorithm which takes in an inputs image, assign importance (learnable weigths and biases) to various aspects/objects in the image and be able to differetiate one from the other. 
# - The pre-processing required in a convNet is much lower as compared to other classification algorithms.
# - While in primitive methods filters are hand-engineered, with enough training, convNets have the ability to learn these filters/characteristics.
# ======================================================================

# ======================================================================
# ![convolution layer.png](https://www.kaggleusercontent.com/kf/99705616/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mS32FYLYwSsMSYO5vJAuZg.cXnDOwrGdtNgp8rFz2msenOptZMX1EkbxHcpivlCbaZ8bCZGfGkQWCS1XxhjMEFc_tfDkA-vchSRIrVp0_Det0nPjIY2ULmSaJl4K1qh4dgnYITyJKh5aLqu6e-cTzYTI3Z_9VH3RKrFoHRSgtRFOX5nHhD0W0N1twKKCXt9FOtoaPtILmefWCaj5jLwilnPc51SLU4YV-fhMkzPsBbXPS-NwCOlVGh9ro4D9BqmVeh2CLnfr4ob9C5oshfYqSUrj59HjhF596oM6HLqMv3Qvo1aee5YAWsQQaIHomc78ET9Pc08Cq_cgH-HYpMpxdBQSHTaTqexErxZZ3e4ChSZT5ehUQCX7pInt0ShsLOIO2VVDQUHzVx1Q1iGaZCQXpfY1xIFQpEw7MlW-z7J0RRFvaiscyyT6rZKdpGxQJBi2OReFAt7ubyRFEfMJ-MCzgkiOV9xtN84smPA5NQqCS16TVnQeBAUS79fkbpGXm2SbjSrtZgsiYVHmH6KFxrBDj8_W-Wg8p1u9sbZ-ry0ZjFRn4BUDy5XgK-wv-Lv749mWggYyGFRoO3FALi8DLCz7wdEVow-MndI3C3bdCUa2eUSfKctOCQntGoh6JvYEvVNRi2aACQJ5xHdyOOWEj84QCUOJd8vSSRoPZDMW6ycYZA9dYDOylHhwZlpPvl313VQeKA.ai3Ue7GYk7OizC8bppsMUA/__results___files/__results___45_0.png)
# ======================================================================

# ======================================================================
# ![con2.img](https://www.kaggleusercontent.com/kf/99705616/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mS32FYLYwSsMSYO5vJAuZg.cXnDOwrGdtNgp8rFz2msenOptZMX1EkbxHcpivlCbaZ8bCZGfGkQWCS1XxhjMEFc_tfDkA-vchSRIrVp0_Det0nPjIY2ULmSaJl4K1qh4dgnYITyJKh5aLqu6e-cTzYTI3Z_9VH3RKrFoHRSgtRFOX5nHhD0W0N1twKKCXt9FOtoaPtILmefWCaj5jLwilnPc51SLU4YV-fhMkzPsBbXPS-NwCOlVGh9ro4D9BqmVeh2CLnfr4ob9C5oshfYqSUrj59HjhF596oM6HLqMv3Qvo1aee5YAWsQQaIHomc78ET9Pc08Cq_cgH-HYpMpxdBQSHTaTqexErxZZ3e4ChSZT5ehUQCX7pInt0ShsLOIO2VVDQUHzVx1Q1iGaZCQXpfY1xIFQpEw7MlW-z7J0RRFvaiscyyT6rZKdpGxQJBi2OReFAt7ubyRFEfMJ-MCzgkiOV9xtN84smPA5NQqCS16TVnQeBAUS79fkbpGXm2SbjSrtZgsiYVHmH6KFxrBDj8_W-Wg8p1u9sbZ-ry0ZjFRn4BUDy5XgK-wv-Lv749mWggYyGFRoO3FALi8DLCz7wdEVow-MndI3C3bdCUa2eUSfKctOCQntGoh6JvYEvVNRi2aACQJ5xHdyOOWEj84QCUOJd8vSSRoPZDMW6ycYZA9dYDOylHhwZlpPvl313VQeKA.ai3Ue7GYk7OizC8bppsMUA/__results___files/__results___46_0.png)
# ======================================================================

# %%
import numpy as np
from sklearn import datasets

# %%
class Conv:

    def __init__(self, num_filters):
        self.num_filters = num_filters
        
        # why diviide by 9...Xavier initialization
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        #generate all possible 3*3 image regions using valid padding

        h, w = image.shape

        for i in range(h-2):
            for j in range(w-2):
                im_region = image[i: (i+3), j:(j+3)]
                yield im_region, i, j
    
    def forward(self, input):
        self.last_input = input

        h,w = input.shape

        output = np.zeros((h-2, w-2, self.num_filters))

        for im_regions, i ,j in self.iterate_regions(input):
            output[i, j] = np.sum(im_regions * self.filters, axis=(1,2))
        
        return output

    def backprop(self, d_l_d_out, learn_rate):
        '''
        performs a backward pas of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        d_l_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_l_d_filters[f] += d_l_d_out[i, j ,f] * im_region

            #update filters
            self.filters -= learn_rate * d_l_d_filters

            return None

# ======================================================================
# # Maxpooling layer
# ======================================================================

# ======================================================================
# - A max pooling layer can't be trained it doesn't actually have any weights, but we still need  to implement a backprop() method for it to calculate gradients. We’ll start by adding forward phase caching again. All we need to cache this time is the input:
# 
# - During the forward pass, the Max Pooling layer takes an input volume and halves its width and height dimensions by picking the max values over 2x2 blocks. The backward pass does the opposite: we’ll double the width and height of the loss gradient by assigning each gradient value to where the original max value was in its corresponding 2x2 block.
# ======================================================================

# ======================================================================
# ![maxpool.png](https://www.kaggleusercontent.com/kf/99705616/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mS32FYLYwSsMSYO5vJAuZg.cXnDOwrGdtNgp8rFz2msenOptZMX1EkbxHcpivlCbaZ8bCZGfGkQWCS1XxhjMEFc_tfDkA-vchSRIrVp0_Det0nPjIY2ULmSaJl4K1qh4dgnYITyJKh5aLqu6e-cTzYTI3Z_9VH3RKrFoHRSgtRFOX5nHhD0W0N1twKKCXt9FOtoaPtILmefWCaj5jLwilnPc51SLU4YV-fhMkzPsBbXPS-NwCOlVGh9ro4D9BqmVeh2CLnfr4ob9C5oshfYqSUrj59HjhF596oM6HLqMv3Qvo1aee5YAWsQQaIHomc78ET9Pc08Cq_cgH-HYpMpxdBQSHTaTqexErxZZ3e4ChSZT5ehUQCX7pInt0ShsLOIO2VVDQUHzVx1Q1iGaZCQXpfY1xIFQpEw7MlW-z7J0RRFvaiscyyT6rZKdpGxQJBi2OReFAt7ubyRFEfMJ-MCzgkiOV9xtN84smPA5NQqCS16TVnQeBAUS79fkbpGXm2SbjSrtZgsiYVHmH6KFxrBDj8_W-Wg8p1u9sbZ-ry0ZjFRn4BUDy5XgK-wv-Lv749mWggYyGFRoO3FALi8DLCz7wdEVow-MndI3C3bdCUa2eUSfKctOCQntGoh6JvYEvVNRi2aACQJ5xHdyOOWEj84QCUOJd8vSSRoPZDMW6ycYZA9dYDOylHhwZlpPvl313VQeKA.ai3Ue7GYk7OizC8bppsMUA/__results___files/__results___49_0.png)
# ======================================================================

# ======================================================================
# ![example.png](https://www.kaggleusercontent.com/kf/99705616/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mS32FYLYwSsMSYO5vJAuZg.cXnDOwrGdtNgp8rFz2msenOptZMX1EkbxHcpivlCbaZ8bCZGfGkQWCS1XxhjMEFc_tfDkA-vchSRIrVp0_Det0nPjIY2ULmSaJl4K1qh4dgnYITyJKh5aLqu6e-cTzYTI3Z_9VH3RKrFoHRSgtRFOX5nHhD0W0N1twKKCXt9FOtoaPtILmefWCaj5jLwilnPc51SLU4YV-fhMkzPsBbXPS-NwCOlVGh9ro4D9BqmVeh2CLnfr4ob9C5oshfYqSUrj59HjhF596oM6HLqMv3Qvo1aee5YAWsQQaIHomc78ET9Pc08Cq_cgH-HYpMpxdBQSHTaTqexErxZZ3e4ChSZT5ehUQCX7pInt0ShsLOIO2VVDQUHzVx1Q1iGaZCQXpfY1xIFQpEw7MlW-z7J0RRFvaiscyyT6rZKdpGxQJBi2OReFAt7ubyRFEfMJ-MCzgkiOV9xtN84smPA5NQqCS16TVnQeBAUS79fkbpGXm2SbjSrtZgsiYVHmH6KFxrBDj8_W-Wg8p1u9sbZ-ry0ZjFRn4BUDy5XgK-wv-Lv749mWggYyGFRoO3FALi8DLCz7wdEVow-MndI3C3bdCUa2eUSfKctOCQntGoh6JvYEvVNRi2aACQJ5xHdyOOWEj84QCUOJd8vSSRoPZDMW6ycYZA9dYDOylHhwZlpPvl313VQeKA.ai3Ue7GYk7OizC8bppsMUA/__results___files/__results___50_0.png)
# ======================================================================

# %%
class MaxPool:
    def iterate_regions(self, image):
        h , w , _ = image.shape

        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2): (i*2+2), (j*2): (j*2+2)]
                yield im_region , i ,j

    
    def forward(self, input):
        
        self.last_input = input
        
        h, w, num_filters = input.shape
        output = np.zeros((h//2, w//2, num_filters))
        
        for im_region, i, j in self.iterate_regions(input):
            output[i,j] = np.amax(im_region,axis=(0,1))
            
        return output
    
    def backprop(self, d_l_d_out):
        '''
        Performs a backward pass of the maxpool layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        '''
        d_l_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0,1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        #if the pixel was the max value, copy the gradient to it
                        if(im_region[i2,j2,f2] == amax[f2]):
                            d_l_d_input[i*2+i2, j*2+j2 ,f2] = d_l_d_out[i, j, f2]
                            break;
        return d_l_d_input

# ======================================================================
# # softmax layer
# ======================================================================

# ======================================================================
# ![softmax layer](https://www.kaggleusercontent.com/kf/99705616/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mS32FYLYwSsMSYO5vJAuZg.cXnDOwrGdtNgp8rFz2msenOptZMX1EkbxHcpivlCbaZ8bCZGfGkQWCS1XxhjMEFc_tfDkA-vchSRIrVp0_Det0nPjIY2ULmSaJl4K1qh4dgnYITyJKh5aLqu6e-cTzYTI3Z_9VH3RKrFoHRSgtRFOX5nHhD0W0N1twKKCXt9FOtoaPtILmefWCaj5jLwilnPc51SLU4YV-fhMkzPsBbXPS-NwCOlVGh9ro4D9BqmVeh2CLnfr4ob9C5oshfYqSUrj59HjhF596oM6HLqMv3Qvo1aee5YAWsQQaIHomc78ET9Pc08Cq_cgH-HYpMpxdBQSHTaTqexErxZZ3e4ChSZT5ehUQCX7pInt0ShsLOIO2VVDQUHzVx1Q1iGaZCQXpfY1xIFQpEw7MlW-z7J0RRFvaiscyyT6rZKdpGxQJBi2OReFAt7ubyRFEfMJ-MCzgkiOV9xtN84smPA5NQqCS16TVnQeBAUS79fkbpGXm2SbjSrtZgsiYVHmH6KFxrBDj8_W-Wg8p1u9sbZ-ry0ZjFRn4BUDy5XgK-wv-Lv749mWggYyGFRoO3FALi8DLCz7wdEVow-MndI3C3bdCUa2eUSfKctOCQntGoh6JvYEvVNRi2aACQJ5xHdyOOWEj84QCUOJd8vSSRoPZDMW6ycYZA9dYDOylHhwZlpPvl313VQeKA.ai3Ue7GYk7OizC8bppsMUA/__results___files/__results___53_0.png)
# ======================================================================

# ======================================================================
# ![gradient](https://www.kaggleusercontent.com/kf/99705616/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mS32FYLYwSsMSYO5vJAuZg.cXnDOwrGdtNgp8rFz2msenOptZMX1EkbxHcpivlCbaZ8bCZGfGkQWCS1XxhjMEFc_tfDkA-vchSRIrVp0_Det0nPjIY2ULmSaJl4K1qh4dgnYITyJKh5aLqu6e-cTzYTI3Z_9VH3RKrFoHRSgtRFOX5nHhD0W0N1twKKCXt9FOtoaPtILmefWCaj5jLwilnPc51SLU4YV-fhMkzPsBbXPS-NwCOlVGh9ro4D9BqmVeh2CLnfr4ob9C5oshfYqSUrj59HjhF596oM6HLqMv3Qvo1aee5YAWsQQaIHomc78ET9Pc08Cq_cgH-HYpMpxdBQSHTaTqexErxZZ3e4ChSZT5ehUQCX7pInt0ShsLOIO2VVDQUHzVx1Q1iGaZCQXpfY1xIFQpEw7MlW-z7J0RRFvaiscyyT6rZKdpGxQJBi2OReFAt7ubyRFEfMJ-MCzgkiOV9xtN84smPA5NQqCS16TVnQeBAUS79fkbpGXm2SbjSrtZgsiYVHmH6KFxrBDj8_W-Wg8p1u9sbZ-ry0ZjFRn4BUDy5XgK-wv-Lv749mWggYyGFRoO3FALi8DLCz7wdEVow-MndI3C3bdCUa2eUSfKctOCQntGoh6JvYEvVNRi2aACQJ5xHdyOOWEj84QCUOJd8vSSRoPZDMW6ycYZA9dYDOylHhwZlpPvl313VQeKA.ai3Ue7GYk7OizC8bppsMUA/__results___files/__results___54_0.png)
# ======================================================================

# %%
class Softmax:
    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        self.weights = np.random.randn(input_len, nodes)/input_len
        self.biases = np.zeros(nodes)
    
    def forward(self, input):
        
        self.last_input_shape = input.shape
        
        input = input.flatten()
        self.last_input = input
        
        input_len, nodes = self.weights.shape
        
        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals
        
        exp = np.exp(totals)
        return(exp/np.sum(exp, axis=0)) 
    
    def backprop(self, d_l_d_out, learn_rate):
        """  
        Performs a backward pass of the softmax layer.
        Returns the loss gradient for this layers inputs.
        - d_L_d_out is the loss gradient for this layers outputs.
        """
        
        #We know only 1 element of d_l_d_out will be nonzero
        for i, gradient in enumerate(d_l_d_out):
            if(gradient == 0):
                continue
            
            #e^totals
            t_exp = np.exp(self.last_totals)
            
            #Sum of all e^totals
            S = np.sum(t_exp)
            
            #gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp/ (S**2)
            d_out_d_t[i] = t_exp[i] * (S-t_exp[i]) /(S**2)
            
            # Gradients of totals against weights/biases/input
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights
            
            #Gradients of loss against totals
            d_l_d_t = gradient * d_out_d_t
            
            #Gradients of loss against weights/biases/input
            d_l_d_w = d_t_d_w[np.newaxis].T @ d_l_d_t[np.newaxis]
            d_l_d_b = d_l_d_t * d_t_d_b  
            d_l_d_inputs = d_t_d_inputs @ d_l_d_t
            
            #update weights/biases
            self.weights -= learn_rate * d_l_d_w
            self.biases -= learn_rate * d_l_d_b
            return d_l_d_inputs.reshape(self.last_input_shape)

# %%
import torch
from torchvision import datasets, transforms

# Download and load the MNIST dataset using PyTorch
transform = transforms.Compose([
    transforms.ToTensor(),
])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Get numpy arrays like the original API, normalizing to 0-255 range
train_images = mnist_train.data[:1000].numpy()
train_labels = mnist_train.targets[:1000].numpy()
test_images = mnist_test.data[:1000].numpy()
test_labels = mnist_test.targets[:1000].numpy()

# ======================================================================
# ![cnn-train.png](https://www.kaggleusercontent.com/kf/99705616/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mS32FYLYwSsMSYO5vJAuZg.cXnDOwrGdtNgp8rFz2msenOptZMX1EkbxHcpivlCbaZ8bCZGfGkQWCS1XxhjMEFc_tfDkA-vchSRIrVp0_Det0nPjIY2ULmSaJl4K1qh4dgnYITyJKh5aLqu6e-cTzYTI3Z_9VH3RKrFoHRSgtRFOX5nHhD0W0N1twKKCXt9FOtoaPtILmefWCaj5jLwilnPc51SLU4YV-fhMkzPsBbXPS-NwCOlVGh9ro4D9BqmVeh2CLnfr4ob9C5oshfYqSUrj59HjhF596oM6HLqMv3Qvo1aee5YAWsQQaIHomc78ET9Pc08Cq_cgH-HYpMpxdBQSHTaTqexErxZZ3e4ChSZT5ehUQCX7pInt0ShsLOIO2VVDQUHzVx1Q1iGaZCQXpfY1xIFQpEw7MlW-z7J0RRFvaiscyyT6rZKdpGxQJBi2OReFAt7ubyRFEfMJ-MCzgkiOV9xtN84smPA5NQqCS16TVnQeBAUS79fkbpGXm2SbjSrtZgsiYVHmH6KFxrBDj8_W-Wg8p1u9sbZ-ry0ZjFRn4BUDy5XgK-wv-Lv749mWggYyGFRoO3FALi8DLCz7wdEVow-MndI3C3bdCUa2eUSfKctOCQntGoh6JvYEvVNRi2aACQJ5xHdyOOWEj84QCUOJd8vSSRoPZDMW6ycYZA9dYDOylHhwZlpPvl313VQeKA.ai3Ue7GYk7OizC8bppsMUA/__results___files/__results___58_0.png)
# ======================================================================

# %%
conv = Conv(8)
pool = MaxPool()
softmax = Softmax(13 * 13 * 8, 10)

def forward(image, label):
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.
    
    out = conv.forward((image/255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)
    
    #calculate cross-entropy loss and accuracy
    loss = -np.log(out[label])
    acc = 1 if(np.argmax(out) == label) else 0
    
    return out, loss, acc


def train(im, label, lr=0.005):
    #forward
    out,loss,acc = forward(im, label)
    
    #calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1/out[label]
    
    
    #Backprop
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)
    
    return loss, acc
    
    
print('MNIST CNN initialized')

for epoch in range(3):
    print('----EPOCH %d ---'%(epoch+1))
    
    #shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]


    loss = 0
    num_correct = 0

    for i, (im, label) in enumerate(zip(train_images, train_labels)):

        #print stats every 100 steps
        if(i>0 and i %100 == 99):
            print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %(i + 1, loss / 100, num_correct))

            # Reset loss and num_correct counters after every 100 steps to correctly track average loss and accuracy for each reporting interval.
            # This avoids counters accumulating over the entire epoch, providing accurate per-interval statistics.
            loss = 0
            num_correct = 0
        l, acc = train(im, label)
        loss += l
        num_correct += acc

# ======================================================================
# ![backpropagation-indepth.png](https://www.kaggleusercontent.com/kf/99705616/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mS32FYLYwSsMSYO5vJAuZg.cXnDOwrGdtNgp8rFz2msenOptZMX1EkbxHcpivlCbaZ8bCZGfGkQWCS1XxhjMEFc_tfDkA-vchSRIrVp0_Det0nPjIY2ULmSaJl4K1qh4dgnYITyJKh5aLqu6e-cTzYTI3Z_9VH3RKrFoHRSgtRFOX5nHhD0W0N1twKKCXt9FOtoaPtILmefWCaj5jLwilnPc51SLU4YV-fhMkzPsBbXPS-NwCOlVGh9ro4D9BqmVeh2CLnfr4ob9C5oshfYqSUrj59HjhF596oM6HLqMv3Qvo1aee5YAWsQQaIHomc78ET9Pc08Cq_cgH-HYpMpxdBQSHTaTqexErxZZ3e4ChSZT5ehUQCX7pInt0ShsLOIO2VVDQUHzVx1Q1iGaZCQXpfY1xIFQpEw7MlW-z7J0RRFvaiscyyT6rZKdpGxQJBi2OReFAt7ubyRFEfMJ-MCzgkiOV9xtN84smPA5NQqCS16TVnQeBAUS79fkbpGXm2SbjSrtZgsiYVHmH6KFxrBDj8_W-Wg8p1u9sbZ-ry0ZjFRn4BUDy5XgK-wv-Lv749mWggYyGFRoO3FALi8DLCz7wdEVow-MndI3C3bdCUa2eUSfKctOCQntGoh6JvYEvVNRi2aACQJ5xHdyOOWEj84QCUOJd8vSSRoPZDMW6ycYZA9dYDOylHhwZlpPvl313VQeKA.ai3Ue7GYk7OizC8bppsMUA/__results___files/__results___61_0.png)
# ======================================================================

# ======================================================================
# ![backpropagation-indepth.png](https://www.kaggleusercontent.com/kf/99705616/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mS32FYLYwSsMSYO5vJAuZg.cXnDOwrGdtNgp8rFz2msenOptZMX1EkbxHcpivlCbaZ8bCZGfGkQWCS1XxhjMEFc_tfDkA-vchSRIrVp0_Det0nPjIY2ULmSaJl4K1qh4dgnYITyJKh5aLqu6e-cTzYTI3Z_9VH3RKrFoHRSgtRFOX5nHhD0W0N1twKKCXt9FOtoaPtILmefWCaj5jLwilnPc51SLU4YV-fhMkzPsBbXPS-NwCOlVGh9ro4D9BqmVeh2CLnfr4ob9C5oshfYqSUrj59HjhF596oM6HLqMv3Qvo1aee5YAWsQQaIHomc78ET9Pc08Cq_cgH-HYpMpxdBQSHTaTqexErxZZ3e4ChSZT5ehUQCX7pInt0ShsLOIO2VVDQUHzVx1Q1iGaZCQXpfY1xIFQpEw7MlW-z7J0RRFvaiscyyT6rZKdpGxQJBi2OReFAt7ubyRFEfMJ-MCzgkiOV9xtN84smPA5NQqCS16TVnQeBAUS79fkbpGXm2SbjSrtZgsiYVHmH6KFxrBDj8_W-Wg8p1u9sbZ-ry0ZjFRn4BUDy5XgK-wv-Lv749mWggYyGFRoO3FALi8DLCz7wdEVow-MndI3C3bdCUa2eUSfKctOCQntGoh6JvYEvVNRi2aACQJ5xHdyOOWEj84QCUOJd8vSSRoPZDMW6ycYZA9dYDOylHhwZlpPvl313VQeKA.ai3Ue7GYk7OizC8bppsMUA/__results___files/__results___62_0.png)
# ======================================================================

# ======================================================================
# ![backpropagation-indepth](https://www.kaggleusercontent.com/kf/99705616/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mS32FYLYwSsMSYO5vJAuZg.cXnDOwrGdtNgp8rFz2msenOptZMX1EkbxHcpivlCbaZ8bCZGfGkQWCS1XxhjMEFc_tfDkA-vchSRIrVp0_Det0nPjIY2ULmSaJl4K1qh4dgnYITyJKh5aLqu6e-cTzYTI3Z_9VH3RKrFoHRSgtRFOX5nHhD0W0N1twKKCXt9FOtoaPtILmefWCaj5jLwilnPc51SLU4YV-fhMkzPsBbXPS-NwCOlVGh9ro4D9BqmVeh2CLnfr4ob9C5oshfYqSUrj59HjhF596oM6HLqMv3Qvo1aee5YAWsQQaIHomc78ET9Pc08Cq_cgH-HYpMpxdBQSHTaTqexErxZZ3e4ChSZT5ehUQCX7pInt0ShsLOIO2VVDQUHzVx1Q1iGaZCQXpfY1xIFQpEw7MlW-z7J0RRFvaiscyyT6rZKdpGxQJBi2OReFAt7ubyRFEfMJ-MCzgkiOV9xtN84smPA5NQqCS16TVnQeBAUS79fkbpGXm2SbjSrtZgsiYVHmH6KFxrBDj8_W-Wg8p1u9sbZ-ry0ZjFRn4BUDy5XgK-wv-Lv749mWggYyGFRoO3FALi8DLCz7wdEVow-MndI3C3bdCUa2eUSfKctOCQntGoh6JvYEvVNRi2aACQJ5xHdyOOWEj84QCUOJd8vSSRoPZDMW6ycYZA9dYDOylHhwZlpPvl313VQeKA.ai3Ue7GYk7OizC8bppsMUA/__results___files/__results___63_0.png)
# ======================================================================

# ======================================================================
# ![backpropagation-indepth.png](https://www.kaggleusercontent.com/kf/99705616/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mS32FYLYwSsMSYO5vJAuZg.cXnDOwrGdtNgp8rFz2msenOptZMX1EkbxHcpivlCbaZ8bCZGfGkQWCS1XxhjMEFc_tfDkA-vchSRIrVp0_Det0nPjIY2ULmSaJl4K1qh4dgnYITyJKh5aLqu6e-cTzYTI3Z_9VH3RKrFoHRSgtRFOX5nHhD0W0N1twKKCXt9FOtoaPtILmefWCaj5jLwilnPc51SLU4YV-fhMkzPsBbXPS-NwCOlVGh9ro4D9BqmVeh2CLnfr4ob9C5oshfYqSUrj59HjhF596oM6HLqMv3Qvo1aee5YAWsQQaIHomc78ET9Pc08Cq_cgH-HYpMpxdBQSHTaTqexErxZZ3e4ChSZT5ehUQCX7pInt0ShsLOIO2VVDQUHzVx1Q1iGaZCQXpfY1xIFQpEw7MlW-z7J0RRFvaiscyyT6rZKdpGxQJBi2OReFAt7ubyRFEfMJ-MCzgkiOV9xtN84smPA5NQqCS16TVnQeBAUS79fkbpGXm2SbjSrtZgsiYVHmH6KFxrBDj8_W-Wg8p1u9sbZ-ry0ZjFRn4BUDy5XgK-wv-Lv749mWggYyGFRoO3FALi8DLCz7wdEVow-MndI3C3bdCUa2eUSfKctOCQntGoh6JvYEvVNRi2aACQJ5xHdyOOWEj84QCUOJd8vSSRoPZDMW6ycYZA9dYDOylHhwZlpPvl313VQeKA.ai3Ue7GYk7OizC8bppsMUA/__results___files/__results___64_0.png)
# ======================================================================

# ======================================================================
# ![backpropagation-indepth.png](https://www.kaggleusercontent.com/kf/99705616/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..mS32FYLYwSsMSYO5vJAuZg.cXnDOwrGdtNgp8rFz2msenOptZMX1EkbxHcpivlCbaZ8bCZGfGkQWCS1XxhjMEFc_tfDkA-vchSRIrVp0_Det0nPjIY2ULmSaJl4K1qh4dgnYITyJKh5aLqu6e-cTzYTI3Z_9VH3RKrFoHRSgtRFOX5nHhD0W0N1twKKCXt9FOtoaPtILmefWCaj5jLwilnPc51SLU4YV-fhMkzPsBbXPS-NwCOlVGh9ro4D9BqmVeh2CLnfr4ob9C5oshfYqSUrj59HjhF596oM6HLqMv3Qvo1aee5YAWsQQaIHomc78ET9Pc08Cq_cgH-HYpMpxdBQSHTaTqexErxZZ3e4ChSZT5ehUQCX7pInt0ShsLOIO2VVDQUHzVx1Q1iGaZCQXpfY1xIFQpEw7MlW-z7J0RRFvaiscyyT6rZKdpGxQJBi2OReFAt7ubyRFEfMJ-MCzgkiOV9xtN84smPA5NQqCS16TVnQeBAUS79fkbpGXm2SbjSrtZgsiYVHmH6KFxrBDj8_W-Wg8p1u9sbZ-ry0ZjFRn4BUDy5XgK-wv-Lv749mWggYyGFRoO3FALi8DLCz7wdEVow-MndI3C3bdCUa2eUSfKctOCQntGoh6JvYEvVNRi2aACQJ5xHdyOOWEj84QCUOJd8vSSRoPZDMW6ycYZA9dYDOylHhwZlpPvl313VQeKA.ai3Ue7GYk7OizC8bppsMUA/__results___files/__results___65_0.png)
# ======================================================================

# ======================================================================
# # Visualization of Neural Network using tensorspace.js
# ======================================================================

# ======================================================================
# ![visual-png](https://cdn-images-1.medium.com/max/1000/1*_iuD-XPoKrBKG2TyftR8zA.gif)
# ======================================================================

# ======================================================================
# 
# ======================================================================

