import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import dump_utils as dump
import numpy as np  # Matrix and vector computation package
import random

# Set the seed for reproducability
np.random.seed(seed=1)


##################################################################################################################################

#Visualize data with more precision
torch.set_printoptions(precision=10, sci_mode=False)

parser = argparse.ArgumentParser("RNN Layer Test")
#dataset
parser.add_argument( '--batch_size', type=int, default=1 ) #numero batch in cui divido dataset
parser.add_argument( '--seq_length', type=int, default=10) # quanti sample ci sono in ogni batch, cioè quante ricorsioni deve fare rete
#rete
parser.add_argument( '--in_size', type=int, default= 30) # numero colonne input
parser.add_argument( '--hidden_size', type=int, default=30)# che è anche la dimensione di ogni output
parser.add_argument( '--out_size', type=int, default=30)# coincide con hidden size
parser.add_argument( '--num_layers', type=int, default=1)
#file
parser.add_argument( '--file_name', type=str, default='RNN-data.h')
parser.add_argument( '--step', type=str, default='FORWARD')     # Possible steps: FORWARD, BACKWARD_GRAD, BACKWARD_ERROR
args = parser.parse_args()

# Network parameters in_size
batch_size = args.batch_size
seq_length = args.seq_length
in_size = args.in_size
out_size = args.out_size
hidden_size=args.hidden_size
num_layers=args.num_layers
simple_kernel = False
current_step = args.step

# Net step
f_step = open('step-check.h', 'w')
f_step.write('#define ' + str(current_step) + '\n')
f_step.close()

# Data file
f = open(args.file_name, "w") 

f.write('#define Tin_l0 ' + str(in_size) + '\n')
f.write('#define Tout_l0 ' + str(out_size) + '\n')
f.write('#define RICORS ' + str(seq_length) + '\n')
f.write("#define L0_IN_CH     (Tin_l0)\n")
f.write("#define L0_OUT_CH    (Tout_l0)\n")
f.write("#define L0_STATE    (Tout_l0*(RICORS+1))\n")
f.write("#define L0_WEIGHTS_input   (L0_IN_CH*L0_OUT_CH)\n")
f.write("#define L0_WEIGHTS_hidden   (L0_OUT_CH*L0_OUT_CH)\n")



##################################################################################################################################

#Golden Model 1 class

# RNN layer
class RNN (nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.num_layers = num_layers #stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # x needs to be: (batch_size, seq, input_size)


    def forward(self, x):
        # Set initial hidden states to zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate RNN
        out,_ = self.rnn(x, h0)  
        #return output of each time step (each layer unfolded),  h_n n-esimo hidden state
        #is the output of the RNN from all timesteps from the last RNN layer
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :] #I take the last timestep because I have input untill the end of the batch_size, every sample of a batch is an input of a neuron
        # out: (batch_size, hidden_size)
         
        return out

#################################################################################################################################

#Golden Model 1

# Training hyperparameters
lr = 1
initial_weights_ih = torch.ones(hidden_size, in_size)
initial_weights_hh = torch.ones(hidden_size, hidden_size)
initial_bias_ih = torch.zeros(hidden_size) 
initial_bias_hh = torch.zeros(hidden_size) 

#It is possible to inizialize weights
"""
#WEIGHTS
temp_value = 0.1
if simple_kernel:
    initial_weights_ih[0:hidden_size] = 0.01
else:
    for i in range(hidden_size):
        for j in range(in_size):
            initial_weights_ih[i][j] = temp_value
            temp_value = temp_value + 0.01

temp_value = 0.01
if simple_kernel:
    initial_weights_hh[0:hidden_size] = 0.01
else:
    for i in range(hidden_size):
        for j in range(hidden_size):
            initial_weights_hh[i][j] = temp_value
            temp_value = temp_value + 0.01
"""

#INDATA
indata = torch.ones(batch_size,seq_length,in_size)/1000
"""
for i in range (seq_length):
    for j in range (in_size):
        indata[0,i,j]=0.1*(i+j)
"""

indata.requires_grad = True
print("\nInput data is: ", indata, indata.shape, indata.dtype)
f.write('PI_L2 float INPUT_VECTOR[L0_IN_CH * RICORS] = {'+dump.tensor_to_string(indata)+'};\n')

#OUTPUT TARGET
label = torch.ones(batch_size, hidden_size)*100


# Define and initialize net
net = RNN(in_size, hidden_size, num_layers)
print("\nInitializing net parameters weights ih to {}.\nParameters are: ".format(initial_weights_ih))
print("\nInitializing net parameters weights hh to {}.\nParameters are: ".format(initial_weights_hh))

net.rnn.weight_ih_l0 = nn.Parameter(initial_weights_ih)
net.rnn.weight_hh_l0 = nn.Parameter(initial_weights_hh)
RNN.weight_ih_l0= nn.Parameter(initial_weights_ih)
RNN.weight_hh_l0= nn.Parameter(initial_weights_hh)
net.rnn.bias_ih_l0 = nn.Parameter(initial_bias_ih)
net.rnn.bias_hh_l0 = nn.Parameter(initial_bias_hh)
RNN.bias_ih_l0= nn.Parameter(initial_bias_ih)
RNN.bias_ih_l0= nn.Parameter(initial_bias_hh)


for name, parameter in net.named_parameters():
    print(name, parameter, parameter.shape)


f.write('PI_L2 float L0_WEIGHTS_params_INPUT[L0_WEIGHTS_input] = {'+dump.tensor_to_string(net.weight_ih_l0)+'};\n')
f.write('PI_L2 float L0_WEIGHTS_params_HIDDEN[L0_WEIGHTS_hidden] = {'+dump.tensor_to_string(net.weight_hh_l0)+'};\n')

# Optimizer and criterion
criterion = nn.MSELoss()

for i in range(1):
    # Do a forward computation
    net.zero_grad()
    output = net.forward(indata) # shape(batch_size x hidden_size) un output di lughezza hidden_size per ogni batch
    print("\nNet output is: ", output, output.shape, output.dtype)
    f.write('PI_L2 float L0_OUT_FW [L0_OUT_CH] = {'+dump.tensor_to_string(output)+'};\n\n')


    loss = criterion(output, label)
    print("\nLoss is: ", loss, loss.shape, loss.dtype)
    f.write('PI_L2 float L0_LOSS = '+str(loss.item())+';\n')

    # Manually compute outdiff
    loss_meanval = 1/hidden_size
    output_diff = loss_meanval * 2.0 * (output - label)
    print("\nOutput loss is: ", output_diff, output_diff.shape, output_diff.dtype)
    f.write('PI_L2 float L0_OUT_GRAD [L0_OUT_CH] = {'+dump.tensor_to_string(output_diff)+'};\n')

    # Backward and show gradients
    loss.backward()
    print("\nNetwork gradients are: ")
    for name, parameter in net.named_parameters():
        print(name, parameter.grad)
    #weights grandients over the all timesteps
    f.write('PI_L2 float L0_WEIGHT_params_INPUT_GRAD_FINAL_pytorch [L0_WEIGHTS_input] = {'+dump.tensor_to_string(net.rnn.weight_ih_l0.grad)+'};\n')
    f.write('PI_L2 float L0_WEIGHT_params_HIDDEN_GRAD_FINAL_pytorch [L0_WEIGHTS_hidden] = {'+dump.tensor_to_string(net.rnn.weight_hh_l0.grad)+'};\n')

    print("\nInput grad is: ", indata.grad)
    f.write('PI_L2 float L0_IN_GRAD [L0_IN_CH * RICORS] = {'+dump.tensor_to_string(indata.grad)+'};\n')




###################################################################################################################################


# Golden Model 2 functions
#this second Golden Model is created to be able to access the values of the hidden state  (impossible in pytorch) 
# and to create a model that is less black box, and shows more clearly the operations of the net


def copy(X):
    Y=np.zeros((1,X.shape[0]),dtype=float)
    for i in range (0,Y.shape[1]):
        Y[0,i]=X[i]
    return Y

def tensor_to_string(S):
    tensor_string=''
    sz0= len(S[:,0])
    sz1= len(S[0,:])
    for i in range(sz0):
        for j in range(sz1):
            tensor_string+=str(S[i,j].item())
            tensor_string+='f, ' if (j*sz0+i)<(sz1*sz0-1) else 'f'
    return tensor_string    


#Compute state k from the previous state (sk) and current input (xk), by use of the input weights (wx) and recursive weights (wRec).
def update_state(xk, sk, wx, wRec,bih,bhh):
    
    return np.tanh(np.matmul(wx,xk) + bih   +  np.matmul(wRec,sk) + bhh )


def forward_states(X, wx, wRec,bih,bhh):
   
    #Unfold the network and compute all state activations given the input X, input weights (wx), and recursive weights (wRec).
    #Return the state activations in a matrix, the last column S[:,-1] contains the final activations.
   
    # Initialise the matrix that holds all states for all 
    #  input sequences. The initial state s0 is set to 0.
    S = np.zeros((hidden_size, len(X[:,1])+1))
    # Use the recurrence relation defined by update_state to update 
    #  the states trough time.
    for k in range(0, len(X[:,1])):
        #S[k] = S[k-1] * wRec + X[k] * wx
        S[:,k+1] = update_state(X[k,:], S[:,k], wx, wRec,bih,bhh)

    f.write('PI_L2 float L0_HIDDEN_STATE [L0_OUT_CH*(RICORS+1)] = {'+tensor_to_string(S.T)+'};\n')

    return S

def loss(y, t): 
    #MSE between the targets t and the outputs y
    return np.mean((t - y)**2)

def output_gradient(y, t):
    
    #Gradient of the MSE loss function with respect to the output y.
    return (1/hidden_size)*2. * (y - t)


def backward_gradient(X, S, grad_out, W):
    
    #Backpropagate the gradient computed at the output (grad_out) through the network.
    #Accumulate the parameter gradients for wX and wRec by for each layer by addition.
    # Return the parameter gradients as a tuple, and the gradients at the output of each layer.
    
    # Initialise the array that stores the gradients of the loss with 
    #  respect to the states.
    grad_over_time = np.zeros((hidden_size, nb_of_samples+1))       #(hidden_size x in_size+1)  rows( nb_samples) , column+1 of X
    grad_hidden=np.zeros((hidden_size,1)) 
    grad_hidden_final=np.zeros((hidden_size,1)) 
    grad_over_time[:,-1] = grad_out
    # Set the gradient accumulations to 0
    wx_grad = np.zeros((hidden_size, in_size),dtype=float)
    wRec_grad = np.zeros((hidden_size, hidden_size),dtype=float)
    bih_grad=np.zeros((hidden_size),dtype=float)
    bhh_grad=np.zeros((hidden_size),dtype=float)
    


    for k in range( X.shape[0]-1, -1, -1):
        
        # Compute the parameter gradients and accumulate the results.
        dnext=1-(S[:,k+1])**2 # derivative of tanh dnext (hidden_size x 1)
        derfin= ((dnext*grad_over_time[:,k+1])[np.newaxis]).T

        X_row=copy(X[k,:])
        S_row=copy(S[:,k].T)

        wx_grad += np.dot(  derfin  , X_row)             #(hidden_size x 1) x (1 x in_size)= (hidden_size x in_size) 

        wRec_grad += np.dot(  derfin ,  S_row)            #(hidden_size x 1) x (1 x hidden_size)= (hidden_size x hidden_size) 

        bih_grad +=(dnext)*grad_over_time[:,k+1]
        bhh_grad += (dnext)*grad_over_time[:,k+1]          # (1 x hidden_size)  * (4x1)
    
        # Compute the gradient at the output of the previous layer   
        grad_over_time[:,k] =(dnext)*(np.matmul(grad_over_time[:,k+1],W[1].T))
        
        #weights grandients after a single timestep
        if(k==X.shape[0]-1):
            f.write('PI_L2 float L0_WEIGHT_params_INPUT_GRAD [L0_WEIGHTS_input] = {'+tensor_to_string(wx_grad)+'};\n')
            f.write('PI_L2 float L0_WEIGHT_params_HIDDEN_GRAD [L0_WEIGHTS_hidden] = {'+tensor_to_string(wRec_grad)+'};\n')


    f.write('PI_L2 float L0_WEIGHT_params_INPUT_GRAD_FINAL [L0_WEIGHTS_input] = {'+tensor_to_string(wx_grad)+'};\n')
    f.write('PI_L2 float L0_WEIGHT_params_HIDDEN_GRAD_FINAL [L0_WEIGHTS_hidden] = {'+tensor_to_string(wRec_grad)+'};\n')


    for k in range(X.shape[0]-1,  X.shape[0], 1):
       grad_hidden[:,0]=grad_over_time[:,k]

    grad_hidden_final[:,0]=grad_over_time[:,0]
    


    f.write('PI_L2 float L0_STATE_GRAD [L0_OUT_CH*(RICORS)] = {'+tensor_to_string(grad_hidden)+'};\n')
    f.write('PI_L2 float L0_STATE_GRAD_FINAL [L0_OUT_CH*(RICORS)] = {'+tensor_to_string(grad_hidden_final)+'};\n')

    return (wx_grad, wRec_grad,bih_grad,bhh_grad), grad_over_time



def update_rprop(X, t, W):

   # Perform forward and backward pass to get the gradients
    S = forward_states(X, W[0], W[1],W[2],W[3])
    out=S[:,-1]
    grad_out = output_gradient(S[:,-1], t)          #must be the same dimension
    W_grads, _ = backward_gradient(X, S, grad_out, W)

    return W_grads


###################################################################################################################################

# Golden Model 2


# Create dataset
nb_of_samples = seq_length # che è il numero di ricorsioni nella rete unfoldata
#deve essere stessa dimensione output cioè di y, e se ho un output per ogni sample deve essere uguale a nb_samples


X = np.ones((nb_of_samples,in_size),dtype=float)/1000
"""
for i in range (nb_of_samples):
    for j in range (in_size):
        X[i,j]=0.1*(i+j)
"""

# Create the targets for each sequence
y = np.ones((hidden_size),dtype=float)*100



Wx =np.ones((hidden_size, in_size),dtype=float)
Wrec =  np.ones((hidden_size, hidden_size),dtype=float)
"""
#WEIGHTS
temp_value = 0.1

for i in range(hidden_size):
    for j in range(in_size):
        Wx[i][j] = temp_value
        temp_value = temp_value + 0.01

temp_value = 0.01

for i in range(hidden_size):
    for j in range(hidden_size):
        Wrec[i][j] = temp_value
        temp_value = temp_value + 0.01
"""

Bih =  np.zeros((hidden_size),dtype=float)
Bhh = np.zeros((hidden_size),dtype=float)
W =[Wx,Wrec, Bih, Bhh]

Wx_grad = np.zeros((hidden_size, in_size),dtype=float)
Wrec_grad = np.zeros((hidden_size, hidden_size),dtype=float)
Bih_grad= np.zeros((hidden_size),dtype=float)
Bhh_grad = np.zeros((hidden_size),dtype=float)

W_grad =[Wx_grad,Wrec_grad, Bih_grad, Bhh_grad]


#TRAINING
W_grad = update_rprop(X, y, W,)
print(W_grad)

f.write('\n\n')

f.close()
