import tensorflow as tf
import numpy as np

import RBM
import midi_manipulation


"""
    This file contains the TF implementation of the RNN-RBM, as well as the hyperparameters of the model
"""

note_range         = midi_manipulation.span #The range of notes that we can produce
rbm_timesteps      = midi_manipulation.rbm_timesteps #The number of note timesteps that we produce with each RBM
n_visible          = 2*note_range*rbm_timesteps #The size of each data vector and the size of the RBM visible layer , n_visible=780 in our case
n_hidden           = 200 #The size of the RBM hidden layer
n_hidden_recurrent = 100 #The size of each RNN hidden layer

def rnnrbm():

    #This function builds the RNN-RBM and returns the parameters of the model

    x  = tf.placeholder(tf.float32, [None, n_visible]) #The placeholder variable that holds our data
    lr  = tf.placeholder(tf.float32) #The learning rate. We set and change this value during training.
    
    size_bt = tf.shape(x)[0] #the batch size
    # parameters
    W   = tf.Variable(tf.zeros([n_visible, n_hidden]), name="W")
    Wuh = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden]), name="Wuh")
    Wuv = tf.Variable(tf.zeros([n_hidden_recurrent, n_visible]), name="Wuv")
    Wvu = tf.Variable(tf.zeros([n_visible, n_hidden_recurrent]), name="Wvu")
    Wuu = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wuu")
    bh  = tf.Variable(tf.zeros([1, n_hidden]), name="bh")
    bv  = tf.Variable(tf.zeros([1, n_visible]), name="bv")
    bu  = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="bu")
    u0  = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="u0")
    BH_t = tf.Variable(tf.zeros([1, n_hidden]), name="BH_t")
    BV_t = tf.Variable(tf.zeros([1, n_visible]), name="BV_t")


    def rnn_recurrence(u_tm1, v_t):
        #Iterate through the data in the batch and generate the values of the RNN hidden nodes
        v_t  =  tf.reshape(v_t, [1, n_visible])
        u_t = (tf.sigmoid(bu + tf.matmul(v_t, Wvu) + tf.matmul(u_tm1, Wuu)))
        return u_t

    def visible_bias_recurrence(bv_t, u_tm1):
        #Iterate through the values of the RNN hidden nodes and generate the values of the visible bias vectors
        bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv))
        return bv_t

    def hidden_bias_recurrence(bh_t, u_tm1):
        #Iterate through the values of the RNN hidden nodes and generate the values of the hidden bias vectors
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh))
        return bh_t       

####################
#Below is two functions for generation and construction, but not used in training
####################

    def generate(timesteps, primer=x):
        """
            This function handles generating music. This function is one of the outputs of the build_rnnrbm function
            Args:
                timesteps (int): The number of timesteps to generate
                primer (tf.placeholder): The primer song
            Returns:
                The generated music, as a tf.Tensor
        """
        
        def generate_recurrence(step, u_tm1, music):
            #To generate music: This function builds and runs the gibbs steps for each RBM in the chain
            bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv))
            bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh))

            #Run the Gibbs step to get the music output. Prime the RBM with the previous musical output.
            x_out = RBM.gibbs_sample(tf.reshape(music[-1,:], [1,n_visible]), W, bv_t, bh_t, k=25)
            
            #Update the RNN hidden state based on the musical output and current hidden state.
            u_t  = (tf.sigmoid(bu + tf.matmul(x_out, Wvu) + tf.matmul(u_tm1, Wuu)))

            #Add the new output to the musical piece
            music = tf.concat([music, x_out], axis=0)
            return step+1, u_t, music
            
        Uarr = tf.scan(rnn_recurrence, primer, initializer=u0)# x is of dimension 2, Uarr is of dimension 3
        u_tm1 = Uarr[-1, :, :] # the beginning hidden recurrent unit of our generation
        music = tf.zeros([1, n_visible])
        
        loop_vars=[tf.constant(0), u_tm1, music]
        cond = lambda count, *args : count < timesteps
        
        [_, _, music] = tf.while_loop(cond, generate_recurrence, loop_vars, 
                                    shape_invariants=[tf.constant(0).get_shape(), u_tm1.get_shape(), tf.TensorShape([None, None])])
        return music

    def reconstruction():
        #This function handles reconstructing music. This function is one of the outputs of the rnnrbm() function
        primer=x
        #Scan through the rnn and generate the value for each hidden node in the batch
        Uarr  = tf.scan(rnn_recurrence, primer, initializer=u0) # primer is of dimension 2, Uarr is of dimension 3
        Uarr = tf.concat([tf.reshape(u0,[1, 1, n_hidden_recurrent]), Uarr[:-1,:,:]], 0)
        #Scan through the rnn and generate the visible and hidden biases for each RBM in the batch
        BV_t = tf.reshape(tf.scan(visible_bias_recurrence, Uarr, bv), [size_bt, n_visible])
        BH_t = tf.reshape(tf.scan(hidden_bias_recurrence, Uarr, bh), [size_bt, n_hidden])
        music = RBM.gibbs_sample(primer, W, BV_t, BH_t, k=1)
        return music
            


##########################
# Below is for training
##########################   
    #Reshape our bias matrices to be the same size as the batch.
    tf.assign(BH_t, tf.tile(BH_t, [size_bt, 1]))
    tf.assign(BV_t, tf.tile(BV_t, [size_bt, 1]))
    #Scan through the rnn and generate the value for each hidden node in the batch
    u_t  = tf.scan(rnn_recurrence, x, initializer=u0)
    #Scan through the rnn and generate the visible and hidden biases for each RBM in the batch
    BV_t = tf.reshape(tf.scan(visible_bias_recurrence, u_t, tf.zeros([1, n_visible], tf.float32)), [size_bt, n_visible])
    BH_t = tf.reshape(tf.scan(hidden_bias_recurrence, u_t, tf.zeros([1, n_hidden], tf.float32)), [size_bt, n_hidden])
    #Get the free energy cost from each of the RBMs in the batch 
    cost = RBM.get_free_energy_cost(x, W, BV_t, BH_t, k=15)
    return x, cost, generate, reconstruction, W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, lr, u0

