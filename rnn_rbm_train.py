import time
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import rnn_rbm
import midi_manipulation 

"""
    This file contains the code for training the RNN-RBM by using the data in the Pop_Music_Midi directory
"""
saved_initial_weights_path = "parameter_checkpoints/initialized.ckpt" #The path to the initialized weights checkpoint file

def main(num_epochs):
    #First, we build the model and get pointers to the model parameters
    x, cost, generate, reconstruction, W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, lr, u0 = rnn_rbm.rnnrbm()

    #The trainable parameters, as well as the initial state of the RNN
    params = [W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0]
    opt_func = tf.train.AdamOptimizer(learning_rate=lr) 
    grad_and_params = opt_func.compute_gradients(cost, params)
    grad_and_params = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in grad_and_params]
    updt = opt_func.apply_gradients(grad_and_params)
    
    #The learning rate of the  optimizer is a parameter that we set on a schedule during training
    #opt_func = tf.train.GradientDescentOptimizer(learning_rate=lr)
    #grad_and_params = opt_func.compute_gradients(cost, params)
    #grad_and_params = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in grad_and_params] #We use gradient clipping to prevent gradients from blowing up during training
    #updt = opt_func.apply_gradients(grad_and_params)

    #songs = midi_manipulation.get_songs('Pop_Music_Midi') #Load the songs 
    songs = midi_manipulation.get_songs('./single_music') #Load the songs 
    saver = tf.train.Saver(params, max_to_keep=1)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init) 
        saver.restore(sess, saved_initial_weights_path) #Here we load the initial weights of the model that we created with weight_initializations.py

        #We run through all of the songs n_epoch times
        print ("starting")
        for epoch in range(num_epochs):
            costs = []
            start = time.time()
            for idx, song in enumerate(songs):
                tr_x = song
                #alpha = min(0.01, 0.1/float(i)) # We decrease the learning rate according to a schedule.
                _, C = sess.run([updt, cost], feed_dict={x: tr_x, lr: 0.01}) 
                costs.append(C) 
            #Print the progress at epoch
            print ("epoch: {} cost: {} time: {}".format(epoch, np.mean(costs), time.time()-start))
            print ("\n")
            saver.save(sess, "parameter_checkpoints/epoch_{}.ckpt".format(epoch))

if __name__ == "__main__":
    main(int(sys.argv[1]))


