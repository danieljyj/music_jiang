import tensorflow as tf
import sys
import os
from tqdm import tqdm
import rnn_rbm
import midi_manipulation

"""
    This file contains the code for running a tensorflow session to generate music
"""

num_songs = 3 #The number of songs to generate
primer_song = 'Pop_Music_Midi/You Belong With Me - Chorus.midi' #The path to the song to use to prime the network

def main(saved_weights_path):
    #This function takes as input the path to the weights of the network
    x, cost, generate, W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, lr, u0 = rnn_rbm.rnnrbm()#First we build and get the parameters odf the network

    params=[W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0]

    saver = tf.train.Saver(params) #We use this saver object to restore the weights of the model

    song_primer = midi_manipulation.get_song(primer_song)  # primer_song is just one song, not a batch.I It's of dimension 3
    #output folder
    output_folder = "music_outputs"
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, saved_weights_path) #load the saved weights of the network
        # #We generate num_songs songs
        for i in tqdm(range(num_songs)):
            generated_music = sess.run(generate(300), feed_dict={x: song_primer}) #Prime the network with song primer and generate an original song
            new_song_path = "music_outputs/{}_{}".format(i, primer_song.split("/")[-1]) #The new song will be saved here
            midi_manipulation.write_song(new_song_path, generated_music)

if __name__ == "__main__":
    main(sys.argv[1])
    
