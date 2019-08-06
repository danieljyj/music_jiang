# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:45:46 2019

@author: p102380
"""
import numpy as np
import midi
import numpy as np
import glob
from tqdm import tqdm
import midi_manipulation


files = glob.glob('{}/*.mid*'.format('Pop_Music_Midi'))

for f in tqdm(files):
    try:
        song = midi_manipulation.get_song(f)
        if np.shape(song)[0]<20:
            print(np.shape(song))
            print(f)
    except Exception as e:
        print (f, e)            
