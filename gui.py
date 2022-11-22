from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
pref_mat = loadmat('PaviaU.mat')
im_array = pref_mat['paviaU'] # this is an array of the hyperspectral cube
print (f'the array cube has a shape of : {im_array.shape}')



from spectral import *
im_spectral = open_image('92AV3C.lan')
im_spectral = im_spectral.load()# when loading it returns an image file 
print (f'the spectral cube has a shape of : {im_spectral.shape}')

import numpy as np
im_spectral_2 = np.memmap('92AV3C.lan', shape=im_spectral.shape)
im_spectral_2.shape


import numpy as np
im_spectral_2 = np.memmap('92AV3C.lan', shape=im_spectral.shape)
im_spectral_2.shape
import os 

def previous_slice():
    pass

def next_slice():
    pass

def process_key(event):
    if event.key == 'p':
        previous_slice()
    elif event.key == 'n':
        next_slice()
fig, ax = plt.subplots()
ax.imshow(im_array[:,:, 43])
fig.canvas.mpl_connect('key_press_event', process_key)

def multi_slice_viewer(volume):
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[2] // 2
    ax.imshow(volume[:,:,ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'p':
        previous_slice(ax)
    elif event.key == 'n':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    """Go to the previous slice."""
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
    ax.images[0].set_array(volume[:,:,ax.index])

def next_slice(ax):
    """Go to the next slice."""
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[2]
    ax.images[0].set_array(volume[:,:,ax.index])
    
    
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)
def multi_slice_viewer(volume):
    remove_keymap_conflicts({'p', 'n'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[2] // 2
    ax.imshow(volume[:,:,ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'p':
        previous_slice(ax)
    elif event.key == 'n':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
    ax.images[0].set_array(volume[:,:,ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[2]
    ax.images[0].set_array(volume[:,:,ax.index])
    
    

multi_slice_viewer(im_array)