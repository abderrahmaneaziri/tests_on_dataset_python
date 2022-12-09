# dependencies 
from spectral import open_image
import plotly.express as px
from scipy.io import loadmat
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go   
import warnings
import numpy as np 
warnings.filterwarnings("ignore")

class open_hsi:
    
    '''We first have to import the spectral module and then we can use 
    the open_image function to open the image in case it's a .lan format.
    In the case of .mat format, we have to use the scipy.io.loadmat function
    In the case of .tif or .tiff format, we have to use the cv2.imread function'''
    def __init__(self, hsipath):
        self.hsipath = hsipath
    def get_extention(self):
        '''This function returns the extention of the file'''
        return self.hsipath.split('.')[-1]
    def open_image(self):
        '''This function opens the image and returns the array of the image (width, height, band)'''
        if self.get_extention() == 'mat':
            ''' When opening the .mat file, 
            the image data is usualy stored on
            the last key of the dictionnary.''' 
            pref_mat = loadmat(self.hsipath)
            im_array = pref_mat[list(pref_mat.keys())[-1]] 
            return im_array
        if self.get_extention() == 'lan':
            
            im_spectral = open_image(self.hsipath)
            im_spectral = im_spectral.load()
            return im_spectral
        
        if (self.get_extention() == 'tif') or (self.get_extention() == 'tiff'):
            im_array = cv2.imread(self.hsipath)
            return im_array
        else :
            print('The file extention is not supported')
        
        
    def normalize(self):
        '''This function normalizes the array of 
        the image so that for each band we get a grayscale image
        note: this returns a float array, if an int array is needed
        the conversion must be done manually'''
        im_array = self.open_image()
        im_array = im_array.astype(float)
        im_array = (((im_array-im_array.min()) / (im_array.max()-im_array.min()))*255).astype('uint8')
        return im_array
    def view_image(self, band):
        
        '''This function plots the grayscale image of the hyperspectral cube's band'''
        im_array = self.normalize()
        fig = px.imshow(im_array[:,:,band],color_continuous_scale='gray')
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.show()
        
    def get_image(self, band):
        '''This function returns the grayscale image of the hyperspectral cube's band'''
        im_array = self.normalize()
        return im_array[:,:,band]
    
    def video(self, fps =5.0):
        '''This function returns a video of the hyperspectral cube'''
        im_array = self.normalize()
        height,width,layers=im_array.shape
        filename = f'output_video_{time.strftime("%Y%m%d-%H%M%S")}.mp4'
        video = cv2.VideoWriter(filename ,cv2.VideoWriter_fourcc(*'MP4V'), fps,
                                (width,height), isColor = False)
        images = []
        for i in range(0,layers-1):
            images.append(im_array[:,:,i])
            data = np.array(im_array[:,:,i], np.uint8)
            video.write(data)
        video.release()
        cv2.destroyAllWindows()
        print(f'The video has been saved successfully under the name: {filename}')
        
        #return images
        
        
class open_hsi_mask(open_hsi):
    def __init__(self, hsipath, maskpath):
        ''' This function initializes the class with the path of the hyperspectral image and the path of the mask'''
        super().__init__(hsipath)
        self.maskpath = maskpath
    
        
    def get_extention(self):
        '''This function returns the extention of the file'''
        return self.maskpath.split('.')[-1]
    
    def open_ground_truth(self):
        '''This function reads the ground truth file and returns a numpy array'''
        if self.get_extention() == 'mat':
            pref_mat = loadmat(self.maskpath)
            im_array_gt = pref_mat[list(pref_mat.keys())[-1]]
            return im_array_gt
        if self.get_extention() == 'lan':
            im_spectral_gt = open_image(self.maskpath)
            im_spectral_gt = im_spectral_gt.load()
            return im_spectral_gt
        if (self.get_extention() == 'tif') or (self.get_extention() == 'tiff'):
            im_array_gt = cv2.imread(self.maskpath)
            return im_array_gt
        else: 
            print('The file extention is not supported')
            
    
    
    def get_class_pixels(self, class_number):
        '''This function returns the pixels of a specific class'''
        im_array_gt = self.open_ground_truth()
        if class_number > im_array_gt.max():
            print('The class number is higher than the maximum number of classes')
            return 0
        if class_number < im_array_gt.min():
            print('The class number is lower than the minimum number of classes')
            return 0

        class_pixels = im_array_gt == class_number
        return class_pixels
    
    def class_number(self):
        im_array_gt = self.open_ground_truth()
        '''This function returns the number of classes in the mask'''
        return int(im_array_gt.max()) - int(im_array_gt.min()) +1  
    

    def get_reflectance(self, class_number):
        '''This function returns the reflectance of a specific class'''
        class_pixels = self.get_class_pixels(class_number)
        im_array = open_hsi(self.hsipath).normalize()
        layers = im_array.shape[2]
        band = []
        reflectance = []
        for i in range(0,layers-1):
            band.append(i)
            filtered_pixels = class_pixels * im_array[:,:,i]
            reflectance.append(filtered_pixels.sum()/class_pixels.sum())
        return reflectance, band
        
    def view_reflectance(self, label= False):
        '''This function plots the reflectance of all classes'''
        plt.figure(figsize=(15,10))
        for i in range(0, self.class_number()):
            reflectance, band = self.get_reflectance(i)
            plt.plot(band, np.log(reflectance), label = label[i-1])
        plt.legend(loc='upper right')
