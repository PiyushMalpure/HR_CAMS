import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, TensorBoard
from tensorflow.keras.applications.resnet50 import preprocess_input
import time
from skimage.transform import resize
import matplotlib.pyplot as plt

class hr_cams():
    def __init__(self, model, init_weights, image, layer_ids, train_data, test_data, pred_layer=None, classes=None, h=None, w=None):
        """
        Function to initialize all the required variable as defined below : 
        model : Pretrained model to find hr_cams on
        init_weights : Path to the weights of the trained model
        image : images to compute hr_cams for
        layer_ids : layer ids of important layers used in the hr_cams
        train_data : Training data from DataGenerator
        test_data : Testing data from DataGenerator
        pred_layer : layer of the model from which the weights need to be computed set to -1th layer by default
        classes : number of classes 
        h = height of image, should be specified if output height needs to be different than the input image
        w = width of image, should be specified if output width needs to be different than the input image
        """
        
        self.model = model
        self.model_hr = None
        self.init_weights = init_weights
        self.image = image
        self.layer_ids = layer_ids
        self.train_datagen = train_data
        self.test_datagen = test_data
        self.pred_layer = pred_layer
        self.classes = classes
        self.h = h
        self.w = w
    
    
    def new_model_hr(self):
        """
        Function to create an HR_CAM model from the given pretrained model 
        """
        model_hr = self.model

        for layer in model_hr.layers:
            layer.trainable=False
        out_ = []
        for i in range(len(layer_ids)):
            out_[i] = GlobalAveragePooling2D()(model_hr.get_layer(layer_ids[i]).output)
        merge = concatenate([i for i in out_], axis=-1, name='merge')
        op = Dense(2, activation='softmax', name='dense_hr')(merge)

        return keras.Model(inputs=self.model.inputs, outputs=op)
    
    
    
    def get_heatmaps_hr(self):
        """
        Function to compute the final hr_cams from the trained hr_cam model
        model_hr : Trained hr_cam model
        image : images to compute hr_cams for
        layer_ids : layer ids of important layers used in the hr_cams
        pred_layer : layer of the model from which the weights need to be computed set to -1th layer by default
        classes : number of classes 
        h = height of image, should be specified if output height needs to be different than the input image
        w = width of image, should be specified if output width needs to be different than the input image
        """
        model_hr = self.model_hr
        if self.pred_layer is None:
            self.pred_layer = model_hr.get_layer(index=-1)
        heatmaps = []
        wts = []

        self.h = image.shape[-3] if self.h == None else self.h
        self.w = image.shape[-2] if self.w == None else self.w

        if self.classes is None:
            pred = model_hr.predict(self.image)
            classes = np.argmax(pred, axis=1) #___---TO-DO---___If final activation layer is sigmod set classes = pred 
                                              #where pred is your predicted output. For multitask/multilable set pred[.]
                                              #where . is the respective class  

        imp_layers = []
        output_imp_layers = []
        for i in range(len(self.layer_ids)):
            imp_layers[i] = model_hr.get_layer(self.layer_ids[i])
            output_imp_layers[i] = imp_layers[i].output

        hmp_model = Model(inputs=model_hr.inputs, outputs= output_imp_layers)

        out = hmp_model.predict(x=image,verbose=1,batch_size=int(batch_size/4))
        weights = model_hr.get_layer(pred_layer).get_weights()[0] # [0] is for wts [1] for bias

        for i in range(len(self.classes)):

            print('Image no : ',i)

            class_idx = self.classes[i]           
            class_w = weights[:, class_idx] #___---TO-DO---___ If final activation is sigmoid set class_w = weights

            hmp_ = []
            for j in range(len(self.layer_ids)):
                if j == 0:
                    i_idx = 0
                    f_idx = imp_layer[j].output_shape[-1]
                else:
                    i_idx = imp_layer[j-1].output_shape[-1]
                    f_idx = i_idx + imp_layer[j].output_shape[-1]
                hmp_[j] = K.sum(out[0][i:i+1] * class_w[i_idx:f_idx,0], axis=-1)
                hmp_[j] = K.expand_dims(hmp_[j], axis=-1)
                hmp_[j] = K.eval(tf.image.resize_bilinear(hmp_[j], [h, w]))

            hmp = np.squeeze(sum(hmp_))
            heatmaps.append(hmp)
            wts.append(class_w)    

        return np.stack(heatmaps,axis = 0), classes, np.stack(class_w,axis=0)
    
    def display_hr_cams(self, model_hr=self.model, hr_path='hr_model.hdf5'):

        """
            Model to display the final hr cams
            model_hr =  trained HR cam model
            hr_path = path of the hr model
        """
        model_hr = model_hr
        hr_path = hr_path
        model_hr.load_weights(hr_path)
        
        print('Calculating HR hmp')
        hr_hmp, hr_pred, hr_w1 = get_heatmaps_hr(model_hr,self.images, self.layer_ids, self.pred_layer)
        for i in range(len(hr_hmp)):
            hr_hmp[i] = np.maximum(hr_hmp[i],0,hr_hmp[i])
            hr_hmp[i] = (hr_hmp[i]-hr_hmp[i].min())/(hr_hmp[i].max()-hr_hmp[i].min())

        idx = np.random.randint(0,len(_y))
        x_img = np.copy(_x[idx])
        y_img = np.copy(_y[idx])
        print('Idx: ',idx,' Class: ',y_img)

        plt.figure(figsize=(20, 20))

        plt.subplot(1, 2, 1)
        img = cv2.resize(np.squeeze(x_img[:,:,0]), tuple(input_shape[0:2]), 0, 0, 0, interpolation=cv2.INTER_LINEAR)
        plt.imshow(img, 'gray', interpolation='none')

        plt.subplot(1, 2, 2)
        output_0 = cv2.resize(np.squeeze(hr_hmp_0[idx]), tuple(input_shape[0:2]), 0, 0, 0, interpolation=cv2.INTER_LINEAR)
        plt.imshow(img, 'gray', interpolation='none')
        plt.imshow(output_0, 'jet', interpolation='none', alpha=0.50)

        #plt.show()
        plt.savefig('Hr_cam.png')
