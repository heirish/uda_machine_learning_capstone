#========================4.create model=========================
from keras.models import Sequential
from keras.layers import *
from keras import regularizers

def model_mycase2_tune1(image_width, image_height):
    model = Sequential()
    #model.add(Convolution2D(16, 3, 3, input_shape=(image_width, image_height,3), activation='relu', border_mode='same', name='block1_cov1'))
    #model.add(MaxPooling2D((2,2), border_mode='same', name='block1_pool1'))
    model.add(Conv2D(16, (3, 3), input_shape=(image_width, image_height,3), activation='relu', padding='same', name='block1_cov1'))
    model.add(MaxPooling2D((2,2), padding='same', name='block1_pool1'))
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_cov1'))
    model.add(MaxPooling2D((2,2), padding='same', name='block2_pool1'))
    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_cov1', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D((2,2), padding='same', name='block3_pool1'))
 
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_cov1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_cov2'))
    model.add(MaxPooling2D((2,2), padding='same', name='block4_pool'))
    
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_cov1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_cov2'))
    model.add(Dropout(0.5))
    
    model.add(Flatten(name='flat'))
    #model.add(Dense(512, activation='relu', name='dense1'))
    model.add(Dense(256, activation='relu', name='dense2'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', name='dense3'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', name='dense4'))
        
    return model

def model_vgg161(image_width, image_height):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(image_width, image_height,3), activation='relu', padding='same', name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2,2), padding='same', name='block1_pool'))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2,2), padding='same', name='block2_pool'))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2,2), padding='same', name='block3_pool'))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2,2), padding='same', name='block4_pool'))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(MaxPooling2D((2,2), padding='same', name='block5_pool'))

    model.add(Flatten(name='flat'))
    model.add(Dense(4096, activation='relu', name='dense1'))
    model.add(Dense(512, activation='relu', name='dense2'))
    model.add(Dense(256, activation='relu', name='denseout1'))
    model.add(Dense(64, activation='relu', name='denseout2'))
    model.add(Dense(1, activation='sigmoid', name='denseout3'))
    
    return model

def model_vgg16_pre_tune1(image_width, image_height):
    initial_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(image_width,image_height,3)))
    
    model = Sequential()
    for layer in initial_model.layers:
        layer.trainable = False
        model.add(layer)

    model.add(Flatten(input_shape=initial_model.output_shape[1:]))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def model_vgg16_pre_tune2(image_width, image_height):
    initial_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(image_width,image_height,3)))
    
    model = Sequential()
    for layer in initial_model.layers:
        layer.trainable=False
        model.add(layer)

    model.add(GlobalAveragePooling2D())
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

from keras import applications
import numpy as np
from keras.models import Model
def model_pre_tune3(image_width, image_height):
    initial_model = applications.ResNet50(weights='imagenet', include_top=False, 
                             input_tensor=Input(shape=(image_width,image_height,3)),
                             pooling = 'avg' )  

    x = initial_model.output
    predictions = Dense(1, activation='sigmoid')(x) 
    model = Model(inputs=initial_model.input, outputs=predictions)
    for layer in initial_model.layers:
        layer.trainable = False

    model.summary()
    
    return model

#==========================5.train model==========================
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import math

def train_data(model, model_name, epoch, image_size, num_perbatch, 
               train_dir, train_size, 
               valid_dir, valid_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        data_format='channels_last') 
    valid_datagen = ImageDataGenerator(rescale=1./255,
                           data_format='channels_last') 

    train_generator = train_datagen.flow_from_directory(
       train_dir,
       target_size=image_size,
       batch_size = num_perbatch,
       shuffle = True,
       class_mode='binary')
    valid_generator = valid_datagen.flow_from_directory( 
       valid_dir,
       target_size=image_size,
       batch_size = num_perbatch,
       shuffle = True,
       class_mode='binary')

    log_location = "./" + model_name
    '''
    The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number     of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. 
    Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed.
    '''
    history = model.fit_generator(train_generator,
                steps_per_epoch = math.ceil(train_size / num_perbatch), 
                epochs = epoch,
                validation_data=valid_generator,
                validation_steps = math.ceil(valid_size / num_perbatch), 
                callbacks=[TensorBoard(log_dir=log_location)])
    
    return history


from keras import callbacks
def train_data_earlystopping(model, model_name, epoch, image_size, num_perbatch, 
               train_dir, train_size, 
               valid_dir, valid_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        data_format='channels_last') 
    valid_datagen = ImageDataGenerator(rescale=1./255,
                           data_format='channels_last')
   
    train_generator = train_datagen.flow_from_directory(
       train_dir,
       target_size=image_size,
       batch_size = num_perbatch,
       shuffle = True,
       class_mode='binary')
    valid_generator = valid_datagen.flow_from_directory( 
       valid_dir,
       target_size=image_size,
       batch_size = num_perbatch,
       shuffle = True,
       class_mode='binary')

    log_location = "./" + model_name
    '''
    The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number     of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. 
    Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed.
    '''
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5)  
    file_bath = "./" + model_name + "_top.h5"
    check_point = callbacks.ModelCheckpoint(file_bath, "val_loss", verbose=1, save_best_only=True)
    history = model.fit_generator(train_generator,
                steps_per_epoch = math.ceil(train_size / num_perbatch), 
                epochs = epoch,
                validation_data=valid_generator,
                validation_steps = math.ceil(valid_size / num_perbatch),
                callbacks=[TensorBoard(log_dir=log_location), early_stopping, check_point])
    
    return history

def evaluate_model(model, epoch, image_size, num_perbatch, 
               train_dir, train_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        data_format='channels_last') 
    
    train_generator = train_datagen.flow_from_directory(
       train_dir,
       target_size=image_size,
       batch_size = num_perbatch,
       shuffle = True,
       class_mode='binary')
    
    scores  = model.evaluate_generator(train_generator,
                steps = 10# math.ceil(train_size / num_perbatch)
                )
    return scores

#================================5.visulize model training log data==========================
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model, model_to_dot #newer
#from keras.utils.visualize_util import plot, model_to_dot
from IPython.display import Image, SVG

# visualize model
def visualize_model(model, model_name=None):
    if model == None or model_name == None:
        raise Exception("in save_model, invalid parameter")    
    image_name = model_name + ".png"
    plot_model(model, to_file=image_name, show_shapes=True) #newer
    #plot(model, to_file=image_name, show_shapes=True)
    SVG(model_to_dot(model).create(prog='dot', format='svg'))
    
def visualize_history(history, model_name=None):
    if model_name == None:
        raise Exception("in visualize_history, please input your model_name")
    print(history.history.keys())
    plt.figure(figsize=(12,4))
    plt.subplot(1, 2,1)
    
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(model_name + ' model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='center right')
    plt.subplots_adjust(wspace = .5)
    
    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(model_name + ' model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='center right')
    
    plt.show()


#==============================6.predict test data============================
import pandas as pd

def predict_data(model, model_name, image_size, num_perbatch):
    if model == None or model_name == None:
        raise Exception("in predict_data, invalid parameter")
        
    #test下面还要有一层test目录,./test/test/*.jpg
    gen = ImageDataGenerator(rescale=1./255,
        data_format='channels_last')
    test_generator = gen.flow_from_directory("./test", target_size=image_size, shuffle=False, 
                                              batch_size=num_perbatch,
                                              class_mode=None)
    test = model.predict_generator(test_generator,
                        test_generator.samples) #newer
                        #test_generator.nb_sample)
    test = test.clip(min=0.005, max=0.995) #https://www.kaggle.com/wiki/LogLoss
    df = pd.read_csv("./sample_submission.csv")
    for i, fname in enumerate(test_generator.filenames):
        print(i,fname)
        index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
        df.set_value(index-1, 'label', test[i])

    df.to_csv(model_name + '.csv', index=None)
    

#==============================7.save model=================
#import h5py as h5py
try:
    import h5py
except ImportError:
    h5py = None
def save_model(model, model_name):
    if model == None or model_name == None:
        raise Exception("in save_model, invalid parameter")
    #model.save_weights(model_name + '.h5')
    with open(model_name + '.json', 'w') as f:
        f.write(model.to_json())
              
def predict_small_data(model, data_dir, image_width, image_height, perbatch):
    gen = ImageDataGenerator(rescale=1./255,
        data_format='channels_last')
    test_generator = gen.flow_from_directory(data_dir, 
                                target_size=(image_width, image_height), 
                                shuffle=False, 
                                batch_size=perbatch,
                                class_mode=None)
    test = model.predict_generator(test_generator,
                        test_generator.samples) #newer
                        #test_generator.nb_sample)
    for i, fname in enumerate(test_generator.filenames):
        print(i,fname, test[i])
              
