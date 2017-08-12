#========================4.create model=========================
from keras.models import Sequential
from keras.layers import *

# step_size and padding 会影响内存占用，step和padding调小以后perbatch就可以增大了 
def model_vgg19(image_width, image_height):
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, input_shape=(image_width, image_height,3), activation='relu', border_mode='same', name='block1_conv1'))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2'))
    model.add(MaxPooling2D((2,2), border_mode='same', name='block1_pool'))

    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1'))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2'))
    model.add(MaxPooling2D((2,2), border_mode='same', name='block2_pool'))

    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv4'))
    model.add(MaxPooling2D((2,2), border_mode='same', name='block3_pool'))

    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv4'))
    model.add(MaxPooling2D((2,2), border_mode='same', name='block4_pool'))

    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv4'))
    model.add(MaxPooling2D((2,2), border_mode='same', name='block5_pool'))

    model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='cov4'))
    model.add(MaxPooling2D((3,3), border_mode='same', name='pool4'))

    model.add(Flatten(name='flat'))
    model.add(Dense(4096, activation='relu', name='dense1'))
    model.add(Dense(4096, activation='relu', name='dense2'))
    #model1.add(Dropout(0.4, name="dropout2"))
    model.add(Dense(1, activation='sigmoid', name='dense3'))
    
    return model

def model_case1(image_width, image_height):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(image_width, image_height,3), activation='relu', border_mode='same', name='block1_cov1'))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block1_cov2'))
    model.add(MaxPooling2D((3,3), border_mode='same', name='block1_pool1'))
    
    model.add(Convolution2D(64, 5, 5, activation='relu', border_mode='same', name='block2_cov1'))
    model.add(Convolution2D(64, 5, 5, activation='relu', border_mode='same', name='block2_cov2'))
    model.add(MaxPooling2D((3,3), border_mode='same', name='block2_pool1'))

    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block3_cov1'))
    model.add(MaxPooling2D((3,3), border_mode='same', name='block3_pool'))

    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_cov1'))
    model.add(MaxPooling2D((3,3), border_mode='same', name='block4_pool'))
    #model.add(Dropout(0.2, name="block3_dropout"))

    model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='block5_cov1'))
    model.add(MaxPooling2D((3,3), border_mode='same', name='block5_pool'))
    
    model.add(Convolution2D(2048, 2, 2, activation='relu', border_mode='same', name='block6_cov1'))
    model.add(MaxPooling2D((2,2), border_mode='same', name='block6_pool'))

    model.add(Flatten(name='flat'))
    model.add(Dense(64, activation='relu', name='dense1'))
    model.add(Dense(64, activation='relu', name='dense2'))
    #model.add(Dropout(0.4, name="dropout"))
    model.add(Dense(1, activation='sigmoid', name='dense3'))
    
    return model


#==========================5.train model==========================
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

def train_data(model, model_name, epoch, image_size, num_perbatch, 
               train_dir, train_size, 
               valid_dir, valid_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        dim_ordering='tf')
        #data_format='channels_last') #newer
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
       train_dir,
       target_size=image_size,
       batch_size = num_perbatch,
       class_mode='binary')
    valid_generator = train_datagen.flow_from_directory(
       valid_dir,
       target_size=image_size,
       batch_size = num_perbatch,
       class_mode='binary')

    log_location = "./" + model_name
    history = model.fit_generator(train_generator,
                samples_per_epoch = train_size,
                nb_epoch=epoch,
                validation_data=valid_generator,
                nb_val_samples=valid_size,
                callbacks=[TensorBoard(log_dir=log_location)])
    
    return history


#================================5.visulize model training log data==========================
import matplotlib.pyplot as plt
#from keras.utils.vis_utils import plot_model, model_to_dot #newer
from keras.utils.visualize_util import plot, model_to_dot
from IPython.display import Image, SVG

# visualize model
def visualize_model(model, model_name=None):
    if model == None or model_name == None:
        raise Exception("in save_model, invalid parameter")    
    image_name = model_name + ".png"
    #plot_model(model, to_file=image_name, show_shapes=True) #newer
    plot(model, to_file=image_name, show_shapes=True)
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
    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory("./test", target_size=image_size, shuffle=False, 
                                              batch_size=num_perbatch,
                                              class_mode=None)
    test = model.predict_generator(test_generator, test_generator.nb_sample)
    df = pd.read_csv("./sample_submission.csv")
    for i, fname in enumerate(test_generator.filenames):
        print(i,fname)
        index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
        df.set_value(index-1, 'label', test[i])

    df.to_csv(model_name + '.csv', index=None)
    df.head(10)
    
#==============================7.save model=================
#import h5py as h5py
try:
    import h5py
except ImportError:
    h5py = None
def save_model(model, model_name):
    if model == None or model_name == None:
        raise Exception("in save_model, invalid parameter")
    model.save_weights(model_name + '.h5')
    with open(model_name + '.json', 'w') as f:
        f.write(model.to_json())
