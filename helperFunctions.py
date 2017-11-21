import os
import numpy as np
import cv2
import time
import h5py

mean_subtract = np.asarray([103.062623801, 115.902882574, 123.151630838],np.float32)

curr_w = 320
curr_h = 240
height = width = 224

def loadFrame(args):
    (filename,augment) = args

    data = np.zeros((height,width,3),dtype=np.float32)

    try:
        ### load file from HDF5
        filename = filename.replace('.avi','.hdf5')
        filename = filename.replace('UCF-101','UCF-101-hdf5')
        h = h5py.File(filename,'r')
        nFrames = len(h['video'])
        frame_index = np.random.randint(nFrames)
        frame = h['video'][frame_index]

        if(augment==True):
            ## RANDOM CROP - crop 70-100% of original size
            ## don't maintain aspect ratio
            if(np.random.randint(2)==0):
                resize_factor_w = 0.3*np.random.rand()+0.7
                resize_factor_h = 0.3*np.random.rand()+0.7
                w1 = int(curr_w*resize_factor_w)
                h1 = int(curr_h*resize_factor_h)
                w = np.random.randint(curr_w-w1)
                h = np.random.randint(curr_h-h1)
                frame = frame[h:(h+h1),w:(w+w1)]
            
            ## FLIP
            if(np.random.randint(2)==0):
                frame = cv2.flip(frame,1)

            frame = cv2.resize(frame,(width,height))
            frame = frame.astype(np.float32)

            ## Brightness +/- 15
            brightness = 30
            random_add = np.random.randint(brightness+1) - brightness/2.0
            frame += random_add
            frame[frame>255] = 255.0
            frame[frame<0] = 0.0

            ## shrink image by up to 10% within frame
            shrink_size =  int(0.10*np.random.rand()*width)
            frame = cv2.copyMakeBorder(frame,top=shrink_size,bottom=shrink_size,left=shrink_size,right=shrink_size,borderType=cv2.BORDER_CONSTANT,value=(0.0, 0.0, 0.0))
            frame = cv2.resize(frame, (width,height))

            ## random rotate  +/- 12.5 degrees
            angle = 25
            random_rotate = np.random.randint(angle+1) - angle/2
            M = cv2.getRotationMatrix2D((width/2,height/2),random_rotate,1.0)
            frame = cv2.warpAffine(frame,M,(width,height),borderMode=cv2.BORDER_CONSTANT,borderValue=(0.0, 0.0, 0.0))
        else:
            # don't augment
            frame = cv2.resize(frame,(width,height))
            frame = frame.astype(np.float32)

        ## resnet model was trained on images with mean subtracted
        data[:,:,:] = frame - mean_subtract
    except:
        print("Exception: " + filename)
        data = np.array([])
    return data

def getUCF101(base_directory = ''):

    # action class labels
    class_file = open('ucfTrainTestlist/classInd.txt','r')
    lines = class_file.readlines()
    lines = [line.split(' ')[1].strip() for line in lines]
    class_file.close()
    class_list = np.asarray(lines)

    # training data
    train_file = open('ucfTrainTestlist/trainlist01.txt','r')
    lines = train_file.readlines()
    filenames = ['UCF-101/' + line.split(' ')[0] for line in lines]
    y_train = [int(line.split(' ')[1].strip())-1 for line in lines]
    y_train = np.asarray(y_train)
    filenames = [base_directory + filename for filename in filenames]
    train_file.close()

    train = (np.asarray(filenames),y_train)

    # testing data
    test_file = open('ucfTrainTestlist/testlist01.txt','r')
    lines = test_file.readlines()
    filenames = ['UCF-101/' + line.split(' ')[0].strip() for line in lines]
    classnames = [filename.split('/')[1] for filename in filenames]
    y_test = [np.where(classname == class_list)[0][0] for classname in classnames]
    y_test = np.asarray(y_test)
    filenames = [base_directory + filename for filename in filenames]
    test_file.close()

    test = (np.asarray(filenames),y_test)

    return class_list, train, test

def loadSequence(args):

    (filename,sequence_length,is_train) = args

    try:
        filename = filename.replace('.avi','.hdf5')
        filename = filename.replace('UCF-101','UCF-101-features')

        h = h5py.File(filename,'r')
        nFrames = len(h['features'])

        if(nFrames<sequence_length):
            start_index = 0
            new_sequence_length = int(nFrames)
        else:
            start_index = np.random.randint(nFrames - (sequence_length-1))
            new_sequence_length = sequence_length

        indices = range(start_index, start_index+new_sequence_length)
        features = h['features'][indices]
        h.close()

        if(features.shape[0]<sequence_length): # didn't have enough frames in this video to fully load it
            features = repeatSequence(features,sequence_length)
            features = features[0:sequence_length]
        if(is_train):
            if(np.random.randint(2)==0):
                features = np.flipud(features)

    except:
        print("Exception: " + filename)
        features = np.array([])

    return features

def repeatSequence(rnn_input,length_of_sequence):

    while(rnn_input.shape[0]<length_of_sequence):
        rnn_input_flipped = np.flipud(rnn_input)
        rnn_input=np.vstack((rnn_input,rnn_input_flipped))
    return rnn_input