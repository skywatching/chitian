
# coding: utf-8

# In[14]:


import numpy as np
from keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import SeparableConv2D
from keras.layers import BatchNormalization
from keras.models import Model
import keras.preprocessing.image
import keras.backend as K
#from keras.applications.Detect import ResNet50, preprocess_input
#from keras.applications.xception import Xception, preprocess_input
#from keras.applications.vgg19 import VGG19, preprocess_input

from keras.layers.advanced_activations import LeakyReLU

import os
import sys
import glob
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import keras.backend.tensorflow_backend as KTF
from keras import optimizers 
import datetime
import re
import math
import pandas as pd
import json
import keras.optimizers


# In[3]:


BATCH_SIZE_TRAIN = 6
EPOCH_TOTAL = 100000
FILE_TAG= "dong"

IMG_HEIGHT = 1920
IMG_WIDTH = 2560

INPUT_HEIGHT = IMG_HEIGHT // 2
INPUT_WIDTH = IMG_WIDTH // 2

os.environ["CUDA_VISIBLE_DEVICES"] = '3'


# In[4]:


train_round1_train1 = glob.glob(os.path.join("./data/guangdong_round1_train1_20180903/","*.jpg"))
train_round1_train2_norm= glob.glob(os.path.join("./data/guangdong_round1_train2_20180916/","*/*.jpg"))
train_round1_train2_defect = glob.glob(os.path.join("./data/guangdong_round1_train2_20180916/","*/*/*.jpg"))
train_round1_train2_other = glob.glob(os.path.join("./data/guangdong_round1_train2_20180916/","*/*/*/*.jpg"))


# In[4]:


print("len:{} {} {} {}".format(len(train_round1_train1),len(train_round1_train2_norm),len(train_round1_train2_defect),len(train_round1_train2_other)))


# In[5]:


train_all_files = []
train_all_files.extend(train_round1_train1)
train_all_files.extend(train_round1_train2_norm)
train_all_files.extend(train_round1_train2_defect)
train_all_files.extend(train_round1_train2_other)


# In[6]:


g_name2eng = {
    "正常": "norm",
    "不导电": "defect1",
    "擦花": "defect2",
    "横条压凹": "defect3",
    "桔皮": "defect4",
    "漏底": "defect5",
    "碰伤": "defect6",
    "起坑": "defect7",
    "凸粉": "defect8",
    "涂层开裂": "defect9",
    "脏点": "defect10",
    "其他": "defect11",
}


# In[7]:


g_defect_class = [
    'norm',
    'defect1',
    'defect2',
    'defect3',
    'defect4',
    'defect5',
    'defect6',
    'defect7',
    'defect8',
    'defect9',
    'defect10',
    'defect11',
]

g_class_mapping = {defect:classid for classid, defect in enumerate(g_defect_class)}
g_class_count = len(g_class_mapping.keys())
CLASS_COUNT = g_class_count


# In[8]:


def filename2label(filename):
    basename = os.path.basename(filename)
    fileclass = basename.split("2018")[0]
    label = g_class_mapping["defect11"]
    if fileclass in g_name2eng.keys():
        label = g_class_mapping[g_name2eng[fileclass]]
    return label


# In[ ]:


#testfiles = train_all_files.copy()


# In[42]:


#np.random.shuffle(testfiles)
#for i in range(25):
#    print("label: {:2d} fn: {:8} name:{}".format(filename2label(testfiles[i]), g_defect_class[filename2label(testfiles[i])], testfiles[i]))


# In[9]:


g_defect_all2three = np.ones_like(g_defect_class)
g_defect_all2three[0] = 0
g_defect_all2three[-1] = 2
g_defect_all2three


# In[10]:


def preprocess_input(x):
    x = x / 255.
    x = x - 0.5
    x = x * 2.
    return x


# In[11]:


import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import logging
import skimage.io
import skimage.color
import skimage.transform
from keras.utils import Sequence, to_categorical

def prehandle_image(img):
    imgCLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgLAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    planesLAB = cv2.split(imgLAB)
    planesLAB[0] = imgCLAHE.apply(planesLAB[0])
    imgLAB = cv2.merge(planesLAB)
    imgLAB = cv2.cvtColor(imgLAB, cv2.COLOR_LAB2RGB)
    return imgLAB

def load_image(imgfile, augment):
    img = cv2.imread(imgfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(INPUT_WIDTH, INPUT_HEIGHT))
    img = prehandle_image(img)
    if augment:
        seq = iaa.Sequential([
                iaa.Affine(
                    shear=(-10,10),
                    rotate=(-10,10),
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ], random_order=True)
        seq_det = seq.to_deterministic()
        images_aug = seq_det.augment_images([img])
        img = images_aug[0]

    return img

class FileSequence(Sequence):
    def __init__(self, samples, batch_size, augment):
        self.samples = samples.copy()
        self.batch_size = batch_size
        self.augment = augment
        self.on_epoch_end()
        self.block_size = int(np.ceil(len(self.samples)/float(self.batch_size)))

    def epoch_samples(self):
        np.random.shuffle(self.samples)

    def on_epoch_end(self):
        self.epoch_samples()

    def __len__(self):
        return self.block_size

    def __getitem__(self, idx):
        idx = idx % self.block_size
        batch = self.samples[idx*self.batch_size: (idx+1)*self.batch_size]
        batch_images = []
        batch_labels = []
        batch_labels_3 = []
        for f in batch:
            img  = load_image(f, self.augment)
            classid = filename2label(f)
            batch_images.append(img)
            batch_labels.append(to_categorical(classid, CLASS_COUNT))
            batch_labels_3.append(to_categorical(g_defect_all2three[classid], 3))
        batch_images = np.array(batch_images)
        if self.augment:
            seq = iaa.Sequential([
                iaa.Sometimes(0.5,
                    iaa.GaussianBlur(sigma=(0, 0.5))
                ),
                iaa.ContrastNormalization((0.75, 1.5)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                iaa.Multiply((0.8, 1.2), per_channel=0.2),
            ])
            batch_images = seq.augment_images(batch_images)
        batch_images = preprocess_input(batch_images.astype(np.float32))
        batch_labels = np.array(batch_labels)
        batch_labels_3 = np.array(batch_labels_3)
        return batch_images, [batch_labels_3, batch_labels, batch_labels]


# In[17]:


ACTFUNC = LeakyReLU(0.1)
#'relu'

def detect_model(classcnt):
    model_input = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))
    
    x = model_input
    x05 = Conv2D(64, ( 5,  5), strides=(3, 4), activation=ACTFUNC, padding='same', name='block0_conv1')(x)
    x07 = Conv2D(64, ( 7,  7), strides=(3, 4), activation=ACTFUNC, padding='same', name='block0_conv2')(x)
    x09 = Conv2D(64, ( 9,  9), strides=(3, 4), activation=ACTFUNC, padding='same', name='block0_conv3')(x)
    x11 = Conv2D(64, (11, 11), strides=(3, 4), activation=ACTFUNC, padding='same', name='block0_conv4')(x)
    x = layers.concatenate([x05, x07, x09, x11])

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(x)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = layers.add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)
    
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
 
    pred = Dense(3, activation='softmax', name='pred')(x)
    #x = Flatten()(x)
    #x = Dense(512, activation=ACTFUNC)(x)
    out = Dense(classcnt, activation='softmax', name='out')(x)

    model = Model(inputs=model_input, outputs=[pred, out])
    return model


# In[15]:


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def FScore1(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)

def FScore2(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)


# In[16]:


model = detect_model(CLASS_COUNT)
#model.load_weights("output/gd-20180928T1118/gd_0167_0.97.h5")
print(model.summary())


# In[ ]:


def classfly_model(classcnt):
    x = layers.Concatenate()(model.outputs)
    x = layers.Dense(classcnt, activation='softmax', name='cls')(x)
    return Model(inputs=model.inputs, outputs = [model.outputs[0], model.outputs[1], x])


# In[ ]:


cls_model = classfly_model(CLASS_COUNT)
cls_model.summary()


# In[2]:


import json
jsondata = json.load(open("samples.json"))
samples_train = jsondata["data"]["train"]
samples_valid = jsondata["data"]["valid"]


# In[20]:


train_generator = FileSequence(samples_train, BATCH_SIZE_TRAIN, augment=True)
valid_generator = FileSequence(samples_valid, BATCH_SIZE_TRAIN, augment=False)


# In[ ]:


cls_model.load_weights("./output/guangdong-20180930T1131/guangdong_0487_0.99.h5")


# In[21]:


modelfiles = sorted(glob.glob('./output/{}*/*.h5'.format(FILE_TAG)))
init_epoch = 0
log_path = os.path.join('./output/{}-{:%Y%m%dT%H%M}'.format(FILE_TAG, datetime.datetime.now()))
if modelfiles:
    modelfile = modelfiles[-1]
    log_path = os.path.dirname(modelfile)
    cls_model.load_weights(modelfile)
    filename = os.path.splitext(os.path.basename(modelfile))[0]
    init_epoch = int(filename.split('_')[-2])
checkout_file = os.path.join(log_path, "{}_*epoch*.h5".format(FILE_TAG))
checkout_file = checkout_file.replace("*epoch*", "{epoch:04d}_{out_acc:.2f}")


# In[22]:


#optimizer = keras.optimizers.SGD(lr=0.0125, momentum=0.9, decay=1e-6, nesterov=False)
#model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc', precision, recall])
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc', precision, recall])
cls_model.compile(
    optimizer='rmsprop',
    loss={
        'pred':'categorical_crossentropy',
        'out':'categorical_crossentropy',
        'cls':'categorical_crossentropy',
    },
    loss_weights={
        'pred': 0.1,
        'out': 0.1,
        'cls': 1.,
    },
    metrics=['acc', precision, recall, FScore1]
)


# In[23]:


cls_model.load_weights('./output/dong-20181008T1604/dong_0003_0.99.h5')
testfiles = glob.glob("data/guangdong_round1_test_a_20180916/*.jpg")


# In[ ]:


test_results = []
i = 0
for f in testfiles:
    fname = os.path.basename(f)
    imgdata = load_image(f, augment=False)
    imgdata = np.array(imgdata,dtype=np.float32)
    imgdata = preprocess_input(imgdata)
    imgdata = np.expand_dims(imgdata, axis=0)

    pred, out, res = cls_model.predict(imgdata)
    index = np.argmax(res[0])
    result = [fname, g_defect_class[index]]
    test_results.append(result)
    i = i+1
    print("{}/{}".format(i, len(testfiles)))

import datetime
df_result = pd.DataFrame(test_results)
dtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
fname = 'submit-'+ dtime + '.csv'
df_result.to_csv(fname, index=False, sep=',', header=None)

