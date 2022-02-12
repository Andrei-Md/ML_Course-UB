#%%

# some imports
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"


# TensorFlow ≥2.0 is required
import tensorflow as tf
assert tf.__version__ >= "2.0"


# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.rc('font', size=12)
plt.rc('figure', figsize = (12, 5))

# Settings for the visualizations
#import seaborn as sns
#sns.set_style("whitegrid")
#sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2,'font.family': [u'times']})

import pandas as pd
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 50)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
# Others
import cv2
import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from skimage.io import imread

#%% md

 ## Import data

#%%

# Paths
X_train_path = './dataset/train/train/train/'
X_test_path = './dataset/test/test/test/'
dataset_csv = './dataset/train.txt'
dataset_train = './dataset/train/train'
dataset_test = './dataset/test/test'

#%%

_STOP = 10000
def preprocess(img_paths,dataset_path,preprocess_img_method = None):
    print("loading data")
    data = []
    stop = 0
    for img_path in tqdm.tqdm(img_paths):
        path = os.path.realpath(os.path.join(dataset_path,img_path[1:]))
        img = imread(path)
        if preprocess_img_method:
            img = preprocess_img_method(img)
        else:
            # img = cv2.resize(img,(224,224),cv2.INTER_AREA)
            img = img / 255. #normalize
        data.append(img)
        stop += 1
        if(stop == _STOP):
            break
    print("loading data done")
    return data

def preprocess_img_vgg(img):
    return preprocess_input(img)


#%%

df = pd.read_csv(dataset_csv, delimiter='\ ', header=None)
y_train_full = np.array(df[1])[:_STOP]
img_paths = df[0]
X_train_full = preprocess(df[0], dataset_train,preprocess_img_method=None)
X_train_full= np.stack(X_train_full)

#%%
print(np.max(y_train_full))

#%%
n_classes = np.max(y_train_full)+1
# transform output
new_y_train=[]
for val in y_train_full:
    list = [0]*n_classes
    list[val] = 1
    new_y_train.append(np.array(list))
#%%
new_y_train=np.stack(new_y_train)
#%%

# Split dataset
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, new_y_train, random_state=42)
X_train = X_train.astype("float32")
X_valid = X_valid.astype("float32")
y_train = y_train.astype("float32")
y_valid = y_valid.astype("float32")
#%%

print(X_train[0].shape)
print(X_train[0])

#%%

print(len(X_train))
print(len(y_train))
print(len(X_valid))
print(len(y_valid))

#%%

print(len(X_train))
print(X_train[0].shape)
print(type(X_train))
print(X_train.shape)
print(type(X_train[0]))

#%%

print(y_train[:10])
plt.imshow(X_train[1], cmap='gray')

no_classes = np.max(y_train_full)+1

#%% md

## Model

#%%
tf.random.set_seed(42)
np.random.seed(42)
model = keras.models.Sequential([
    keras.layers.Reshape([158, 158, 1], input_shape=[158, 158]),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(16, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.Conv2D(16, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.Conv2D(32, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.Conv2D(128, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(256, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.Conv2D(256, kernel_size=3, padding="SAME", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Flatten(name="flatten1"),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(no_classes, activation="softmax",name="prediction")
])
# model = keras.models.Sequential([conv_encoder, conv_decoder])


#%%
model.summary()
y_truess=[]
y_predss=[]
def RMSE(y_true, y_pred):
    # y_truess.append(y_true)
    # y_predss.append(y_pred)
    tf.print(y_true.values,"This is y_true")
    tf.print(y_pred,"This is y_pred")
    return tf.sqrt(tf.reduce_mean((y_true - y_pred)**2))

def root_mean_squared_error(y_true, y_pred):
        tf.print(y_true, "This is y_true")
        tf.print(y_pred, "This is y_pred")
        return tf.sqrt(tf.math.reduce_mean(tf.square(y_pred - y_true),axis=-1))

#%%
class CustomMSE(keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        # print(str(y_true)," - " , str(y_pred))
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        return mse + reg * self.regularization_factor



def main():
    rmse = tf.keras.metrics.RootMeanSquaredError()
    sgd = tf.keras.optimizers.SGD(lr=1e-5, decay=(5 * 1e-5), momentum=0.95)
    # model.compile(optimizer=sgd, loss="sparse_categorical_crossentropy", metrics=[rmse,'accuracy'])#['accuracy','mse'])
    model.compile(optimizer=sgd, loss=root_mean_squared_error,
                  metrics=[root_mean_squared_error, 'accuracy'])  # ['accuracy','mse'])

    history = model.fit(x=X_train, y=y_train, epochs=10, verbose=1, validation_data=(X_valid, y_valid), batch_size=50)

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    # %%
    print(y_predss[0])
    tf.print(y_predss[0])

if __name__ == "__main__":
    main()

