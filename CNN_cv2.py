import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from os.path import join as opj
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import load_data
import gc
import cv2

plt.rcParams['figure.figsize'] = 10, 10

#Load the data.
train = pd.read_json("input/train.json")
test = pd.read_json("input/test.json")

##### set the target of train and test
train_X,X_angle = load_data.load(train)
test_X,X_test_angle =  load_data.load(test)



#############pre process on images :)
import pre_pros
train_X = pre_pros.pre_pros(train_X)
test_X = pre_pros.pre_pros(test_X)


def get_augment(imgs):
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []

    for i in range(0, imgs.shape[0]):
        a = imgs[i, :, :, 0]
        b = imgs[i, :, :, 1]
        c = imgs[i, :, :, 2]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)
        bv = cv2.flip(b, 1)
        bh = cv2.flip(b, 0)
        cv = cv2.flip(c, 1)
        ch = cv2.flip(c, 0)

        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)

    more_images = np.concatenate((imgs, v, h))

    return more_images

train_y = train['is_iceberg']
### drop nan_angles_value :)
nan_angle = np.where(X_angle[:,0]!=0)
X_angle = X_angle[nan_angle[0]]
train_y = train_y[nan_angle[0]]
train_X = train_X[nan_angle[0],...]


Xtr_aug = get_augment(train_X)

Ytr_aug = np.concatenate((train_y,train_y,train_y))


### configure the callbacks of the model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import get_model
model = get_model.CNN1()
model.summary()

batch_size = 32

callbacks_list = [
    ModelCheckpoint(monitor='val_loss',
                    filepath='best_ml.h5',
                    save_best_only=True),
    EarlyStopping('val_loss', patience=7, mode="min"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                   patience=5, min_lr=0.001)
   # TensorBoard(log_dir='my_log_dir',histogram_freq=1)
    ]
model.fit(Xtr_aug, Ytr_aug, batch_size=batch_size, epochs=50, verbose=1, callbacks=callbacks_list, validation_split=0.2)

pred_t = model.predict_proba(test_X)

submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=pred_t.reshape((pred_t.shape[0]))
submission.to_csv('sub/sub_cnn3.csv', index=False)