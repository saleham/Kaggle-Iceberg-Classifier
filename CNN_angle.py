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

plt.rcParams['figure.figsize'] = 10, 10

#Load the data.
train = pd.read_json("input/train.json")
test = pd.read_json("input/test.json")

##### set the target of train and test
train_X,X_angle = load_data.load(train)
test_X,X_test_angle =  load_data.load(test)
train_y = np.array(train['is_iceberg'])

### drop nan_angles_value :)
nan_angle = np.where(X_angle[:,0]!=0)
X_angle = X_angle[nan_angle[0]]
train_y = train_y[nan_angle[0]]
train_X = train_X[nan_angle[0],...]


#############pre process on images :)
import pre_pros
train_X = pre_pros.pre_pros(train_X)
test_X = pre_pros.pre_pros(test_X)

#################################### pre analysis
ice_train = train_X[train_y==1]
ship_train = train_X[train_y==0]
## take a look at average heatmap of iceberg and ship
# 1- in each band
# 2- on dot pruduct of each band :)
# 3-
fig = plt.figure()
ax = fig.gca(projection='3d')
row_size = 75
col_size = 75
# Make data.
X = np.linspace(-5, 5,row_size)
Y = np.linspace(-5, 5,col_size)
X, Y = np.meshgrid(X, Y)
Z = ship_train.mean(axis=0)
# Plot the surface.
surf = ax.plot_surface(X, Y, Z[:,:,2], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.title('band_1_ship_train')
plt.show()


fig = plt.figure()
ax = fig.gca(projection='3d')
Z = ice_train.mean(axis=0)
# Plot the surface.
surf = ax.plot_surface(X, Y, Z[:,:,2], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.title('band_1_ice_train')
plt.show()



### configure the callbacks of the model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard

callbacks_list = [
    ModelCheckpoint(monitor='val_loss',
                    filepath='best_ml.h5',
                    save_best_only=True),
    EarlyStopping('val_loss', patience=7, mode="min"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                   patience=5, min_lr=0.001)
   # TensorBoard(log_dir='my_log_dir',histogram_freq=1)
    ]


####### model
from keras import layers
from keras import models
import get_model_angle

######################## data Augmentation
## set train and validation data
from keras.preprocessing.image import ImageDataGenerator
# Define the image transformations here
train_gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = False,
                         width_shift_range = 0.,
                         height_shift_range = 0.,
                         channel_shift_range=0,
                         zoom_range = 0.3,
                         rotation_range = 180 ,
                        )

### we use this for data augmentation of angle and image :)
batch_size=32
epochs = 100
def gen_flow_img_angle(X1, X2, y):
    genX1 = train_gen.flow(X1,y,  batch_size=batch_size,seed=55)
    genX2 = train_gen.flow(X1,X2, batch_size=batch_size,seed=55)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]

############################################################################3
####################  k fold_cross validation to choose best model:)
gc.collect()
k = 5
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=16).split(train_X, train_y))
predicted_test = 0 ## final submission
cv_models = []
for j,(train_idx, valid_idx) in enumerate(folds):
    ### split data :)
    train_X_cv = train_X[train_idx]
    train_y_cv = train_y[train_idx]
    X_angle_cv = X_angle[train_idx]

    valid_X_cv = train_X[valid_idx]
    valid_y_cv = train_y[valid_idx]
    X_angle_valid_cv = X_angle[valid_idx]
    ## for data Augmentation:)
    gen_flow = gen_flow_img_angle(train_X_cv,X_angle_cv,train_y_cv)
    model_cv = get_model_angle.CNN1()
    model_cv.fit_generator(
                    gen_flow,
                    steps_per_epoch=len(train_X_cv) / batch_size,
                    shuffle=True,
                    validation_data=([valid_X_cv,X_angle_valid_cv], valid_y_cv), verbose=1,
                    callbacks=callbacks_list,
                    epochs=epochs)
    print("############################################### cross validation number",j)

    # calculate test score by this cross val model :))
    predicted_test += model_cv.predict([test_X, X_test_angle])
    ## save the model for this
    #cv_models.append(model_cv)
    gc.collect()
#####
predicted_test /= k

submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=predicted_test.reshape((predicted_test.shape[0]))
submission.to_csv('sub_cnn1.csv', index=False)