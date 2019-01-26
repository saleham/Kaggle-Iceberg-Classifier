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
train_y = train['is_iceberg']

#############pre process on images :)
import pre_pros
train_X = pre_pros.pre_pros(train_X)
test_X = pre_pros.pre_pros(test_X)

from keras.applications import VGG19
conv_base = VGG19(weights='imagenet',
include_top=False,
input_shape=(75, 75, 3),pooling='max')

import cv2 # Used to manipulated the images

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

train_X = get_augment(train_X)

tr_feature_data = conv_base.predict(train_X)
ts_feature_data = conv_base.predict(test_X)

print('finish1')

train_X = tr_feature_data.reshape([tr_feature_data.shape[0],-1])
test_X = ts_feature_data.reshape([ts_feature_data.shape[0],-1])

train_y = np.concatenate((train_y,train_y, train_y))

#np.save('train_feature.npy', train_X)
#np.save('test_feature.npy',test_X)

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
# training
test_ratio = 0.2
nr_runs = 3
split_seed = 25
kf = StratifiedShuffleSplit(n_splits=nr_runs, test_size=test_ratio, train_size=None, random_state=split_seed)
pred_xgb = 0

gc.collect()

for r, (train_index, test_index) in enumerate(kf.split(train_X, train_y)):

    x1, x2 = train_X[train_index], train_X[test_index]
    y1, y2 = train_y[train_index], train_y[test_index]
    # x1, x2, y1, y2 = train_test_split(train_X, train_y, test_size=test_ratio, random_state=split_seed + r)
    import xgboost as xgb
    # XGB
    xgb_train = xgb.DMatrix(x1, y1)
    xgb_valid = xgb.DMatrix(x2, y2)
    #
    watchlist = [(xgb_train, 'train'), (xgb_valid, 'valid')]
    params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9,
              'objective': 'binary:logistic', 'seed': 99, 'silent': True}
    params['eta'] = 0.03
    params['max_depth'] = 5
    params['subsample'] = 0.9
    params['eval_metric'] = 'logloss'
    params['colsample_bytree'] = 0.8
    params['colsample_bylevel'] = 0.8
    params['max_delta_step'] = 3
    # params['gamma'] = 5.0
    # params['labmda'] = 1
    params['scale_pos_weight'] = 1.0
    params['seed'] = split_seed + r
    nr_round = 2000
    min_round = 100
    test_X_dup = test_X.copy()

    model1 = xgb.train(params,
                       xgb_train,
                       nr_round,
                       watchlist,
                       verbose_eval=50,
                       early_stopping_rounds=min_round)

    pred_xgb += model1.predict(xgb.DMatrix(test_X_dup), ntree_limit=model1.best_ntree_limit + 45)

    gc.collect()

pred_xgb /= (r+1)

submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=pred_xgb.reshape((pred_xgb.shape[0]))
submission.to_csv('sub/sub_xgb_vgg.csv', index=False)

################################### another machine learning algorithms:))
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

n_features = train_X.shape[1]
from sklearn.ensemble import RandomForestClassifier
from math import sqrt

Rnd_for = RandomForestClassifier(n_estimators=100,max_depth=None,oob_score=True,max_features='auto',min_samples_split=2)

Rnd_for =Rnd_for.fit(train_X,train_y)

pred_rf = Rnd_for.predict_proba(test_X)[:,1]

submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=pred_rf.reshape((pred_rf.shape[0]))
submission.to_csv('sub/sub_rf_vgg.csv', index=False)


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

#ada = AdaBoostClassifier(n_estimators=1000,algorithm = 'SAMME', base_estimator= MLPClassifier(alpha=1))
#ada = ada.fit(train_X,train_y)
#ada_pred = ada.predict_proba(test_X)

################### not worked :))

sample_count = train_X.shape[0]
tr = int(sample_count)
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=0).fit(train_X[:tr], train_y[:tr])
#clf.score(train_X[tr:],train_y[tr:])
pred_gbc= clf.predict_proba(test_X)[:,1]

submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=pred_gbc.reshape((pred_gbc.shape[0]))
submission.to_csv('sub/sub_gbc_vgg.csv', index=False)




from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from itertools import product
from sklearn.ensemble import VotingClassifier
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
clf4 = LogisticRegression(random_state=1)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3),('lg',clf4),('gb',clf)], voting='soft', weights=[2,1,2,3,4])
eclf = eclf.fit(train_X,train_y)

pred_vot = eclf.predict_proba(test_X)[:,1]


submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=pred_vot.reshape((pred_vot.shape[0]))
submission.to_csv('sub/sub_vot_vgg.csv', index=False)

#################### MLP approch :)
from keras import models
from keras import layers
from keras import optimizers

### configure the callbacks of the model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard

callbacks_list = [
    ModelCheckpoint(monitor='val_loss',
                    filepath='best_ml.h5',
                    save_best_only=True),
    EarlyStopping('val_loss', patience=10, mode="min"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001),
   # TensorBoard(log_dir='my_log_dir',histogram_freq=1)
    ]

def get_model():
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_dim=2048))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(512, activation='sigmoid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))


    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
    loss='binary_crossentropy',
    metrics=['acc'])
    return model



# training
test_ratio = 0.2
nr_runs = 5
split_seed = 25
kf = StratifiedShuffleSplit(n_splits=nr_runs, test_size=test_ratio, train_size=None, random_state=split_seed)
pred_mlp = 0
gc.collect()

for r, (train_index, test_index) in enumerate(kf.split(train_X, train_y)):

    x1, x2 = train_X[train_index], train_X[test_index]
    y1, y2 = train_y[train_index], train_y[test_index]
    # x1, x2, y1, y2 = train_test_split(train_X, train_y, test_size=test_ratio, random_state=split_seed + r)
    model = get_model()
    model.fit(x1,y1,
              epochs=100,
              batch_size=32,
              validation_data=(x2, y2),
              callbacks=callbacks_list)

    gc.collect()
    pred_mlp += model.predict(test_X)


pred_mlp /= (r+1)

submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=pred_mlp.reshape((pred_mlp.shape[0]))
submission.to_csv('sub/sub_MLP_vgg1.csv', index=False)
