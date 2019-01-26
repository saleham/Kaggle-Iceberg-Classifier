import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from os.path import join as opj
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import gc

#Load the data.
train = pd.read_json("input/train.json")
test = pd.read_json("input/test.json")
### train data
#C3_Tr = np.load('input/features/C3_Tr.npy')
#C4_Tr = np.load('input/features/C4_Tr.npy')
F1_Tr = np.load('input/features/F1_Tr.npy')
F2_Tr = np.load('input/features/F2_Tr.npy')

#####  set nan values in train and test data in angle column :)
train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')  # NaN
train['inc_angle'] = train['inc_angle'].fillna(0)
train_angle = np.array(train['inc_angle'])
train_angle = np.tile(train_angle[:, np.newaxis], (1, 6))

###count
train_count = F1_Tr.shape[0]
##### flatening data

train_X = np.concatenate([F1_Tr.reshape(train_count,-1),
                          F2_Tr.reshape(train_count,-1),
                          train_angle], axis=1)


### test data
#C3_Ts = np.load('input/features/C3_Ts.npy')
#C4_Ts = np.load('input/features/C4_Ts.npy')
F1_Ts = np.load('input/features/F1_Ts.npy')
F2_Ts = np.load('input/features/F2_Ts.npy')

#####  set nan values in train and test data in angle column :)
test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')  # NaN
test['inc_angle'] = test['inc_angle'].fillna(0)
test_angle = np.array(test['inc_angle'])
test_angle = np.tile(test_angle[:, np.newaxis], (1, 6))

test_count = F1_Ts.shape[0]
test_X = np.concatenate([F1_Ts.reshape(test_count,-1),
                         F2_Ts.reshape(test_count,-1),
                         test_angle], axis=1)

##### normalization
#train_X = StandardScaler().fit_transform(train_X)
#test_X = StandardScaler().fit_transform(test_X)

##### set the target of train and test
train_y = np.array(train['is_iceberg'])


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
submission.to_csv('sub/sub_xgb_f12.csv', index=False)

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
submission.to_csv('sub/sub_rf_f12.csv', index=False)


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
submission.to_csv('sub/sub_gbc_9.csv', index=False)




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

clf1 = clf1.fit(train_X,train_y)
clf2 = clf2.fit(train_X,train_y)
clf3 = clf3.fit(train_X,train_y)
clf4 = clf4.fit(train_X,train_y)
eclf = eclf.fit(train_X,train_y)

pred_vot = eclf.predict_proba(test_X)[:,1]

