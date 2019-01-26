import os
import numpy as np
import pandas as pd
from subprocess import check_output

sub_path = "sub"
all_files = os.listdir(sub_path)



# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "is_iceberg_" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
concat_sub.head()

cl_num = concat_sub.shape[1]

concat_sub.corr()

# get the data fields ready for stacking
concat_sub['is_iceberg_max'] = concat_sub.iloc[:, 1:cl_num].max(axis=1)
concat_sub['is_iceberg_min'] = concat_sub.iloc[:, 1:cl_num].min(axis=1)
concat_sub['is_iceberg_mean'] = concat_sub.iloc[:, 1:cl_num].mean(axis=1)
concat_sub['is_iceberg_median'] = concat_sub.iloc[:, 1:cl_num].median(axis=1)

ac = int(0.6 *cl_num)

# set up cutoff threshold for lower and upper bounds, easy to twist
cutoff_lo = 0.45
cutoff_hi = 0.55

concat_sub['is_iceberg'] = np.zeros_like(concat_sub['id'])
for i in range(len(concat_sub['is_iceberg'])):
    if (sum(concat_sub.iloc[i,1:cl_num] < cutoff_lo) > ac):
        concat_sub.loc[i,'is_iceberg'] = concat_sub.loc[i,'is_iceberg_min']
    elif((sum(concat_sub.iloc[i,1:cl_num] > cutoff_hi) > ac)):
        concat_sub.loc[i,'is_iceberg'] = concat_sub.loc[i,'is_iceberg_max']
    else:
        concat_sub.loc[i,'is_iceberg'] = concat_sub.iloc[i,2]

concat_sub[['id', 'is_iceberg']].to_csv('stack_minmax_median.csv',
                                        index=False, float_format='%.6f')

concat_sub.to_csv('stack3.csv',
                                        index=False, float_format='%.6f')
