
import pandas as pd
import numpy as np

# Get train set
data_set = pd.read_csv(r"C:\Users\Sandisk\Desktop\Machine Learning\Data Hackathon\Project\data_set/Train.csv")
print('------Data set size: {}------'.format(data_set.shape))
print('---------Data type: ---------')
print(data_set.dtypes)
print('-----------------------------')



# Die id extraction
process_set = data_set.iloc[:, :-2]
die_id_split = data_set['die_id'].str.split(pat='-', expand=True)
process_set['W'] = die_id_split[1].astype('float32')
process_set['X'] = die_id_split[2].astype('float32')
process_set['Y'] = die_id_split[3].astype('float32')
data_set = process_set.join(data_set['passfail'])

print('---------die_id split: ---------')
print(data_set.head(1))
print(data_set.dtypes)
print('-----------------------------')

# Normalization
feature_num = data_set.shape[1] - 1
drop_col_list = []
drop_col_name = []
drop_col_num = 0
scattered_col_num = 0
log_correction = 0.1
weight_adj = 0.2


for col in range(feature_num):
    feature_mean = data_set.iloc[:, col].mean()
    feature_max = data_set.iloc[:, col].max()
    feature_min = data_set.iloc[:, col].min()


    # Normalize by mean, max and min
    data_set.iloc[:, col] = ((data_set.iloc[:, col] - feature_mean) / (feature_max - feature_min) * 2).fillna(0)
    data_set.iloc[:, col] = data_set.iloc[:, col].astype('float32')

    '''
    if (feature_max / (feature_mean+0.01) > 10):
        # Take log for scattered values
        data_set.iloc[:, col] = (np.log (abs(data_set.iloc[:, col] - feature_mean + log_correction))
                                 / (np.log(feature_max - feature_min + log_correction)) * 2).fillna(0) * weight_adj
        data_set.iloc[:, col] = data_set.iloc[:, col].astype('float32')
        scattered_col_num += 1

    else:
        # Normalize by mean, max and min
        data_set.iloc[:, col] = ((data_set.iloc[:, col] - feature_mean) / (feature_max - feature_min) * 2).fillna(0)
        data_set.iloc[:, col] = data_set.iloc[:, col].astype('float32')
    '''


# Drop columns and shuffle rows
data_set = data_set.sample(frac = 1)

# Increase dimension
data_set_square = (data_set.iloc[:, :-1]).pow(2)
data_set_combine = data_set.iloc[:, :-1].join(data_set_square, lsuffix='_1', rsuffix='_2')
data_set = data_set_combine.join(data_set.iloc[:,-1])

# Split into train set and test set
train_num = int(data_set.shape[0]*0.75)
train_set = data_set.iloc[:train_num, :]
test_set = data_set.iloc[train_num+1:-1, :]

# DataFrame to CSV
train_set.to_csv(r"C:\Users\Sandisk\Desktop\Machine Learning\Data Hackathon\Project\data_set/train_set_norm.csv", index=False)
test_set.to_csv(r"C:\Users\Sandisk\Desktop\Machine Learning\Data Hackathon\Project\data_set/test_set_norm.csv", index=False)

# Print info
print('------Scattered column number: {}------'.format(scattered_col_num))
print('------Same data column number: {}------'.format(drop_col_num))
print('------Train set size: {}------'.format(train_set.shape))
print('------Test set size: {}------'.format(test_set.shape))

print('---------After normalization: ---------')
print(test_set.head(1))
print(test_set.dtypes)
print('-----------------------------')
