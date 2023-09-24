
import pandas as pd
import csv
import math
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from ds_predict_nn import *

# Dataset preparation and preprocess
test_set = pd.read_csv(r"C:\Users\Sandisk\Desktop\Machine Learning\Data Hackathon\Project\data_set/Test_norm.csv")
test_orig = pd.read_csv(r"C:\Users\Sandisk\Desktop\Machine Learning\Data Hackathon\Project\data_set/Test.csv")
# Get data size
test_data_size = test_set.shape

# Print data size
print('test data size:{}'.format(test_data_size))

# Pull data groups from DataFrame into (data group, label)
test_data = torch.tensor(test_set.values).to(torch.float)

# Build/load neuron network
#ds_predict_model = ds_predict_nn()
ds_predict_model = torch.load('ds_predict_model.pth')
output_list = []

counter = 0
for data in test_data:
    counter += 1
    # Forward propagation and check loss
    output = ds_predict_model(data)
    output_list.append(output.argmax(-1).item())
    if counter%1000 == 0:
        print('current data: {}'.format(counter))


output_df = pd.DataFrame({'passfail': output_list})
print(output_df.head(1))
result = test_orig.join(output_df)[['die_id', 'passfail']]
print(result.head(1))
result.to_csv(r"C:\Users\Sandisk\Desktop\Machine Learning\Data Hackathon\Project\data_set/Result.csv", index=False)

