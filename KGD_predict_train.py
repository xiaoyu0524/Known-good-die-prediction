
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
train_set = pd.read_csv(r"C:\Users\Sandisk\Desktop\Machine Learning\Data Hackathon\Project\data_set/train_set_norm.csv")
test_set = pd.read_csv(r"C:\Users\Sandisk\Desktop\Machine Learning\Data Hackathon\Project\data_set/test_set_norm.csv")

# Get data size
train_data_size = train_set.shape
test_data_size = test_set.shape

# Print data size
print('train data size:{}'.format(train_data_size))
print('test data size:{}'.format(test_data_size))

# Pull data groups from DataFrame into (data group, label)
train_label = torch.tensor(train_set['passfail'].values).to(torch.long)
train_data = torch.tensor(train_set.iloc[:,:-1].values).to(torch.float)
test_label = torch.tensor(test_set['passfail'].values).to(torch.long)
test_data = torch.tensor(test_set.iloc[:,:-1].values).to(torch.float)

train = TensorDataset(train_data, train_label)
test = TensorDataset(test_data, test_label)

batch_size = 40
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)


# Build/load neuron network
#ds_predict_model = ds_predict_nn()
ds_predict_model = torch.load('ds_predict_model.pth')

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
learning_rate = 2e-7
optimizer = torch.optim.RMSprop(ds_predict_model.parameters(), lr = learning_rate)

# Other model parameters
total_train_step = 0
total_test_step = 0
epoch = 100

# Add Tensorboard
writer = SummaryWriter('ds_predict_logs')



# Epoch
for step in range(epoch):
    print('----------Start {} round of training----------'.format(step+1))

    # Start training
    ds_predict_model.train()
    current_sample_num = 0
    counter = 0
    for data in train_loader:
        counter += 1
        # Forward propagation
        input_data, label = data
        output = ds_predict_model(input_data)

        # Compute loss, backward propagation and optimization
        optimizer.zero_grad()
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        # Get train step and loss
        current_sample_num += 1
        total_train_step += 1
        if total_train_step %200 == 0:
            print('Train step: {}, Loss: {}'.format(total_train_step, loss.item()))
            writer.add_scalar('Train loss', loss.item(), total_train_step)
            #print(label.item())
            #print(output.item())
        #if counter == 5000:
        #    break

    # Start testing in current epoch
    ds_predict_model.eval()
    total_test_loss = 0
    total_accuracy = 0
    total_true_positive = 0
    total_false_negative = 0
    total_false_positive = 0
    total_true_negative = 0
    current_sample_num = 0
    with torch.no_grad():
        counter = 0
        for data in test_loader:
            true_positive = 0
            false_negative = 0
            false_positive = 0
            true_negative = 0
            counter += 1

            # Forward propagation and check loss
            input_data, label = data
            output = ds_predict_model(input_data)

            loss = loss_fn(output, label)
            total_test_loss += loss
            current_sample_num += batch_size

            label_list = label.tolist()
            output_list = output.argmax(1).tolist()
            for i in range(len(label_list)):
                if (label_list[i] == 1) and (output_list[i] == 1):
                    true_positive += 1
                elif (label_list[i] == 1) and (output_list[i] == 0):
                    false_negative += 1
                elif (label_list[i] == 0) and (output_list[i] == 1):
                    false_positive += 1
                elif (label_list[i] == 0) and (output_list[i] == 0):
                    true_negative += 1
            total_true_positive += true_positive
            total_false_negative += false_negative
            total_false_positive += false_positive
            total_true_negative += true_negative

            #accuracy = (output.argmax(1) == label).sum()
            #if counter == 2000:
            #    break

    # Get test loss and accuracy
    #accuracy_percentage = total_accuracy / current_sample_num
    TPR = total_true_positive / (total_true_positive + total_false_negative)
    TNR = total_true_negative / (total_true_negative + total_false_positive)
    print ('Total test loss in current epoch: {}'.format(total_test_loss))
    #print ('Total accuracy in current epoch: {}'.format(total_accuracy))
    #print ('Total accuracy in current epoch: {}'.format(accuracy_percentage))
    print ('TPR in current epoch: {}'.format(TPR))
    print ('TNR in current epoch: {}'.format(TNR))

    writer.add_scalar('Test loss', total_test_loss, total_test_step)
    #writer.add_scalar('Test accuracy', total_accuracy, total_test_step)
    #writer.add_scalar('Accuracy', accuracy_percentage, total_test_step)
    writer.add_scalar('TPR', TNR, total_test_step)
    writer.add_scalar('TNR', TNR, total_test_step)
    total_test_step += 1

    torch.save(ds_predict_model, 'ds_predict_model_{}.pth'.format(step))

writer.close()