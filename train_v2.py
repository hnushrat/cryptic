import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold

from datasets import Dataset

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from torchinfo import summary

from transformers import AutoTokenizer, AutoModel

import random

from transform_data import return_data_v2

########################
seed = 42
print(f"\n{'*'*20}\nUsing seed: {seed}\n{'*'*20}\n")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
########################

dataset_name = "TrainingData"

df = return_data_v2(f"{dataset_name}.csv")

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
LEARNING_RATE = 1e-5
MAX_LENGTH = 64 + 2
LOG_FILE = "log.txt"
EPOCHS = 20


name = "bert-base-uncased"
model_name = "bert"
num_classes = 1

tokenizer = AutoTokenizer.from_pretrained(name)

class bert(nn.Module):
    def __init__(self):
        super(bert, self).__init__()
        self.bert = AutoModel.from_pretrained(name)
        self.linear1 = nn.Linear(768, num_classes)

    def forward(self, input_ids = None, attention_mask = None):
        
        op = self.bert(input_ids = input_ids, attention_mask = attention_mask)# sequence_output has the following shape: (batch_size, sequence_length, 768)

        
        linear1_output = self.linear1(op[0][:, 0, :])

        return linear1_output


class CreateDataset(torch.utils.data.Dataset):
    '''
    Initiating Variables
    df: the training dataframe
    source_column : the name of source text column in the dataframe
    target_columns : the name of target column in the dataframe
    '''
    
    def __init__(self, df):
    
        self.df = df        
        self.source_texts = self.df.iloc[:, 1].values
        self.target = self.df.iloc[:, 0].values
        
        
    def __len__(self):
        return len(self.df)
    '''
    __getitem__ runs on 1 example at a time. Here, we get an example at index and return its numericalize source and
    target values we created in __init__
    '''
    def __getitem__(self, index):
        source_text = self.source_texts[index]
        target_text = self.target[index]
        
        temp = tokenizer.encode_plus(source_text, 
                                    truncation = True, padding = "max_length", max_length = MAX_LENGTH,
                                    return_tensors = 'pt')

        return temp["input_ids"], temp["attention_mask"], torch.tensor(target_text, dtype = torch.float)



def accuracy_fn(ytrue, ypred):
    correct = torch.eq(ytrue, ypred).sum().item()
    return (correct/len(ypred))*100 

train_df, test_df = train_test_split(df, test_size = 0.2, stratify = df["class"].values)

train_df = CreateDataset(train_df)
test_df = CreateDataset(test_df)

best_acc = -1
train_acc = -1

train_dataloader = DataLoader(train_df, batch_size = BATCH_SIZE)
test_dataloader = DataLoader(test_df, batch_size = BATCH_SIZE)


model = bert()

model = nn.DataParallel(model)

model.to(device)
print(f"Model: {name}")
print(f"\n\nModel Loaded to device: {device}\n\n")

for param in model.parameters(): # setting the parameters to be trainable
    # print(param)
    param.requires_grad = True

# print(summary(model))

lossfn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(lr = LEARNING_RATE, params = model.parameters())
scheduler = ExponentialLR(optimizer, gamma=0.9, verbose = True)

print(f"\n**Model Initialized**\n")

print(f"***Training Started***\n\n")
print(f"Max Sequence Length: {MAX_LENGTH}\nBatch Size: {BATCH_SIZE}\n")


with open(LOG_FILE, "w") as f:
    f.writelines(f"Model: {name}\n\nModel Loaded to device: {device}\n\n**Model Initialized**\n\
***Training Started***\n\n\
Max Sequence Length: {MAX_LENGTH}\nBatch Size: {BATCH_SIZE}\n")

for epoch in range(EPOCHS):
    
    train_loss = 0
    acc = 0
    
    model.train()
    for batch, (input_ids, attention_mask, label) in enumerate(tqdm(train_dataloader)):
        # print(input_ids, attention_mask)
        
        pred = model(input_ids.squeeze().to(device), attention_mask.squeeze().to(device))
        pred = pred.squeeze()
        # print(pred, label)
        # print(torch.max(pred))

        loss = lossfn(pred, label.to(device))

        train_loss+=loss
        
        acc+=accuracy_fn(label.to(device), torch.round(torch.sigmoid(pred)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

    train_loss/=len(train_dataloader) # over all the samples
    acc/=len(train_dataloader)
    
    model.eval()

    test_loss = 0
    test_acc = 0
    with torch.inference_mode():
        for batch, (input_ids_test, attention_mask_test, label_test) in enumerate(tqdm(test_dataloader)):
            
            ypred = model(input_ids_test.squeeze().to(device), attention_mask_test.squeeze().to(device))
            ypred = ypred.squeeze()

            # print("Test : ", y.shape)
            test_loss+=lossfn(ypred, label_test.to(device))
            test_acc+=accuracy_fn(label_test.to(device), torch.round(torch.sigmoid(ypred)))

        test_loss/=len(test_dataloader)
        test_acc/=len(test_dataloader)
    
    scheduler.step()

    if acc > train_acc:
        train_acc = acc

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model, f"{model_name}_{dataset_name}_epoch_{epoch+1}_train_acc_{train_acc:.3f}_test_acc_{best_acc:.3f}.pth")

    print(f"\nEnd of epoch: {epoch+1}\n \t train loss: {train_loss:.3f}\t train_acc: {acc:.3f}\t test loss: {test_loss:.3f}\t test acc: {test_acc:.3f}\n")
    with open(f"{LOG_FILE}", 'a') as f:
        f.writelines(f"\nEnd of epoch: {epoch+1}\n \t train loss: {train_loss:.3f}\t train_acc: {acc:.3f}\t test loss: {test_loss:.3f}\t test acc: {test_acc:.3f}\n")
    f.close()