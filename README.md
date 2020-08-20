
## Table of Contents (Optional)

- [Data collection](#Data_collection)
- [Understanding the data](#Understanding_data)
- [Data Cleaning](#Data_Cleaning)
- [Model Architecture](#Model_Architecture)

---


## Installation
### from conda
- `conda install -c anaconda tensorflow-gpu` 
### from pip
- `pip install tensorflow-gpu` 

## Data_collection
- the data consist of human and dog images downloaded from 2 diffrent locations 
---

## Data collection
- <a href="https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip" target="_blank">Dog Data Set</a> .
- <a href="https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip" target="_blank">Human Data Set</a> .

## Understanding_data
- data sets consist Dog and Humans images 
- our task to create a Neural Network which diffrentiate them 

## Data_Cleaning
- Load the dataset using glob method
- Detect Humans using Haar feature-based cascade classifiers
- write Dogs Detectector function using vgg16 model
- Specify Data Loaders for the Dog Dataset

### Model_Architecture
```
# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)      
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
       
        self.pool = nn.MaxPool2d(2, 2)
        
        # fully-connected
        self.fc1 = nn.Linear(7*7*128, 512)
        self.fc2 = nn.Linear(512, 133) 
        
        # drop-out
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        ## Define forward behavior
        x = F.relu(self.conv1(x)) 
        x = self.pool(x) 
        x = F.relu(self.conv2(x)) 
        x = self.pool(x) 
        x = F.relu(self.conv3(x)) 
        x = self.pool(x) 
        x = x.view(x.size(0), -1) 
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```
### Train the model , validate it and test it
### Run app shows your prediction