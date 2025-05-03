import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

class Emotion_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Emotion_Classifier, self),__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)  #input -> hidden
        self.relu = nn.ReLU()                             #ReLU is activation function
        self.layer2 = nn.Linear(hidden_size, num_classes) #hidden -> output

    def forward_pass(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

#number of neurons in each layer
input_size = len(features)
hidden_size = 128
num_classes = 4
num_epochs = 30



