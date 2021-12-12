import random
import torch
import pandas as pd
import numpy as np
from torch import nn
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from torch.nn.parameter import Parameter

def normalize(x):
    return np.dot(np.diag(x.sum(1)**-1),x)
    
def parse_data(content,cites):
    features = content.values[:,1:-1].astype(float)
    features = normalize(features)
    orin_ids = content[0]
    orin_label_types = set(content.values[:-1,-1])
    table = {}
    label_table = {}
    for index,id_ in enumerate(orin_ids):
        table[id_] = index
    for index,id_ in enumerate(orin_label_types):
        label_table[id_] = index
    label_ids = [label_table[i] for i in content.values[:,-1]]
    ids = [table[i] for i in content.values[:,0]]
    num_node = len(ids)
    edges = np.zeros((num_node,num_node))
    source_list = [table[i] for i in cites.to_dict('list')[1]]
    target_list = [table[i] for i in cites.to_dict('list')[0]]
    edges[target_list,source_list] = 1
    edges[source_list,target_list] = 1
    edges = normalize(edges+np.eye(len(edges)))
    return features,edges,label_ids


class Block(torch.nn.Module):
    def __init__(self,in_num,hidden_num,dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.weight = Parameter(torch.FloatTensor(in_num, hidden_num))
        self.bias = Parameter(torch.FloatTensor(hidden_num))
        torch.nn.init.kaiming_uniform_(self.weight,nonlinearity='relu')
        self.bias.data.fill_(0)
    
    def encode(self,x):
        return torch.mm(x,self.weight)

    def aggregate(self,features,edges):
        x = torch.mm(edges,features)
        return x

    def forward(self,features,edges=None):
        x = self.encode(self.dropout(features))
        x = self.aggregate(x,edges)+self.bias
        return x

class GNN(torch.nn.Module):
    def __init__(self,in_num,hidden_num,out_num,dropout_rate):
        super().__init__()
        self.net_1 = Block(in_num,hidden_num,dropout_rate)
        self.net_2 = Block(hidden_num,out_num,dropout_rate)
        self.act = nn.LeakyReLU(0.01)

    def forward(self,features,edges):
        x = self.act(self.net_1(features,edges))
        x = self.net_2(x,edges)
        return x

def read_data():
    content = pd.read_csv('cora.content',header=None,sep='\t')
    cites = pd.read_csv('cora.cites',header=None,sep='\t')
    features,edges,label_ids = parse_data(content,cites)
    indice = np.arange(features.shape[0])
    random.seed(0)
    random.shuffle(indice)
    train_num = int(round(len(indice)*0.7))
    test_num = int(round(len(indice)*0.1))
    train_indice = indice[:train_num]
    val_indice = indice[train_num:-test_num]
    test_indice = indice[-test_num:]
    features = torch.FloatTensor(features)
    edges = torch.FloatTensor(edges)
    label_ids = torch.LongTensor(label_ids)
    return features,edges,label_ids,train_indice,val_indice,test_indice
    
def main():
    features,edges,label_ids,train_indice,val_indice,test_indice = read_data()
    net = GNN(features.shape[1],32,7,0.1)
    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(net.parameters(),1e-2,weight_decay=1)
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    best_val_loss = None
    counter = 0
    train_label_ids = label_ids[train_indice]
    val_label_ids = label_ids[val_indice]
    for i in range(500):
        net.train(True)
        optim.zero_grad()
        output = net(features,edges)
        train_loss = loss_func(output[train_indice],train_label_ids)
        train_loss.backward()
        optim.step()
        train_losses.append(train_loss.cpu().detach().numpy())
        net.train(False)
        val_loss = loss_func(output[val_indice],val_label_ids)
        val_loss_value = val_loss.cpu().detach().numpy()
        val_losses.append(val_loss_value)
        train_f1 = f1_score(train_label_ids,output.detach().numpy()[train_indice].argmax(1),average='macro')
        val_f1 = f1_score(val_label_ids,output.detach().numpy()[val_indice].argmax(1),average='macro')
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        if best_val_loss is None or best_val_loss >= val_loss_value:
            best_val_loss = val_loss_value
            counter = 0
        counter += 1
        if counter >= 10:
            break
    test_f1 = f1_score(label_ids[test_indice],output.detach().numpy()[test_indice].argmax(1),average='macro')
    test_loss = loss_func(output[test_indice],label_ids[test_indice])
    confusion_matrix(label_ids[test_indice],output.detach().numpy()[test_indice].argmax(1))
    metrics = {
        "train_f1":train_f1s,"val_f1":val_f1s,"test_f1":test_f1,
        "train_loss":train_losses,"val_loss":val_losses,"test_loss":test_loss
    }
    return net,metrics
                
