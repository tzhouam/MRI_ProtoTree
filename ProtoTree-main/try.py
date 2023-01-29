import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_iris
from prototree.prototree import ProtoTree
from util.hyper import Hyper
from util.net import get_network, freeze
from torchsummary import summary

device=torch.device('cuda:'+str(Hyper().cuda))
args=Hyper()
features_net, add_on_layers = get_network(3, args)
tree = ProtoTree(num_classes=4,
                    feature_net = features_net,
                    args = args,
                    add_on_layers = add_on_layers).to(device)
summary(tree,input_size=(3,100,100,100))