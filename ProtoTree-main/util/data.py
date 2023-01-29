
import numpy as np
import argparse
import os
import torch
import torch.optim
import torch.utils.data
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose, Lambda
import torchvision.transforms as transforms
from .dataset import Dataset
from util.hyper import Hyper
def get_data(args:Hyper):
    """
    Load the proper dataset based on the parsed arguments
    :param args: The arguments in which is specified which dataset should be used
    :return: a 5-tuple consisting of:
                - The train data set
                - The project data set (usually train data set without augmentation)
                - The test data set
                - a tuple containing all possible class labels
                - a tuple containing the shape (depth, width, height) of the input images
    """
    if args.dataset =='Duke':
        return get_duke(True, '/jhcnas1/zhoutaichang/prototree/train/', '/jhcnas1/zhoutaichang/prototree/original/',
                        '/jhcnas1/zhoutaichang/prototree/test/')
    if args.dataset== 'Duke_full':
        return get_duke(True, '/jhcnas1/zhoutaichang/prototree_full_image/train/', '/jhcnas1/zhoutaichang/prototree_full_image/original/',
                        '/jhcnas1/zhoutaichang/prototree_full_image/test/')
    # if args.dataset == 'CARS':
    #     return get_cars(True, './data/cars/dataset/train', './data/cars/dataset/train', './data/cars/dataset/test')
    raise Exception(f'Could not load data set "{args.dataset}"!')

def get_dataloaders(args):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, projectset, testset, classes, shape  = get_data(args)
    c, w, h, d = shape
    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=cuda
                                              )
    projectloader = torch.utils.data.DataLoader(projectset,
                                            #    batch_size=args.batch_size,
                                              batch_size=int(args.batch_size/4), #make batch size smaller to prevent out of memory errors during projection
                                              shuffle=False,
                                              pin_memory=cuda
                                              )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=cuda
                                             )
    print("Num classes (k) = ", len(classes), flush=True)
    return trainloader, projectloader, testloader, classes, c


def get_duke(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size = 100):
    shape = (3, img_size, img_size,img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    # transform_no_augment = transforms.Compose([
    #                         transforms.Resize(size=(img_size, img_size,img_size)),
    #                         # transforms.ToTensor(),
    #                         normalize
    #                     ])
    # if augment:
    #     transform = transforms.Compose([
    #         transforms.Resize(size=(img_size, img_size,img_size)),
    #         transforms.RandomOrder([
    #         transforms.RandomPerspective(distortion_scale=0.2, p = 0.5),
    #         transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.02,0.02)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomAffine(degrees=10, shear=(-2,2),translate=[0.05,0.05]),
    #         ]),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # else:
    #     transform = transform_no_augment

    trainset = Dataset(train_dir)
    projectset = Dataset(project_dir)
    testset = Dataset(test_dir)
    # classes = ['tumor','common']
    classes=Hyper().class_names

    # trainset = torchvision.datasets.ImageFolder(train_dir)
    # projectset = torchvision.datasets.ImageFolder(project_dir)
    # testset = torchvision.datasets.ImageFolder(test_dir)
    # classes = trainset.classes

    return trainset, projectset, testset, classes, shape




