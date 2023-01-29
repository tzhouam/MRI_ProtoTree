base_architecture_to_features = ['resnet18','resnet34','resnet50','resnet50_inat','resnet101','resnet152',
                                 'densenet121','densenet161','densenet169','densenet201',
                                 'vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn']
class Hyper:
    def __init__(self):
        self.epochs=99
        self.log_dir='./log'
        self.dataset='Duke' #Duke->tumor patch and common tissue patch, Duke_full -> Full image
        self.batch_size=100
        self.class_names =['lumil-like', 'ER/PR pos, HER2 pos', 'her2', 'trip neg']
        self.attribute='Mol Subtype'
        self.classes=len(self.class_names)
        self.lr=1e-3
        self.lr_block=0.001
        self.lr_net=1e-3
        self.num_features=256
        self.depth=9
        self.net='resnet34'
        self.freeze_epochs=0
        self.milestones=[60,70,80,90,100]
        self.optimizer='AdamW'
        self.lr_pi=0.001
        self.momentum=0.9
        self.weight_decay=0

        self.cuda = 1
        self.cuda_baseline = 1

        self.W1=1
        self.H1=1
        self.D1=1
        self.gamma=0.5
        self.state_dict_dir_net=''
        self.state_dict_dir_tree=''
        self.dir_for_saving_images='upsampling_results'
        self.upsample_threshold=0.98
        self.pruning_threshold_leaves = 0.01
        self.nr_trees_ensemble = 5

        self.worker = 20
        self.disable_pretrained=True
        self.disable_cuda=False
        self.disable_derivative_free_leaf_optim=False
        self.kontschieder_train=False
        self.kontschieder_normalization=False
        self.log_probabilities=False
