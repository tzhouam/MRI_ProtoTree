from prototree.prototree import ProtoTree
from util.log import Log

from util.args import get_args, save_args, get_optimizer
from util.data import get_dataloaders
from util.init import init_tree
from util.net import get_network, freeze
from util.visualize import gen_vis
from util.analyse import *
from util.save import *

# from prototree.test import eval, eval_fidelity
from prototree.prune import prune
from prototree.project import project, project_with_class_constraints
from prototree.upsample import upsample
from baseline.baseline import Baseline
import torch
from shutil import copy
from copy import deepcopy
from util.hyper import Hyper
from torchmetrics import AUROC
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
args = Hyper()

import warnings

warnings.filterwarnings("ignore", category=Warning)  # 过滤报警信息


def run_tree():
    # Create a logger
    log = Log('./Baseline_log')
    print("Log dir: ", './Baseline_log', flush=True)
    # Create a csv log for storing the test accuracy, mean train accuracy and mean loss for each epoch
    log.create_log('log_epoch_overview', 'epoch', 'test_acc', 'test_auc', 'mean_train_acc', 'mean_train_auc',
                   'mean_train_crossentropy_loss_during_epoch')
    # Log the run arguments
    save_args(args, log.metadata_dir)
    if torch.cuda.is_available():
        # device = torch.device('cuda')
        device = torch.device('cuda:{}'.format(args.cuda_baseline))
    else:
        device = torch.device('cpu')

    # Log which device was actually used
    log.log_message('Device used: ' + str(device))

    # Create a log for logging the loss values
    log_prefix = 'log_train_epochs'
    log_loss = log_prefix + '_losses'
    log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc', 'batch_train_auc')

    # Obtain the dataset and dataloaders
    trainloader, projectloader, testloader, classes, num_channels = get_dataloaders(args)
    # Create a convolutional network based on arguments and add 1x1 conv layer
    features_net, add_on_layers = get_network(num_channels, args)
    # Create a ProtoTree
    tree = Baseline(num_classes=len(classes),
                     feature_net=features_net,
                     args=args)
    tree = tree.to(device=device)
    # Determine which optimizer should be used to update the tree parameters
    optimizer = torch.optim.AdamW(tree.parameters(),lr=args.lr_net,eps=1e-07, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)
    # tree, epoch = init_tree(tree, optimizer, scheduler, device, args)

    # tree.save(f'{log.checkpoint_dir}/tree_init')
    # log.log_message(
    #     "Max depth %s, so %s internal nodes and %s leaves" % (args.depth, tree.num_branches, tree.num_leaves))
    # analyse_output_shape(tree, trainloader, log, device)

    leaf_labels = dict()
    best_train_acc = 0.
    best_test_acc = 0.
    print(args.net)
    for epoch in range(1,args.epochs + 1):
        '''
            TRAIN AND EVALUATE TREE
        '''
        for epoch in range(epoch, args.epochs + 1):
            log.log_message("\nEpoch %s" % str(epoch))
            # Freeze (part of) network for some epochs if indicated in args
            # freeze(tree, epoch, params_to_freeze, params_to_train, args, log)
            # log_learning_rates(optimizer, args, log)
            # print('lr: ', scheduler._last_lr)
            # Train tree
            # if tree._kontschieder_train:
            #     train_info = train_epoch_kontschieder(tree, trainloader, optimizer, epoch,
            #                                           args.disable_derivative_free_leaf_optim, device, log, log_prefix)
            # else:
            train_info = train_epoch(tree, trainloader, optimizer, epoch, args.disable_derivative_free_leaf_optim,
                                         device, log, log_prefix)
            # save_tree(tree, optimizer, scheduler, epoch, log, args)
            # best_train_acc = save_best_train_tree(tree, optimizer, scheduler, best_train_acc,
            #                                       train_info['train_accuracy'], log)
            # leaf_labels = analyse_leafs(tree, epoch, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
            # train_info={'train_auc':0,'train_accuracy':0,'loss':0}
            # Evaluate tree
            if args.epochs > 100:
                if epoch % 10 == 0 or epoch == args.epochs:
                    eval_info = eval(tree, testloader, epoch, device, log)
                    original_test_acc = eval_info['test_accuracy']
                    best_test_acc = save_best_test_tree(tree, optimizer, scheduler, best_test_acc,
                                                        eval_info['test_accuracy'], log)
                    log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], eval_info['test_auc'],
                                   train_info['train_accuracy'], train_info['train_auc'], train_info['loss'])
                else:
                    log.log_values('log_epoch_overview', epoch, "n.a.", "n.a", train_info['train_accuracy'],
                                   train_info['train_auc'], train_info['loss'])
            else:
                eval_info = eval(tree, testloader, epoch, device, log)
                original_test_acc = eval_info['test_accuracy']
                # best_test_acc = save_best_test_tree(tree, optimizer, scheduler, best_test_acc,
                #                                     eval_info['test_accuracy'], log)
                log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], eval_info['test_auc'],
                               train_info['train_accuracy'], train_info['train_auc'], train_info['loss'])

            scheduler.step()

    # else:  # tree was loaded and not trained, so evaluate only
    #     '''
    #         EVALUATE TREE
    #     '''
    #     eval_info = eval(tree, testloader, epoch, device, log)
    #     original_test_acc = eval_info['test_accuracy']
    #     best_test_acc = save_best_test_tree(tree, optimizer, scheduler, best_test_acc, eval_info['test_accuracy'], log)
    #     log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], "n.a.", "n.a.")

    # '''
    #     EVALUATE AND ANALYSE TRAINED TREE
    # '''
    # log.log_message("Training Finished. Best training accuracy was %s, best test accuracy was %s\n" % (
    # str(best_train_acc), str(best_test_acc)))
    # trained_tree = deepcopy(tree)
    # leaf_labels = analyse_leafs(tree, epoch + 1, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    # analyse_leaf_distributions(tree, log)


    # return trained_tree.to('cpu'), pruned_tree.to('cpu'), pruned_projected_tree.to(
    #     'cpu'), original_test_acc, pruned_test_acc, pruned_projected_test_acc, project_info, eval_info_samplemax, eval_info_greedy, fidelity_info
def train_epoch(tree: Baseline,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                disable_derivative_free_leaf_optim: bool,
                device,
                log: Log = None,
                log_prefix: str = 'log_train_epochs',
                progress_prefix: str = 'Train Epoch'
                ) -> dict:

    tree.train()
    tree = tree.to(device)
    # Make sure the model is in eval mode
    tree.eval()
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.
    total_auc=0.
    # Create a log if required
    log_loss = f'{log_prefix}_losses'

    nr_batches = float(len(train_loader))
    # with torch.no_grad():
    #     _old_dist_params = dict()
    #     for leaf in tree.leaves:
    #         _old_dist_params[leaf] = leaf._dist_params.detach().clone()
    #     # Optimize class distributions in leafs
    #     eye = torch.eye(tree._num_classes).to(device)

    # Show progress on progress bar
    train_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=progress_prefix+' %s'%epoch+'/'+'%s'%Hyper().epochs,
                    ncols=0)
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs, ys) in train_iter:
        # Make sure the model is in train mode
        tree.train()
        # Reset the gradients
        optimizer.zero_grad()

        xs, ys = xs.to(device), ys.to(device)

        # Perform a forward pass through the network
        ys_pred = tree.forward(xs)

        # Learn prototypes and network with gradient descent.
        # If disable_derivative_free_leaf_optim, leaves are optimized with gradient descent as well.
        # Compute the loss
        criterion = torch.nn.CrossEntropyLoss()
        # loss = F.nll_loss(torch.log(ys_pred), ys)
        loss = F.nll_loss(F.log_softmax(ys_pred, dim=1), ys)
        # loss = criterion(ys_pred,ys)
        # criterion=torch.nn.CrossEntropyLoss()
        #
        # loss=criterion(F.softmax(ys_pred.type(torch.FloatTensor)),ys.type(torch.FloatTensor))
        # Compute the gradient
        loss.backward()
        # Update model parameters

        optimizer.step()

        # if not disable_derivative_free_leaf_optim:
        #     #Update leaves with derivate-free algorithm
        #     #Make sure the tree is in eval mode
        #     tree.eval()
        #     with torch.no_grad():
        #         target = eye[ys] #shape (batchsize, num_classes)
        #         for leaf in tree.leaves:
        #             if tree._log_probabilities:
        #                 # log version
        #                 update = torch.exp(torch.logsumexp(info['pa_tensor'][leaf.index] + leaf.distribution() + torch.log(target) - ys_pred, dim=0))
        #             else:
        #                 update = torch.sum((info['pa_tensor'][leaf.index] * leaf.distribution() * target)/ys_pred, dim=0)
        #             leaf._dist_params -= (_old_dist_params[leaf]/nr_batches)
        #             F.relu_(leaf._dist_params) #dist_params values can get slightly negative because of floating point issues. therefore, set to zero.
        #             leaf._dist_params += update

        # Count the number of correct classifications
        ys_pred=F.softmax(ys_pred,dim=1)
        ys_pred_max = torch.argmax(ys_pred, dim=1)

        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(xs))
        # AUC = AUROC(task="multiclass", num_classes=Hyper().classes)
        # auc = AUC(ys_pred,ys).item()
        try:
            if Hyper().classes > 2:
                auc = roc_auc_score(y_true=ys.tolist(), y_score=ys_pred.tolist(), multi_class='ovr', average='macro')
            else:
                auc = roc_auc_score(y_true=ys.tolist(), y_score=ys_pred[:,1].tolist())
        except ValueError:
            auc=acc
        train_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.3f}, Acc: {acc:.3f}, Auc: {auc:.3f}'
        )
        # Compute metrics over this batch
        total_loss+=loss.item()
        total_acc+=acc
        total_auc+=auc

        if log is not None:
            log.log_values(log_loss, epoch, i + 1, loss.item(), acc, auc)

    train_info['loss'] = total_loss/float(i+1)
    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['train_auc'] = total_auc / float(i + 1)
    return train_info

@torch.no_grad()
def eval(tree: Baseline,
        test_loader: DataLoader,
        epoch,
        device,
        log: Log = None,
        sampling_strategy: str = 'distributed',
        log_prefix: str = 'log_eval_epochs',
        progress_prefix: str = 'Eval Epoch'
        ) -> dict:
    tree = tree.to(device)

    # Keep an info dict about the procedure
    info = dict()
    if sampling_strategy != 'distributed':
        info['out_leaf_ix'] = []
    # Build a confusion matrix
    cm = np.zeros((Hyper().classes, Hyper().classes), dtype=int)

    # Make sure the model is in evaluation mode
    tree.eval()

    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc=progress_prefix+' %s'%epoch,
                        ncols=0)
    all_ys=[]
    all_pred=[]
    AUC = AUROC(task="multiclass", num_classes=Hyper().classes)

    # Iterate through the test set
    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device)

        # Use the model to classify this batch of input data
        out= tree.forward(xs)
        ys_pred = torch.argmax(out, dim=1)

        # Update the confusion matrix
        cm_batch = np.zeros((Hyper().classes, Hyper().classes), dtype=int)
        for y_pred, y_true in zip(ys_pred, ys):
            cm[y_true][y_pred] += 1
            cm_batch[y_true][y_pred] += 1
        for j in range(len(ys_pred)):

            # print(out[i].shape)
            all_pred.append(out[j].tolist())
            all_ys.append(ys[j].type(torch.IntTensor).tolist())
        acc = acc_from_cm(cm_batch)
        out=F.softmax(out,dim=1)

        try:
            if Hyper().classes> 2:
                auc = roc_auc_score(y_true=ys.tolist(), y_score=out.tolist(), multi_class='ovr', average='macro')
            else:
                auc = roc_auc_score(y_true=ys.tolist(), y_score=out[:,1].tolist())

        except ValueError:
            auc = acc
        test_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(test_iter)}], Acc: {acc:.3f}, Auc: {auc:.3f}'
        )

        # keep list of leaf indices where test sample ends up when deterministic routing is used.
        # if sampling_strategy != 'distributed':
        #     info['out_leaf_ix'] += test_info['out_leaf_ix']
        del out
        del ys_pred
        # del test_info

    info['confusion_matrix'] = cm
    info['test_accuracy'] = acc_from_cm(cm)
    # print(np.array(all_pred).shape,np.array(all_ys).shape)
    all_pred=torch.tensor(all_pred)
    all_pred = F.softmax(all_pred, dim=1)

    try:
        if Hyper().classes > 2:
            info['test_auc'] = roc_auc_score(y_true=all_ys, y_score=all_pred.tolist(), multi_class='ovr', average='macro')
        else:
            # all_pred = torch.argmax(all_pred, dim=1)

            info['test_auc'] = roc_auc_score(y_true=all_ys, y_score=all_pred[:,1].tolist())
    except ValueError:
        info['test_auc'] = info['test_accuracy']
    log.log_message("\nEpoch %s - Test accuracy with %s routing: "%(epoch, sampling_strategy)+str(info['test_accuracy'])+' Test AUC:'+str(info['test_auc']))
    return info

def acc_from_cm(cm: np.ndarray) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = 0
    for i in range(len(cm)):
        correct += cm[i, i]

    total = np.sum(cm)
    if total == 0:
        return 1
    else:
        return correct / total


if __name__ == '__main__':
    # args = get_args()
    run_tree()