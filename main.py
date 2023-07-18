import os, sys
import numpy as np
import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.pfcl import PFCL

from utils.data_utils import get_dataset, get_backbone, filter_classes, get_split
from utils.data_utils import get_transform, transform_resize, progress_bar
from utils.loggers import CsvLogger    
from dataset.aux_dataset import AuxDataset
    
def get_args_parser():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'pfcl')
    parser.add_argument('--results', type = str, default = 'ser')
    parser.add_argument('--root', type = str, default = '/root/data/dataset')
    parser.add_argument('--dataset', type = str, default = 'cifar10')
    parser.add_argument('--aux_dataset', type = str, default = 'caltech256')  
    parser.add_argument('--n_classes', type = int, default = 10)
    parser.add_argument('--n_tasks', type = int, default = 3)
    parser.add_argument('--buffer_size', type = int, default = 0)

    parser.add_argument('--lr', type = float, default = 0.03)
    parser.add_argument('--n_epochs', type = int, default = 50)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--alpha', type = float, default = 1)
    parser.add_argument('--beta', type = float, default = 1)
    parser.add_argument('--n_batches', type = int, default = 0)

    parser.add_argument('--device_id', type = int, default = 0)
    parser.add_argument('--seed', type = int, default = 123)
    
    return parser

def train(cl_model, train_loader, t, args):
    cl_model.net.train()
    
    cl_model.optim = torch.optim.SGD(cl_model.net.parameters(), lr=args.lr)

    scheduler = None
    if args.dataset == 'cifar10':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(cl_model.optim, [35, 45], gamma=0.1)    
    if args.dataset == 'cifar100':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(cl_model.optim, [35, 45], gamma=0.1)
    if args.dataset == 'tinyimg':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(cl_model.optim, [70, 90], gamma=0.1)

    args.stop_kd = False
    for epoch in range(args.n_epochs):            
        for batch_idx, data in enumerate(train_loader):
            if epoch == args.n_epochs-1 and batch_idx >= len(train_loader)-args.n_batches:
                args.stop_kd = True   

            img, label, aux_img = data
            img, label, aux_img = img.to(args.device), label.to(args.device), aux_img.to(args.device)

            loss = cl_model.observe(img, label, aux_img)
            progress_bar(batch_idx, len(train_loader), t+1, epoch+1, loss.item())

        if scheduler is not None:
            scheduler.step()
                        
def evaluate(cl_model, test_splits, setting, args):
    cl_model.net.eval()    

    accs, accs_task = [], []    
    for t in range(len(test_splits)):
        metrics = {'correct': [], 'count': []}
        
        if setting == 'class_il':            
            metrics_task = {'correct': [], 'count': []}

        test_loader = DataLoader(test_splits[t], batch_size=args.batch_size, shuffle=False)

        for batch_idx, data in enumerate(test_loader):
            img, label, _ = data
            img, label = img.to(args.device), label.to(args.device)

            with torch.no_grad():   
                predict = cl_model.net(img)    

            pred = torch.max(predict, 1)[1]
            correct = pred.eq(label.data).cpu().sum().numpy()

            metrics['correct'].append(correct)
            metrics['count'].append(label.size(0))
                        
            if setting == 'class_il':            
                predict = filter_classes(predict, t, args.n_classes, args.n_tasks)                
                pred = torch.max(predict, 1)[1]
                correct = pred.eq(label.data).cpu().sum().numpy()

                metrics_task['correct'].append(correct)
                metrics_task['count'].append(label.size(0))
                        
        accs.append(100 * np.sum(metrics['correct']) / np.sum(metrics['count']))        
        if setting == 'class_il':            
            accs_task.append(100 * np.sum(metrics_task['correct']) / np.sum(metrics_task['count']))
    
    return accs, accs_task 


def main(args):    
    if torch.cuda.is_available():
        args.device = torch.device('cuda', args.device_id)
        torch.cuda.manual_seed(args.seed)    
    else:
        args.device = torch.device('cpu')    

    aux_dataset = None   
            
    tf_resize = transform_resize(args.dataset)
        
    print (f'model: {args.model}, dataset: {args.dataset}({args.n_tasks} tasks)')
    print (f'alpha: {args.alpha}, beta: {args.beta}, buffer_size: {args.buffer_size}')
    
    if args.aux_dataset in ['caltech256', 'flowers102','xpie_s', 'xpie_n', 'xpie', 'all']:    
        aux_dataset = AuxDataset(args.root, args.aux_dataset, tf_resize)      
        print (f'aux_dataset: {args.aux_dataset}({aux_dataset.length} images)')   
    else:
        args.aux_dataset = 'none'
        print (f'aux_dataset: none')    
                              

    save_log_name = os.path.join(args.model, 'rss_log_{:s}_{:d}_{:d}.txt'.format(args.dataset, args.n_tasks, args.n_batches))     
    with open(save_log_name, 'a') as f:
        f.write(f'\n\n model: {args.model}, dataset: {args.dataset}({args.n_tasks} tasks)')
        f.write(f'\n alpha: {args.alpha}, beta: {args.beta}, aux_dataset: {args.aux_dataset}({aux_dataset.length} images)')
    f.close() 
    
    if args.dataset in ['perm-mnist', 'rot-mnist']:
        setting = 'domain_il'
        
    elif args.dataset in ['cifar10', 'cifar100', 'tinyimg']:
        setting = 'class_il'
        train_set = get_dataset(args.root, args.dataset, train=True, transform=transforms.ToTensor())
        train_set.aux_dataset = aux_dataset        
        
        tf_test = get_transform(args.dataset, train=False)        
        test_set = get_dataset(args.root, args.dataset, train=False, transform=tf_test)           

    net = get_backbone(args.dataset, args.n_classes).to(args.device)     
        
    args.setting = setting
    
    if args.model == 'pfcl':
        cl_model = PFCL(net, args)     
        
    results, results_task = [], []                
    test_splits = []
    
    # data augmentation is used in continual model
    for t in range(args.n_tasks):                   
        if args.model == 'pfcl':
            transform = get_transform(args.dataset, train=True, mode='strong') 
            args.transform = transforms.Compose([transforms.ToPILImage(), transform])                 
                
        if setting == 'class_il':            
            train_split = get_split(train_set, t, args.n_classes, args.n_tasks)
            test_split = get_split(test_set, t, args.n_classes, args.n_tasks)
            
        elif setting == 'domain_il':                
            # for Domain-IL, use a same transform for both train and test
            train_split = get_dataset(args.root, args.dataset, train=True, transform=transforms.ToTensor())
            test_split = get_dataset(args.root, args.dataset, train=False, transform=transform)               
            train_split.aux_dataset = aux_dataset
            
        test_splits.append(test_split)
        train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True)                                
            
        train(cl_model, train_loader, t, args)    

            
        accs = evaluate(cl_model, test_splits, setting, args)
        
        if hasattr(cl_model, 'end_task'):
            cl_model.end_task()
        
        results.append(accs[0])                
                
        if setting == 'class_il':  
            results_task.append(accs[1])        
            print ('\nclass-il: ', [float('{:.02f}'.format(d)) for d in accs[0]], round(np.mean(accs[0]), 2))
            print ('task-il : ', [float('{:.02f}'.format(d)) for d in accs[1]], round(np.mean(accs[1]), 2))  
            
            with open(save_log_name, 'a') as f:
                f.write(f'\nTask {t+1}, class-il: [' + ', '.join(map(str, accs[0])) + '], ' + str(round(np.mean(accs[0]), 2)))
                f.write(f'\nTask {t+1}, task-il: [' + ', '.join(map(str, accs[1])) + '], ' + str(round(np.mean(accs[1]), 2)))
            f.close()    
            
        else:
            print ('\ndomain-il: ', [float('{:.02f}'.format(d)) for d in accs[0]], round(np.mean(accs[0]), 2))
            with open(save_log_name, 'a') as f:
                f.write(f'\nTask {t+1}, domain-il: [' + ', '.join(map(str, accs[0])) + '], ' + str(round(np.mean(accs[0]), 2)))
            f.close()    

        save_name = os.path.join(save_path, f'{args.aux_dataset}_{t}.pt')
        torch.save(cl_model.net.state_dict(), save_name)
                
    
    # accuracy on all tasks
    if setting == 'class_il':          
        for i in range(len(results)):
            accs_class = results[i]
            accs_task = results_task[i]     
    elif setting == 'domain_il':
        for i in range(len(results)):
            accs = results[i]                 
            
    csv_logger = CsvLogger(args.model)    
    csv_logger.write(setting, results, args)
    
    # task_il results when setting is class_il
    if setting == 'class_il':          
        csv_logger.write('task-il', results_task, args)    

if __name__ == '__main__':
    args = get_args_parser().parse_args()    
    save_path = os.path.join('ckpt', f'{args.model}_{args.dataset}_{args.n_tasks}')        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    main(args)        
    print()
