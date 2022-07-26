%%file main_amp.py
import os
import time

import argparse
from typing import Tuple
from tqdm import tqdm

import  matplotlib.pyplot as plt

import torch
from torch import nn, optim

from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def save_checkpoint(model, optimizer,args, epoch, total_iter, loss):    
    import os
    import shutil    
    
    state = {
        'args'           : args,      
        'model_state'    : model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch'          : epoch,
        'total_iter'     : total_iter,
        'loss'           : loss
        }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    file_name = 'ch_{:04d}.pt'.format(epoch)
    filepath = os.path.join(args.output_dir, file_name  )
    torch.save(state, filepath)   
    print(" save ", file_name, end='' )
    last_chkpt_filepath = os.path.join(args.output_dir, 'ch_last.pt')
    shutil.copy(filepath, last_chkpt_filepath)


def load_checkpoint(args, model, optimizer, start_epoch, start_iter):
    import os
    path = os.path.join(args.output_dir, 'ch_last.pt')
    #dst = f'cuda:{torch.cuda.current_device()}'
    dst='cuda'
    checkpoint = torch.load(path, map_location=dst)
   
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    val_loss     = checkpoint['loss']    
    start_epoch[0] = checkpoint['epoch'] + 1
    start_iter[0] = checkpoint['total_iter']
    print("\nDEBUG : load {} and resume epoch {:d} total iter : {:d} with loss {:8.4f}".format(path, start_epoch[0], start_iter[0]+1, val_loss  ) )

def create_model( im_size=28 ,  num_hidden1 = 128, num_hidden2 = 256, num_hidden3 = 128,  num_cls=10):
    # create model architecture
    from torch import nn 

    model = nn.Sequential(
        nn.Linear(im_size*im_size, num_hidden1),  # MNIST images are 28x28 pixels
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(num_hidden1, num_hidden2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(num_hidden2, num_hidden3),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(num_hidden3, num_cls, bias=False)  # 10 classes to predict
    )
    return model

def create_data_loaders( batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_loc = './mnist_data'

    train_dataset = datasets.FashionMNIST(dataset_loc,
                                   download=True,
                                   train=True,
                                   transform=transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,  # This is mandatory to set this to False here, shuffling is done by Sampler
                              num_workers=0,
                              sampler=None,
                              pin_memory=True)

    # This is not necessary to use distributed sampler for the test or validation sets.
    test_dataset = datasets.FashionMNIST(dataset_loc,
                                  download=True,
                                  train=False,
                                  transform=transform)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=True)

    return train_loader, test_loader

def train_epoch(epoch, total_iter, model, optimizer, scaler,  criterion, train_loader, args, device ):
    model.train()
    epoch_train_loss=0
    #pbar = tqdm(train_loader)
    tic_epoch = time.time() 
    for iter,  batch  in enumerate(train_loader):
        tic_iter = time.time()
        x, y = batch 
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x = x.view(x.shape[0], -1)
        optimizer.zero_grad()

        if args.fp16: 
            amp_on = True
        else:
            amp_on =False 

        with torch.cuda.amp.autocast(amp_on):
            y_hat = model(x)  ### TODO amp with scaler
            batch_loss = criterion(y_hat, y) ## TODO amp with scaler


        if args.fp16:
            scaler.scale(batch_loss).backward() ## TODO amp scaler gradient clipping  
        else:
            batch_loss.backward()
        if args.fp16:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        else :
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)


        if args.fp16:
            scaler.step(optimizer)
            scaler.update() 
        else:
            optimizer.step()

        batch_train_loss_scalar = batch_loss.item()
        epoch_train_loss += batch_train_loss_scalar / x.shape[0]
        total_iter +=1
        toc_iter = time.time()
        dur_iter = toc_iter - tic_iter 
        str_train = "\n ep:{:>3d}/{:>3d} iter:{:>8d} {:>3d}/{:3d}  train_loss :{:8.4f}   dur: {:>4.2f}ms/iter".format(
             epoch, args.epochs,  total_iter,  iter, len(train_loader), batch_train_loss_scalar,  dur_iter*1000 )
        if iter % args.print_iter  ==0:
            print(str_train, end='' )
        #pbar.set_description(" {}  loss : batch:{:8.4f}  epoch:{:8.4f}".format( epoch, batch_train_loss_scalar, epoch_train_loss ) )
    toc_epoch = time.time()
    dur_epoch = toc_epoch - tic_epoch 
    print(" | train_epoch loss : {:8.4f} {:4.2f}s/epoch".format(epoch_train_loss, dur_epoch ), end ='' )
    return model, epoch_train_loss, total_iter



def eval(epoch, model, test_loader, criterion,args, device):
    # calculate validation loss
    with torch.no_grad():
        model.eval()
        epoch_val_loss = 0
        #pbar = tqdm(test_loader)
        for iter, batch in enumerate(test_loader):
            x, y = batch 
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = x.view(x.shape[0], -1)
            y_hat = model(x)
            batch_loss = criterion(y_hat, y)
            batch_val_loss_scalar = batch_loss.item()

            epoch_val_loss += batch_val_loss_scalar / x.shape[0]

        str_eval = " | {} eval_loss :{:8.4f}    ".format(
                 len(test_loader), epoch_val_loss  )
        print(str_eval, end='' )
    return epoch_val_loss

def train(args, model, optimizer, scaler,  criterion, train_loader, test_loader,  start_epoch):
    # train the model
    total_iter = 0
    history_loss =[]
    for epoch in range(start_epoch, args.epochs+1) :
        model.train()
        epoch_loss = 0
        device='cuda'

        model, epoch_loss, total_iter = train_epoch(epoch, total_iter, model, optimizer, scaler,  criterion, train_loader, args, device )
        history_loss.append(epoch_loss)
        if epoch % args.eval_epoch ==0 :
            val_loss = eval(epoch, model, test_loader,criterion, args, device)
        if epoch % args.save_epoch ==0 : 
            save_checkpoint(model, optimizer,args, epoch, total_iter, epoch_loss)


    plt.plot(history_loss)
    plt.savefig("loss.png")
    return model

def main(args):        


    model=create_model(num_hidden1 = 1024, num_hidden2 = 1024, num_hidden3 = 1024,)

    device = torch.device('cuda')
    model = model.to(device)

    # initialize optimizer and loss function
    #optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), #Adam optimizer
                                    args.lr,
                                    betas=(0.9, 0.99),
                                    weight_decay=args.weight_decay) 


    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = create_data_loaders(args.batch_size)


    #load checkpoint 
    val_loss=0
    start_epoch = [1]
    start_iter  = [0]
    
    if args.resume:
        load_checkpoint(args, model, optimizer, start_epoch, start_iter)
    start_epoch = start_epoch[0]
    total_iter = start_iter[0]    
    print(start_epoch,total_iter , val_loss)

    # TODO amp scaler 
    if args.fp16 :
        scaler = torch.cuda.amp.GradScaler() # amp on 
    else :
        scaler=None    

    model = train(args,
                 model, optimizer, scaler ,criterion,
                 train_loader,
                 test_loader,
                 start_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('-o', '--output-dir', type=str, default='./', help='Directory to save checkpoints')
    parser.add_argument(      '--sample-dir', type=str, default='./', help='Directory to save sample') 


    parser.add_argument('--step', type=int, default=20)
    parser.add_argument('--print-iter',  default=10, type=int,
                        metavar='W', help='print step')
    parser.add_argument('--eval-epoch',  default=2, type=int,
                        metavar='W', help='print step')

    parser.add_argument('--save-epoch',  default=2, type=int,
                        metavar='W', help='print step')

    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--clip', default=0.25, type=float,
                     help='gradient clip threshold ')
    
    parser.add_argument('--local_rank',type=int, default=os.getenv('LOCAL_RANK', 0),  help='Rank of the process for multiproc. Do not set manually.')
    parser.add_argument('--rank',type=int, default=os.getenv('RANK', 0),  help='Rank of the process for multiproc. Do not set manually.')
    parser.add_argument('--world_size',type=int, default=os.getenv('WORLD_SIZE', 1),  help='Number of processes for multiproc. Do not set manually.')  
    parser.add_argument('--seed',type=int, default=1234,help='Seed for PyTorch random number generators')
    parser.add_argument('--multi_gpu',type=str,default='ddp',choices=['ddp', 'dp'],help='Use multiple GPU')

    #chckpoint
    parser.add_argument('--fp16', action='store_true',help='Run training in fp16/mixed precision')
    parser.add_argument('--amp',type=str,default='pytorch', choices=['apex', 'pytorch'], help='Implementation of automatic mixed precision')

    parser.add_argument('--resume',action='store_true',help='Resume training from the last available checkpoint') 
    parser.add_argument('--checkpoint-path',  type=str, default=None,     help='Checkpoint path to resume training')
    args = parser.parse_args()
    
    print(args)
    main(args)
