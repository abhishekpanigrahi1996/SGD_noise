
import argparse
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as tdist
from models import alexnet, fc, lenet
from vgg import *
from resnet import *
from utils import  get_data, accuracy
from utils import get_grads, alpha_estimator, alpha_estimator2
from utils import linear_hinge_loss, get_layerWise_norms


torch.set_default_tensor_type('torch.DoubleTensor')


def eval(eval_loader, net, crit, opt, args, iteration, test=True, grad_batch=-1):

    net.eval()

    # run over both test and train set    
    total_size = 0
    total_loss = 0
    total_acc = 0
    grads = []
    outputs = []
    noise_norm = [] 
    grad_tot_size = -1

    P = 0 # num samples / batch size
    for x, y in eval_loader:
        P += 1
        # loop over dataset
        x, y = x.to(args.device), y.to(args.device)
        opt.zero_grad()
        out = net(x)
        
        outputs.append(out)

        loss = crit(out, y)
        prec = accuracy(out, y)
        bs = x.size(0)

        loss.backward()
    
        if iteration == 0 and ((grad_batch != -1 and total_size < grad_batch) or (grad_batch == -1)):
            grad = get_grads(net)
            grads.append(grad.cpu())
        elif iteration == 0 and (grad_batch != -1 and total_size > grad_batch) and grad_tot_size == -1:
            grad_tot_size = P

        total_size += int(bs)
        total_loss += float(loss) * bs
        total_acc += float(prec) * bs

    alpha = -1   
    M = -1 
    if iteration == 0:
       if grad_tot_size == -1:
          grad_tot_size = P   
       M = len(grads[0]) # total number of parameters 
       grads = torch.cat(grads).view(-1, M)
       mean_grad = grads.sum(0) / grad_tot_size
       noise_norm = (grads - mean_grad).norm(dim=1)

       N = M * grad_tot_size
       for i in range(1, 1 + int(math.sqrt(N))):
        if N%i == 0:
            m = i 
       alpha = 0#alpha_estimator(m, (grads - mean_grad).view(-1, 1)).item()       

       del grads
       del mean_grad
    #N = M * P 

    #for i in range(1, 1 + int(math.sqrt(N))):
    #    if N%i == 0:
    #        m = i
    #alpha = alpha_estimator(m, (grads - mean_grad).view(-1, 1))
    
    #del grads
    #del mean_grad
    
    hist = [
        total_loss / total_size, 
        total_acc  / total_size,
        alpha
        ]

    #print(datatype, hist)
    return hist, outputs, noise_norm, M

def add_gaussian_noise(model, variance):
    n = tdist.Normal(torch.tensor(0.0), torch.tensor(variance))   
    for p in model.parameters():
        noise   = n.sample(p.grad.shape).cuda()  
        noise[torch.isnan(noise)] = 0   
        p.grad += noise 

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
   



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', default=100000, type=int)
    parser.add_argument('--batch_size_train', default=100, type=int)
    parser.add_argument('--batch_size_eval', default=100, type=int,
        help='must be equal to training batch size')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--mom', default=0, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--eval_freq', default=100, type=int)
    parser.add_argument('--dataset', default='mnist', type=str,
        help='mnist | cifar10 | cifar100')
    parser.add_argument('--path', default='./data', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', default='fc', type=str)
    parser.add_argument('--criterion', default='NLL', type=str,
        help='NLL | linear_hinge')
    parser.add_argument('--scale', default=64, type=int,
        help='scale of the number of convolutional filters')
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--width', default=100, type=int, 
        help='width of fully connected layers')
    parser.add_argument('--save_dir', default='results/', type=str)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--lr_schedule', action='store_true', default=False)
    parser.add_argument('--add_gaussian_noise', type=bool, default=False)
    parser.add_argument('--gaussian_variance', type=float, default=0.0) 
    parser.add_argument('--subset', type=int, default=-1)
    parser.add_argument('--load_dir', default='None', type=str) 
    parser.add_argument('--batch_norm',  type=bool, default=False)
    parser.add_argument('--noise_batch_size_train',  type=int, default=256)      
    parser.add_argument('--activation',  type=str, default='relu')
    parser.add_argument('--grad_sample_size', default=-1, type=int)

    #parameters for gaussian data
    parser.add_argument('--gauss_dimension',  type=int, default=100)
    parser.add_argument('--gauss_train_samples',  type=int, default=50000)
    parser.add_argument('--gauss_test_samples',  type=int, default=10000)
    parser.add_argument('--gauss_scale', default=10.0, type=float) 


    args = parser.parse_args()

    #print (args.add_gaussian_noise, args.gaussian_variance, args.subset)

    # initial setup
    if args.double:
        torch.set_default_tensor_type('torch.DoubleTensor')
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    torch.manual_seed(args.seed)
    
    print(args)

    # training setup
    train_loader, test_loader_eval, train_loader_eval, noise_loader_train, num_classes = get_data(args)


    if args.load_dir != 'None':
        # save the setup
        #torch.save(args, args.save_dir + '/args.info')
        # save the outputs
        te_oututs  = torch.load(args.load_dir + '/te_outputs.pyT')
        tr_outputs = torch.load(args.load_dir + '/tr_outputs.pyT')
        # save the model
        net        = torch.load(args.load_dir + '/net.pyT')
        # save the logs
        training_history = torch.load(args.load_dir + '/training_history.hist')
        weight_grad_history = torch.load(args.load_dir + '/weight_history.hist')
        evaluation_history_TEST = torch.load(args.load_dir + '/evaluation_history_TEST.hist')
        evaluation_history_TRAIN = torch.load(args.load_dir + '/evaluation_history_TRAIN.hist')
        noise_norm_history_TEST = torch.load(args.load_dir + '/noise_norm_history_TEST.hist')
        noise_norm_history_TRAIN = torch.load(args.load_dir + '/noise_norm_history_TRAIN.hist')

   
    else:
        if args.model == 'fc':
            if args.dataset == 'mnist' or args.dataset == 'gauss':
                if args.dataset == 'gauss':
                    net = fc(width=args.width, depth=args.depth, num_classes=num_classes, input_dim=args.gauss_dimension, activation=args.activation, use_batch_norm=args.batch_norm).to(args.device)
                else:
                    net = fc(width=args.width, depth=args.depth, num_classes=num_classes, activation=args.activation, use_batch_norm=args.batch_norm).to(args.device)
            elif args.dataset == 'cifar10':
                net = fc(width=args.width, depth=args.depth, num_classes=num_classes, input_dim=3*32*32, activation=args.activation, use_batch_norm=args.batch_norm).to(args.device)
        elif args.model == 'alexnet':
            net = alexnet(ch=args.scale, num_classes=num_classes, use_batch_norm=args.batch_norm).to(args.device)
        elif args.model == 'lenet':
            net = lenet().to(args.device)  
        elif 'VGG' in args.model:
            net = VGG(args.model, batch_norm=args.batch_norm).to(args.device)    
        elif 'Resnet' in args.model: 
            net = ResNet18(use_batch_norm = args.batch_norm).to(args.device) 
     

        training_history = []
        weight_grad_history = []

        # eval logs less frequently
        evaluation_history_TEST = []
        evaluation_history_TRAIN = []
        noise_norm_history_TEST = []
        noise_norm_history_TRAIN = []

        net = torch.nn.DataParallel(net)
        net.apply(weights_init) 
    print(net)
    
    opt = optim.SGD(
        net.parameters(), 
        lr=args.lr, 
        momentum=args.mom,
        weight_decay=args.wd
        )

    if args.lr_schedule:
        milestone = int(args.iterations / 3)
        scheduler = optim.lr_scheduler.MultiStepLR(opt, 
            milestones=[milestone, 2*milestone],
            gamma=0.1)
    
    if args.criterion == 'NLL':
        crit = nn.CrossEntropyLoss().to(args.device)
    elif args.criterion == 'linear_hinge':
        crit = linear_hinge_loss
    
    def cycle_loader(dataloader):
        while 1:
            for data in dataloader:
                yield data

    circ_train_loader = cycle_loader(train_loader)
 
    #circ_train_loader_eval = cycle_loader(train_loader_eval) 
    #circ_test_loader_eval  = cycle_loader(test_loader_eval)   
    # training logs per iteration
    #training_history = []
    #weight_grad_history = []

    # eval logs less frequently
    #evaluation_history_TEST = []
    #evaluation_history_TRAIN = []
    #noise_norm_history_TEST = []
    #noise_norm_history_TRAIN = []

    STOP = False

    #Compute #iterations in one epoch
    if args.subset == -1: 
        epoch_iter = 50000 // args.batch_size_train 
    else:
        epoch_iter = args.subset // args.batch_size_train 

    if args.load_dir != 'None':
        start = evaluation_history_TEST[-1][0]
    else:
        start = 0
         
    #args.gaussian_variance == -1: adaptive gradient noise addition
    if args.gaussian_variance == -1:
        adaptive_var = 0.0 
 
    for iter_num, (x, y) in enumerate(circ_train_loader):
        i = start + iter_num 
        if  i %  args.eval_freq == 0:
            # first record is at the initial point
            te_hist, te_outputs, te_noise_norm, _ = eval(test_loader_eval,  net, crit, opt, args, i % (1 * args.eval_freq))
            if i % (1 * args.eval_freq) == 0: 
               tr_hist, tr_outputs, tr_noise_norm, total_parameters = eval(noise_loader_train, net, crit, opt, args, i % (1 * args.eval_freq), test=False, grad_batch=args.grad_sample_size)
               print (np.amax(tr_noise_norm.numpy()))
            else:
               tr_hist, tr_outputs, tr_noise_norm, _ = eval(train_loader_eval, net, crit, opt, args, i % (1 * args.eval_freq), test=False, grad_batch=args.grad_sample_size)

            print ('Evaluation on test data at iteration ',  i, te_hist[0], te_hist[1])
            print ('Evaluation on train data at iteration ', i, tr_hist[0], tr_hist[1])
            evaluation_history_TEST.append([i, te_hist])
            evaluation_history_TRAIN.append([i, tr_hist])

            if i %  (1 * args.eval_freq) == 0: 
                noise_norm_history_TEST.append(te_noise_norm)
                noise_norm_history_TRAIN.append(tr_noise_norm)

                #update the apdaptive gaussian noise variance to be added
                if args.gaussian_variance == -1:
                    adaptive_var = 0.5 * np.amax(tr_noise_norm.numpy())/np.sqrt(total_parameters)
                    print ('Adaptive gaussian noise std dev. ', adaptive_var)
   
            if tr_hist[1] >= 101:
                print('yaaay all training data is correctly classified!!!')
                STOP = True

        net.train()

        #print (len(x.numpy()))        
        x, y = x.to(args.device), y.to(args.device)

        opt.zero_grad()
        out = net(x)
        loss = crit(out, y)

        # calculate the gradients
        loss.backward()
        if args.add_gaussian_noise:
            if args.gaussian_variance == -1:
               add_gaussian_noise(net, adaptive_var) 
            else:          
               add_gaussian_noise(net, args.gaussian_variance)  
   

        # record training history (starts at initial point)
        training_history.append([i, loss.item(), accuracy(out, y).item()])
        weight_grad_history.append([i, get_layerWise_norms(net)])

        # take the step
        opt.step()

        if i % args.print_freq == 0:
            print(training_history[-1])

        if args.lr_schedule:
            scheduler.step(i)

        if i > args.iterations:
            STOP = True

        if STOP:
            # final evaluation and saving results
            print('eval time {}'.format(i))
           
            te_hist, te_outputs, te_noise_norm, _ = eval(test_loader_eval, net, crit, opt, args,  (i + 1) % (1 * args.eval_freq))
            tr_hist, tr_outputs, tr_noise_norm, _ = eval(train_loader_eval, net, crit, opt, args,  (i + 1) % (1 * args.eval_freq), test=False, grad_batch=args.grad_sample_size)
            evaluation_history_TEST.append([i + 1, te_hist])
            evaluation_history_TRAIN.append([i + 1, tr_hist])
            noise_norm_history_TEST.append(te_noise_norm)
            noise_norm_history_TRAIN.append(tr_noise_norm)

            
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            else:
                print('Folder already exists, beware of overriding old data!')

            # save the setup
            torch.save(args, args.save_dir + '/args.info')
            # save the outputs
            torch.save(te_outputs, args.save_dir + '/te_outputs.pyT')
            torch.save(tr_outputs, args.save_dir + '/tr_outputs.pyT')
            # save the model
            torch.save(net, args.save_dir + '/net.pyT') 
            # save the logs
            torch.save(training_history, args.save_dir + '/training_history.hist')
            torch.save(weight_grad_history, args.save_dir + '/weight_history.hist')
            torch.save(evaluation_history_TEST, args.save_dir + '/evaluation_history_TEST.hist')
            torch.save(evaluation_history_TRAIN, args.save_dir + '/evaluation_history_TRAIN.hist')
            torch.save(noise_norm_history_TEST, args.save_dir + '/noise_norm_history_TEST.hist')
            torch.save(noise_norm_history_TRAIN, args.save_dir + '/noise_norm_history_TRAIN.hist')
            
            break

    
