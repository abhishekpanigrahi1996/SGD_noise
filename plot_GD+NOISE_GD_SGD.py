import torch
import numpy
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os


folder = sys.argv[1]
hist_train = torch.load(folder + '/evaluation_history_TRAIN.hist')
hist_test  = torch.load(folder + '/evaluation_history_TEST.hist')

#exit(0)

loss_train1  = [elem[1][0] for elem in hist_train]
loss_test1   = [elem[1][0] for elem in hist_test ]

acc_train1   = [elem[1][1] for elem in hist_train]
acc_test1    = [elem[1][1] for elem in hist_test ]



iterations = []
for i in range(12500):
#    if i < 1000:
#        iterations.append(i)
#    else:
        iterations.append(i *  10)


folder = sys.argv[2]
hist_train = torch.load(folder + '/evaluation_history_TRAIN.hist')
hist_test  = torch.load(folder + '/evaluation_history_TEST.hist')

#exit(0)

loss_train2  = [elem[1][0] for elem in hist_train]
loss_test2   = [elem[1][0] for elem in hist_test ]

acc_train2   = [elem[1][1] for elem in hist_train]
acc_test2    = [elem[1][1] for elem in hist_test ]




'''
loss_train3 = []
loss_test3  = []

acc_train3  = []
acc_test3   = []

#Read from file
f = open('Results_noise+GD_train')
for line in f:
   elems = line.strip().split(',')
   loss_train3.append(float(elems[2][:-1]))
   acc_train3.append( float(elems[3][:-1]))
f.close()


f = open('Results_noise+GD_test')
for line in f:
   elems = line.strip().split(',')
   loss_test3.append(float(elems[2][:-1]))
   acc_test3.append( float(elems[3][:-1]))
f.close()
'''

if sys.argv[3] != 'None':
    folder = sys.argv[3]
    hist_train = torch.load(folder + '/evaluation_history_TRAIN.hist')
    hist_test  = torch.load(folder + '/evaluation_history_TEST.hist')
    loss_train3  = [elem[1][0] for elem in hist_train]
    loss_test3   = [elem[1][0] for elem in hist_test ]

    acc_train3   = [elem[1][1] for elem in hist_train]
    acc_test3    = [elem[1][1] for elem in hist_test ]


if sys.argv[4] != 'None':
    folder = sys.argv[4]
    hist_train = torch.load(folder + '/evaluation_history_TRAIN.hist')
    hist_test  = torch.load(folder + '/evaluation_history_TEST.hist')
    loss_train4  = [elem[1][0] for elem in hist_train]
    loss_test4   = [elem[1][0] for elem in hist_test ]

    acc_train4   = [elem[1][1] for elem in hist_train]
    acc_test4    = [elem[1][1] for elem in hist_test ]


Destination_folder = sys.argv[5]
if not os.path.isdir(Destination_folder):
   os.mkdir(Destination_folder) 


plt.plot(loss_train1, loss_test1, 'rx', label='SGD_bt256')
plt.plot(loss_train2, loss_test2, 'bx', label='SGD_bt4096')

if sys.argv[3] != 'None':
    plt.plot(loss_train3, loss_test3, 'gx', label='SGD_bt4096+1e-1VarNoise')
if sys.argv[4] != 'None':
    plt.plot(loss_train4, loss_test4, 'mx', label='SGD_bt4096+1e-2VarNoise')

plt.xlabel('Train loss')
plt.ylabel('Test loss')
plt.title('Train Test losses relation for different settings')
plt.legend(loc='upper right')
plt.savefig(Destination_folder + '/Loss_tradeoff.png')
plt.gcf().clear()



plt.plot(acc_train1, acc_test1, 'rx', label='SGD_bt256')
plt.plot(acc_train2, acc_test2, 'bx', label='SGD_bt4096')

if sys.argv[3] != 'None':
    plt.plot(acc_train3, acc_test3, 'gx', label='SGD_bt4096+1e-1VarNoise')
if sys.argv[4] != 'None':
    plt.plot(acc_train4, acc_test4, 'mx', label='SGD_bt4096+1e-2VarNoise')

plt.xlabel('Train accuracy')
plt.ylabel('Test accuracy')
plt.title('Train Test accuracies relation for different settings')
plt.legend(loc='lower right')
plt.savefig(Destination_folder + '/ACC_tradeoff.png')




