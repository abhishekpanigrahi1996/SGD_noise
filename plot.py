import torch
import numpy
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



folder = sys.argv[1]

hist_train = torch.load(folder + '/evaluation_history_TRAIN.hist')
hist_test  = torch.load(folder + '/evaluation_history_TEST.hist') 

loss_train  = [elem[1][0] for elem in hist_train]
loss_test   = [elem[1][0] for elem in hist_test ]

acc_train   = [elem[1][1] for elem in hist_train] 
acc_test    = [elem[1][1] for elem in hist_test ]

'''

loss_train = []
loss_test  = []

acc_train  = []
acc_test   = []

iterations = []


#Read from file
f = open(folder+'_train')
for line in f:
   elems = line.strip().split(',') 
   if int(elems[1]) not in iterations:
       iterations.append(int(elems[1]))
       loss_train.append(float(elems[2]))
       acc_train.append( float(elems[3][:-1]))      
f.close()


tmp_iterations = []
f = open(folder + '_test')
for line in f:
   elems = line.strip().split(',')
   if int(elems[1]) not in tmp_iterations:
       tmp_iterations.append(int(elems[1])) 
       loss_test.append(float(elems[2]))
       acc_test.append(float(elems[3][:-1]))
f.close()
'''

#iteratons   = [i for i in range(2100)]
iterations = []
for i in range(len(loss_train)):
    if i < 8000:
        iterations.append(i * 100)
    #else:
    #    iterations.append(50000 + (i - 5000) * 100)

#Selected Loss and accuracy trend.
#loss_train = [loss_train[i] for i in range(len(loss_train)) if i%5==0]#np.asarray(loss_train) 
#loss_test  = [loss_test[i]  for i in range(len(loss_test))  if i%5==0]#np.asarray(loss_test)

#acc_train = [acc_train[i] for i in range(len(acc_train)) if i%5==0]#np.asarray(acc_train)
#acc_test  = [acc_test[i]  for i in range(len(acc_test))  if i%5==0]#np.asarray(acc_test)

'''
iterations = []
for i in range(len(loss_train)):
    if i < 0:
        iterations.append(i)
    else:
        iterations.append(0 + (i - 0) *  10)
'''

plt.plot(iterations, loss_train, 'r', label='Train Loss')
plt.plot(iterations, loss_test,  'b', label='Test Loss' )
plt.xlabel('Iterations')
plt.ylabel('Loss Value')
plt.legend(loc='upper right')
plt.title(sys.argv[4])
#axes.set_xlim([xmin,xmax])
#plt.ylim([0, 4])
plt.savefig(sys.argv[2])
plt.gcf().clear()
#exit(0)

plt.plot(iterations, acc_train, 'r', label='Train Accuracy')
plt.plot(iterations, acc_test,  'b', label='Test Accuracy' )
plt.xlabel('Iterations')
plt.ylabel('Accuracy Value')
plt.title(sys.argv[5])
plt.legend(loc='upper right')
plt.savefig(sys.argv[3])


