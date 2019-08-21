import torch
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
from PIL import Image
from images2gif import writeGif
import powerlaw



folder = sys.argv[1]
Destination_folder  = sys.argv[2]

if not os.path.isdir(Destination_folder):
    os.mkdir(Destination_folder)

hist_train  = torch.load(folder + '/noise_norm_history_TRAIN.hist')
#print (hist_train)
#hist_test   = torch.load(folder + '/noise_norm_history_TEST.hist')
result_arr_R = []
result_arr_p = []
result_arr_alpha = []
result_arr_diff  = []
result_arr_sigma = []


index      = []
index_counter = 0 
#min_num = 1000.0
#max_num = 0.0

for elems in hist_train:
    if len(elems) > 0 and index_counter % 1 == 0:
        #result_arr.append(elems.numpy())
        fit =  powerlaw.Fit(elems.numpy())        
        R, p =  fit.distribution_compare('power_law', 'exponential')
        #print (len(elems.numpy()), fit.alpha, fit.x_min, fit.sigma, R, p)
        #if np.amin(elems.numpy()) < min_num:
        #    min_num = np.amin(elems.numpy())
        #if np.amax(elems.numpy()) > max_num:
        #    max_num = np.amax(elems.numpy())  
        result_arr_R.append(R)
        result_arr_p.append(p)
        result_arr_diff.append(np.absolute(fit.xmin - np.mean(elems.numpy())))
        result_arr_alpha.append(fit.alpha)
        result_arr_sigma.append(fit.sigma)  


        index.append(1000 * index_counter)
    index_counter += 1  
#    if index_counter > 50000:
#       break  

plt.figure()

#import pickle
#y1, y2, y3, y4, result_arr_sigma, index = pickle.load(open(Destination_folder + '/Comparisontest.pkl', 'rb'))
#y1, y2, y3, y4, result_arr_sigma, index = y1[4:], y2[4:], y3[4:], y4[4:], result_arr_sigma[4:], index[4:]

#print (index)

x1 = index#np.linspace(0.0, 5.0)
x2 = index#np.linspace(0.0, 2.0)
x3 = index
x4 = index


y1 = result_arr_R#np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = result_arr_p#np.cos(2 * np.pi * x2)
y3 = result_arr_diff
y4 = result_arr_alpha

plt.subplot(4, 1, 1)
plt.plot(x1, y1, 'ko-')
#plt.title('Comparison between Power law v/s exponential law fit to SGD noise norm')
plt.title('Log likelihood ration between power law and exponential distribution')


plt.subplot(4, 1, 2)
plt.plot(x2, y2, 'r.-')
plt.plot(x2, np.ones(len(y2)) * 0.1, 'g-')
#plt.xlabel('Iterations')
plt.ylabel('p value of the test')
#plt.savefig(Destination_folder + '/Comparisontest.png')

plt.subplot(4, 1, 3)
plt.plot(x3, y3, 'ko-', label='Absolute Difference in xmin and mean')
#plt.title('Comparison of test data')
#plt.title('Absolute Difference in xmin and mean')
plt.legend(loc='upper left')



plt.subplot(4, 1, 4)
plt.plot(x4, y4, 'r.-')
plt.xlabel('Iterations')
plt.ylabel('Fitted alpha')
plt.savefig(Destination_folder + '/Comparisontest.png')
#import pickle
#pickle.dump([y1, y2, y3, y4, result_arr_sigma, index], open(Destination_folder + '/Comparisontest.pkl', 'wb'))


plt.gcf().clear()


plt.subplot(2, 1, 1)
plt.plot(index, result_arr_sigma, 'ko-')
plt.ylabel('Stddev in alpha computation')
plt.title('Power law fit statstics')

plt.subplot(2, 1, 2)
plt.plot(index, y4, 'ko-')
plt.xlabel('Iterations')
plt.ylabel('Fitted alpha')

plt.savefig(Destination_folder + '/powerlawfit.png')

#print (min_num, max_num)

#result_arr = np.asarray(result_arr)
#print (result_arr)
#min_num = np.amin(result_arr.all())
#max_num = np.amax(result_arr.all())
#num_bins = 1000
#bin_length = (max_num - min_num) / num_bins

#bins = np.linspace(min_num - bin_length, max_num + bin_length, num=num_bins)

#i=0

#fileList = []
#for elems in result_arr:
#        arr = elems
#        plt.hist(arr, bins=1000)                   
#        plt.xlabel('Noise norm Bins')
#        plt.ylabel('Number of batches in the bin')
#        plt.legend(loc='upper right')
#        plt.title(sys.argv[3] + '_Iter_' + str(index[i]))
#        plt.savefig(Destination_folder + '/Img_Iter_' + str(index[i]) + '.png')
#        plt.gcf().clear()
#        fileList.append(Destination_folder + '/Img_Iter_' + str(index[i]) + '.png') 
#        i += 1
 






#fileList = []
#i=0
#for elems in result_arr:
#    fileList.append(Destination_folder + '/Img_Iter_' + str(index[i]) + '.png')
#    i += 1

#print (fileList)
#images = [Image.open(fn) for fn in fileList]
#print (images)
#filename = Destination_folder + '/hist_images.gif'
#writeGif(filename, images, duration=1.0)
#os.chdir(Destination_folder)
#file = open('blob_fileList.txt', 'w')
#for item in fileList:
#    file.write("%s\n" % item)

#file.close()


#os.system('convert -delay 50 @blob_fileList.txt animated.gif')



#frames = []
#for i in fileList:
#    new_frame = Image.open(i)
#    frames.append(new_frame)

#print(frames[1:])
 
# Save into a GIF file that loops forever
#frames[1].save(Destination_folder + '/hist_images.gif', format='GIF',
#               append_images=frames[2:],
#               save_all=True,
#               duration=10, loop=1)


#hist_train = torch.load(folder + '/evaluation_history_TRAIN.hist')
#hist_test  = torch.load(folder + '/evaluation_history_TEST.hist') 

#loss_train  = [elem[1][0] for elem in hist_train]
#loss_test   = [elem[1][0] for elem in hist_test ]

#acc_train   = [elem[1][1] for elem in hist_train] 
#acc_test    = [elem[1][1] for elem in hist_test ]

#iterations = []
#for i in range(len(loss_train)):
#    iterations.append(i * 10)




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
'''

