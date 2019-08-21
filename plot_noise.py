import torch
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
from PIL import Image
from images2gif import writeGif

folder = sys.argv[1]
Destination_folder  = sys.argv[2]

if not os.path.isdir(Destination_folder):
    os.mkdir(Destination_folder)

hist_train  = torch.load(folder + '/noise_norm_history_TRAIN.hist')
#print (hist_train)
#hist_test   = torch.load(folder + '/noise_norm_history_TEST.hist')
result_arr = []
index      = []
index_counter = 0 
min_num = 1000.0
max_num = 0.0
for elems in hist_train:
    if len(elems) > 0:
        result_arr.append(elems.numpy())
        if np.amin(elems.numpy()) < min_num:
            min_num = np.amin(elems.numpy())
        if np.amax(elems.numpy()) > max_num:
            max_num = np.amax(elems.numpy())  

        index.append(index_counter)
    index_counter += 1000   

#print (min_num, max_num)

result_arr = np.asarray(result_arr)
#print (result_arr)
#min_num = np.amin(result_arr.all())
#max_num = np.amax(result_arr.all())
num_bins = 1000
bin_length = (max_num - min_num) / num_bins

bins = np.linspace(min_num - bin_length, max_num + bin_length, num=num_bins)

i=0

fileList = []
for elems in result_arr:
        arr = elems
        plt.hist(arr, bins=bins)                   
        plt.xlabel('Noise norm Bins')
        plt.ylabel('Number of batches in the bin')
        plt.legend(loc='upper right')
        plt.title(sys.argv[3] + '_Iter_' + str(index[i]))
        plt.savefig(Destination_folder + '/Img_Iter_' + str(index[i]) + '.png')
        plt.gcf().clear()
        fileList.append(Destination_folder + '/Img_Iter_' + str(index[i]) + '.png') 
        i += 1
 
#exit(0)

fileList = []
i=0
for elems in result_arr:
    fileList.append(Destination_folder + '/Img_Iter_' + str(index[i]) + '.png')
    i += 1

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



frames = []
for i in fileList:
    new_frame = Image.open(i)
    frames.append(new_frame)

#print(frames[1:])
 
# Save into a GIF file that loops forever
frames[1].save(Destination_folder + '/hist_images.gif', format='GIF',
               append_images=frames[2:],
               save_all=True,
               duration=10, loop=1)


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

