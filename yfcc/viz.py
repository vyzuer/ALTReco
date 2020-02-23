import keras
import numpy as np
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Conv2D, Flatten
from keras.optimizers import SGD,Adam, RMSprop
import cv2
import h5py
from keras.utils.io_utils import HDF5Matrix
from keras.models import *
from keras import losses
import tensorflow as tf
from keras import metrics
from keras.applications.resnet50 import ResNet50
from keras.utils import Sequence
import os 
from loss import CrimLoss
#os.environ["CUDA_VISIBLE_DEVICES"]="2"

X = "/home/c2-1/yogesh/erik/correct/test/hdf5_yfcc_imgs_test.h5"
Y = "/home/c2-1/yogesh/erik/correct/test/hdf5_yfcc_pvects_test.h5"
Z = "/home/c2-1/yogesh/erik/correct/test/hdf5_yfcc_uvects_test_norm.h5"
#classes = np.loadtxt('/home/vyzuer/work/data/DataSets/ftags/tags_1540_list.txt', dtype=str)
classes = np.loadtxt("tags_1540_list.txt", dtype=str)

def huber_loss(y, ypred):
        return tf.losses.huber_loss(y, ypred)

keras.losses.huber_loss = huber_loss
keras.losses.CrimLoss = CrimLoss

imgdump = "vizdump/imgs/"
txtdump = open("vizdump/testtags.txt",'w')
imgs = []
uvs = []
gt = []
class DataGen(Sequence):
        def __init__(self, tot_ex, batch_size = 32):
                self.batch_size =batch_size
		self.gts = []
                self.tot_ex = tot_ex
                self.on_epoch_end()
                self.lwr = 0
                self.upr = batch_size
		list = range(0,60000) 
		np.random.shuffle(list) 
		self.master = list
        def __getitem__(self, index):
                indices = self.master[self.lwr:self.upr]
                (X) =  self.__data_generation(indices = indices )
                self.lwr += self.batch_size
                self.upr +=self.batch_size
                return (X)

        def __len__(self):
                return int(np.floor((self.tot_ex) / self.batch_size))
        def on_epoch_end(self):
                #upr += self.batch_size
                self.lwr = 0
                self.upr = self.batch_size
                return

        def __data_generation(self, indices):
                (X,y) = generator_hdf5(indices = indices, batchSize = self.batch_size)
		global gt 
		gt.append(y)
		global uvs 
		uvs.append(X["input_1"])
                return (X)




ctr =0 
def evaluate_k(gt, est, k=10):
    """
        the est are the estimated labels
        """
    acc = 0.0
    prec = 0.0
    rec = 0.0
    tp = 0.0

    tag_ids = est.argsort()[::-1] #.argsort()

    for i in range(k):
        _id = tag_ids[i]
        if gt[_id] == 1:
            acc = 1.0
            tp += 1.0

    prec = tp/k
    rec = tp/np.sum(gt)
    viz = True
    if viz:
	global ctr
	e = est.argsort()[::-1][:5]
        estV = [classes[e]]
	gtV = classes[np.where(gt)]
	txtdump.write(str(ctr) + " " +  str(acc) + " " + str(prec) + " " + str(rec) + "\nest: " + str(estV) + "  \ngt: " + str(gtV) + "\n")
	ctr +=1
    print acc, prec, rec#, estV, gtV

    return acc, prec, rec


def homebrew(y,ypred):
            num_batches = 1
	    batch_size = 500
            acc = 0.0
            prec = 0.0
            rec = 0.0
            acc1 = 0.0
            #gts = np.asarray(y)
	    #print gts.shape
	    for x in range(num_batches):	
		    gts = y[x]
		    gts = gts["Personal"]
		    ests = np.asarray(ypred)
                    tmp = x*batch_size
                    ests = ests[tmp:tmp+batch_size]

		    #print ests.shape
		    #ests = ests[0]
		    #for k in gts:
			#print k 
		    print gts.shape, ests.shape
		    #exit()
		    #gts = gts[0]
		    for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
		#               acc += hamming_distance(gt, est > 0)
				a, p, r = evaluate_k(gt, est, 5)
				acc1 += a
				prec += p
				rec += r

            acc1 /= (1. * num_batches * batch_size)
            prec /= (1. * num_batches * batch_size)
            rec  /= (1. * num_batches * batch_size)
            print 'acc: %.6f, prec: %.6f, rec: %.6f' %(acc1, prec, rec)

            return #acc / (1. * num_batches * batch_size)

def generator_hdf5(indices, batchSize=100,is_train=True, aug=None):

    db = h5py.File(X, "r")
    label = h5py.File(Y,'r')
    uvec = h5py.File(Z, 'r') 
    
    while True:
                #np.random.shuffle(indices)
        for i in range(0,  1):
            #t0 = time()
            batch_indices = indices[i:i+batchSize]
            batch_indices.sort()

            img = db["data"][batch_indices,:]
	    global imgs 
	    imgs.append(img) #turn me on to draw images 
	    #global ctr1
	    #cv2.imwrite(imgdump +str(ctr1) + ".jpg" , img) 
	    #global ctr1 
	    #ctr1+=1
            #img[:,:,0]-= 109
            #img[:,:,1]-= 104
            #img[:,:,2]-= 97
            pv = label['label'][batch_indices, : ]
	    #   return (by,bx)
	    uv = uvec['uservec'][batch_indices, :]
            #del uvec

            return  {'input_1': uv, 'input_2': img}, {'Personal': pv}


model = keras.models.load_model("./gan_yfcc_15.h5") #('./gan_nus.h5')
print "fetching examples..."
#testlen is 67914
generator = DataGen(tot_ex = 67914, batch_size = 500)
ests = model.predict_generator(generator = generator, steps = 1)
gts = np.asarray(gt)
ests = np.asarray(ests) 
print ests.shape
#this will print the estimated tags 
ests = ests[1]
viz = False
if viz:	
	for x in ests:
		#print x 
		e = x.argsort()[::-1][:10]
		e = classes[e]
		#e = [classes[x]]
		#vizdump.write(e)
homebrew(gts,ests)
#turn me on to write images 
'''
uvs = np.asarray(uvs) 
print uvs.shape
uvs = uvs[0]
uvfile = open("uservects.txt","w") 
for x in uvs:
	uv = classes[np.where(x)]
	uvfile.write(uv)
	uvfile.write("\n")
'''	


imgs = np.asarray(imgs) 
for idx,x in enumerate(imgs[0]):
	#x[:,:,0] += 109
        #x[:,:,1] += 104
        #x[:,:,2] += 97
	cv2.imwrite(imgdump + str(idx)+".jpg", x)


