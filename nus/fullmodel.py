import keras 
import numpy as np  
from loss import CrimLoss
from keras.layers import Dense, Dropout, LeakyReLU, Conv2D, Flatten, concatenate, RepeatVector, Reshape
from keras import layers
from keras.optimizers import SGD,Adam, RMSprop
import cv2
import h5py 
from keras.utils.io_utils import HDF5Matrix
from keras.models import *
from keras import losses
import tensorflow as tf 
from keras.layers import LeakyReLU
from keras.applications.resnet50 import ResNet50
from keras.engine.topology import Layer
from keras import backend as K
from keras.utils import Sequence
import random 
from numpy.random import RandomState
import sys
#from loss import CrimLoss, OBCE 
np.set_printoptions(threshold=np.nan)
#import os 
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
NUM_CLASSES = 1000
ones = np.ones((16,))
zeros = np.zeros((16,))

def huber_loss(y, ypred):
        return tf.losses.huber_loss(y, ypred)


keras.losses.huber_loss = huber_loss
keras.losses.CrimLoss = CrimLoss
#keras.losses.OBCE = OBCE


img = "/home/erikqu/NUS-WIDE-downloader/fullNUS/hdf5gen/nus1k/hdf5_imgs_224.h5"
photovec ="/home/erikqu/NUS-WIDE-downloader/fullNUS/hdf5gen/nus1k/pvects.h5"
uservectors = "/home/erikqu/NUS-WIDE-downloader/fullNUS/hdf5gen/nus1k/hdf5_uvects_norm.h5" 
	
'''
X = "/home/erikqu/lmdb2h5/backup/hdf5_yfcc_imgs.h5"
Y = "/home/erikqu/lmdb2h5/backup/hdf5_yfcc_pvects.h5"
Z = "/home/erikqu/lmdb2h5/hdf5_yfcc_uvects_norm.h5"
'''

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val



class DataGen(Sequence):
        def __init__(self, tot_ex, batch_size = 32):
                self.batch_size = batch_size
                self.tot_ex = tot_ex
                self.on_epoch_end()
                self.lwr = 0
                self.upr = batch_size//2
                self.half = batch_size//2
                list = range(0,self.tot_ex)
                np.random.shuffle(list)
                self.master = list
        def __getitem__(self, index):
                indices = self.master[self.lwr:self.upr]
                #half indices will be actual, other half will be fake
                (X,y) =  self.__data_generation(indices = indices )
                self.lwr += self.half
                self.upr +=self.half
                return (X,y)

        def __len__(self):
                return int(np.floor((self.tot_ex) / self.batch_size))
        def on_epoch_end(self):
                #global lwr, upr, batchSize
                #lwr += self.batch_size
                #print lwr
                #upr += self.batch_size
                self.lwr = 0
                self.upr = self.batch_size//2
                return

        def __data_generation(self, indices):
                #if we roll a 1 then we take a lefitate vector, otherwise we take a fake one.
                (X,y)  = generator_hdf5(indices = indices, batchSize = self.batch_size)
                #print X.shape
		#print self.lwr
                #y = np.append(ones, axis =0) #labels for data
                #now we must shuffle and not lose the places
                #we will get a seed, and reshuffle when resetting randgenerator
                return (X,y)



def shuffle_reset(X,y):
	seed = random.randint(0,10000)
        p = RandomState(seed)
        p.shuffle(X)
        p = RandomState(seed)
        p.shuffle(y)
	return (X,y)

def generator_hdf5(indices, batchSize=32,is_train=True, aug=None):
	#this is where our 'real' example will come rom 
    global photovec, NUM_CLASSES
    pvec = h5py.File(photovec, 'r')
    imgs = h5py.File(img, 'r')
    uvec = h5py.File(uservectors, 'r') 
    while True:
        for i in range(0,  1):
            batch_indices = indices[i:i+batchSize]
            batch_indices.sort()

            pv = pvec['label'][batch_indices, :]
	    im = imgs['data'][batch_indices, :]
	    uv = uvec['uservec'][batch_indices, :]


	    im[:,:,0] -= 109
            im[:,:,1] -= 104
            im[:,:,2] -= 97
            return  {'input_1': uv, 'input_2': im}, {'Reconstruction': uv, 'Personal': pv, 'General': pv}

def huber_loss(y, ypred):
	return tf.losses.huber_loss(y, ypred)



generator = DataGen(batch_size = 32, tot_ex = 205000)
class GAN(object):
	def __init__(self, NUM_CLASSES = NUM_CLASSES):
		self.NUM_CLASSES = NUM_CLASSES 
		self.G = self.__generator() 
		self.D = self.__discriminator()  
		self.optimizer = 'adadelta'
		self.G.compile(loss = [ huber_loss, 'binary_crossentropy','binary_crossentropy'], optimizer = self.optimizer) 
		self.D.compile(loss = 'binary_crossentropy',optimizer = self.optimizer) 
	        self.stacked_G_D= self.__stacked_generator_discriminator()
	        self.stacked_G_D.compile(loss=[huber_loss, 'binary_crossentropy', 'binary_crossentropy' ,'binary_crossentropy' ], optimizer=self.optimizer)
	def __generator(self):
		
		AE_input = Input(shape = (NUM_CLASSES,))
		x = Dense(NUM_CLASSES, activation = 'relu')(AE_input)
		x = Dense(1024, activation = 'relu')(x)
		x_512 = Dense(512, activation = 'relu')(x)
		x_256 = Dense(256, activation = 'relu')(x_512)
		encoded = Dense(128, activation = 'tanh')(x_256)
		y_256 = Dense(256, activation = 'relu')(encoded)
		con_256 = concatenate([x_256,y_256])
		y_512 = Dense(512, activation = 'relu')(con_256)
		con_512 = concatenate([x_512,y_512])
		x = Dense(1024, activation = 'relu')(con_512)
		AE_OUT = Dense(NUM_CLASSES, activation = 'sigmoid', name = "Reconstruction")(x)
		

		resnet = ResNet50(include_top = True, weights = 'imagenet')

		#General Tags
		
		x = Dense(2048, activation = 'relu')(resnet.output)
		gen_tags = Dense(NUM_CLASSES, activation ="sigmoid", name = "General")(x)
		
		
		#Personalized Tags
		 
		resnet_portion = resnet.get_layer("activation_48").output
		encoded = RepeatVector(49)(encoded) 
		encoded = Reshape((7,7,128))(encoded) 
		encoded = concatenate([encoded, resnet_portion])
		x = Conv2D(filters = 64, kernel_size = (1,1))(encoded)
		x = Conv2D(filters=32, kernel_size = (1,1))(x)
		x = Flatten()(x)
		x = Dense(2048, activation = 'relu')(x)
		pers_tags = Dense(NUM_CLASSES, activation = "sigmoid", name = "Personal")(x)
		
		model = Model(input = [AE_input, resnet.input], output = [AE_OUT, pers_tags, gen_tags])
		inputs = [AE_input, resnet.input]
		return model

	def __discriminator(self): 
		#disc = Model(input = gen_tags, output = disc.output) 
		disc_in = Input(shape = (NUM_CLASSES,))
		x = Dense(NUM_CLASSES, activation = LeakyReLU(alpha=.2))(disc_in)
		#x = Dense(1024, activation = LeakyReLU(alpha=.2))(x)
		x = Dense(512, activation = LeakyReLU(alpha=.2))(x)
		x = Dense(256, activation = LeakyReLU(alpha=.2))(x)
		#x = Dense(128, activation = LeakyReLU(alpha=.2))(x)
		x = Dense(64, activation = LeakyReLU(alpha=.2))(x)
		#x = Dense(16, activation = LeakyReLU(alpha=.2))(x)
		x = Dense(4, activation = LeakyReLU(alpha=.2))(x)
		disc_out = Dense(1, activation = 'sigmoid', name = "Discriminator")(x)
		model = Model(input = disc_in, output = disc_out) 
		return model
		
	def __stacked_generator_discriminator(self):
		make_trainable(self.D, False)
		gan_input = self.G.input
		gen_output = self.G(gan_input) 
		x = self.D(gen_output[2])
		'''
		a= self.G.get_input_at(0)
		a = np.asarray(a)
		print a[1].shape
		b = self.G.get_input_at(1) 
		b = np.asarray(b)
		print b.shape 
		'''
		output = [self.G.get_output_at(-1)[0],self.G.get_output_at(-1)[1], self.G.get_output_at(-1)[2], x]
 
	    	model = Model(inputs = self.G.get_input_at(0) , outputs = output) 
		model.load_weights("./ResNet50_ae_nus.h5", by_name = True )		
		return model 

	def train(self, epochs, batch_size = 32):
		global generator
		num_sample = generator.__len__()
		for x in range(epochs):
			print "Epoch " + str(x+1) + "/" + str(epochs)
	
			for cnt in range(num_sample):
				#print str(cnt) + "/" +  str(num_sample)
				#train discriminator  
				#fetch a batch of data to train on 
				(actual_img,actual_labels) = generator.__getitem__(index =0)
				#now that we have our real example we must use self.G.predict to get our 'fake' examples
				
				#print actual_img['input_2']
				fake_labels = self.G.predict([actual_img['input_1'], actual_img['input_2']]) #use images


				#now that we have our real and fake labels we can train discriminator 
				#print actual_labels['General'].shape, fake_labels[2].shape
				x_train = np.append(actual_labels['General'],fake_labels[2], axis=0)
				global ones, zeros 
				y_train = np.append(ones,zeros, axis =0)
				
				seed = random.randint(0,10000)
				p = RandomState(seed)
				p.shuffle(x_train)
				p = RandomState(seed)
				p.shuffle(y_train)
				#print x_train.shape, y_train.shape	
				make_trainable(self.D, True)
				d_loss = self.D.train_on_batch(x_train,y_train)

				#train generator
				#(actual_img, actual_labels) = generator.__getitem__(index=0) 
				mislabels = ones
				make_trainable(self.D,False)
				g_loss = self.stacked_G_D.train_on_batch([actual_img['input_1'],actual_img['input_2']], [actual_labels['Reconstruction'], actual_labels['Personal'], actual_labels['General'], ones])
				aggregate_loss = g_loss[0]
				ae_loss = g_loss[1] 
				personal_loss = g_loss[2] 
				general_loss = g_loss[3] 
				disc_loss = g_loss[4] + d_loss
				#print "Discriminator Loss:" + str(d_loss) + "\tGenerator Loss: " + str(aggregate_loss)+   "\t\t" + str(cnt) + "/" +  str(num_sample)
				print "Disc_Loss: %.5f \t Gen: Tot_loss: %.5f  AE Loss %.5f  Pers_Loss:  %.5f  Gen_Loss: %.5f \t %d/%d" % (disc_loss, aggregate_loss, ae_loss, personal_loss, general_loss, cnt,num_sample )  
		
			#print ('Epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (x+1, d_loss, g_loss))
			generator.on_epoch_end()
			gan.G.save("gan" + "_" + str(x+1) + ".h5")
			

gan = GAN() 
gan.train(epochs =100)
gan.G.save("gen_nus_final.h5")




'''
sgd = SGD(lr = .001, decay = 1e-6, momentum= .9, nesterov=True)
model.compile( loss=[huber_loss, losses.binary_crossentropy, CrimLoss, losses.binary_crossentropy], optimizer = 'adadelta', loss_weights = {"Reconstruction": 1000000, "Personal":1000000, "General": 1000000})

batchSize = 32
filepath="yfcc-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, verbose=1, period = 1)
callbacks_list = [checkpoint]
generator = DataGen(batch_size = 32, tot_ex = 205000)
model.fit_generator(generator = generator, epochs = 42, workers = 12, use_multiprocessing = True, callbacks = callbacks_list)
'''
