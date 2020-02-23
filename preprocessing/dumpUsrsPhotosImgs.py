#import numpy as np 
from random import sample 



file = open("allInfo.txt",'r') 

train = sample(range(0,215774), 212001)

imgfile = open("trainimgs.txt",'w') 
usrfile = open("trainusrs.txt",'w')
tagfile = open("traintags.txt",'w')
	
imgfile_test = open("testimgs.txt",'w')
usrfile_test = open("testusrs.txt",'w')
tagfile_test = open("testtags.txt",'w')


for idx,line in enumerate(file):
	print idx 
	line = line.split("\n")[0] 
	line = line.split(" ") 
	dir = line[0] 
	usr = line[3]
	tags = (line[6:])[:-3]
	tags = " ".join(tags)
	#print tags 
	if idx in train:
		imgfile.write(dir + "\n")
		usrfile.write(usr+ "\n") 
		tagfile.write(tags + "\n") 
		continue	
	
        imgfile_test.write(dir + "\n")
        usrfile_test.write(usr+ "\n")
        tagfile_test.write(tags + "\n")

	
