



def create_dict_fromuser():
	tort = "train"
        photos = open("./../../" + tort + "imgs.txt","r")
        tags = open("./../../" + tort + "tags.txt","r")
        users = open("./../../" + tort + "usrs.txt","r")
        dict = {}
	for user in users:
		#print user
		user = (user.split("\n"))[0]
		curtags = tags.readline()
		line = photos.readline()
                line = line.split("/")[-1]
                line = (line.split("_"))[-1]
                photo = (line.split("."))[0]
		try:
			dict[user]
			dict[user][0].append(photo)
			dict[user][1].append(curtags)
		except KeyError:
			dict[user] = [[photo],[curtags]]
				
		#dict[user].append(photo)
	
	return dict




def create_dict_fromphoto():
	photos = open("./../imageDump/train_imgs.txt","r")
	tags = open("./../imageDump/train_tags.txt","r")
	users = open("./../imageDump/train_users.txt","r")
	dict = {}
	for line in photos:
		line = line.split("/")[-1]
		line = (line.split("_"))[-1]
		photo_id = (line.split("."))[0]
		curtags = tags.readline()
		curuser = users.readline()
                dict[photo_id] = [curuser,curtags]
	return dict 




def main():
	#create_dict()
	create_dict_fromuser()


main()
