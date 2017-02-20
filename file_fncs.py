"""
FUNCTIONS DOING SOMETHING USEFUL AND RELATER TO READING OR WRITING TO AND FROM FILES
"""

import os, shutil

def file2list(filename):

	"""
	read lines from a text file "filename" and put them into a list 
	"""
	with open(filename, "r") as f:
		lst = [line.strip() for line in f if line.strip()]
	
	return lst

def list2file(lst, filename):

	"""
	saves a list into a file 
	"""

	with open(filename, "w") as f:
		for w in lst:
			f.write("{}\n".format(w))

def setup_dirs(dnames, todo="check"):

	"""
	if any directory from the directory list exists, delete it and create anew;
	alternatively, check if a directory exists and if it doesn't, create it
	"""

	for dname in dnames:

		if os.path.isdir(dname):  # if dir exists
			if todo == "reset":  #  we need to delete it and create anew 
				print("directory {} already exists. resetting...".format(dname), end="")
				shutil.rmtree(dname)
				os.mkdir(dname)
				print("ok")			
			elif todo == "check":  # only had to check if exists, do nothing
				pass
		else:  # if directory doesn't exist
			if todo == "check":
				sys.exit(">> error >> directory {} is MISSING!".format(dname))
			elif todo == "reset":
				print("creating new directory {}...".format(dname), end="")
				os.mkdir(dname)
				print("ok")

def enc_lst(lst):
	"""
	read a list from file and then encode the items in order of appearance (top to bottom)
	"""
	return({itm:indx for indx, itm in enumerate(lst)})

def dec_dict(dk):
	"""
	reverse a dictionary
	"""
	return({v: k for k, v in dk.items()})
