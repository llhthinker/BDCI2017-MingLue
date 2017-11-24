#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys, getopt
import random

"""
将原始文本打乱
"""

def main(argv):
	inputfile = ''
	outputfile = ''
	if len(argv) < 2:
		print(argv[0]+" -i <inputfile> -o <outputfile>")

	try:
		opts, args = getopt.getopt(argv[1:], "hi:o:",
				 ["ifile=", "ofile"])
	except getopt.GetoptError:
		print(argv[0]+" -i <inputfile> -o <outputfile>")
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':	
			print(argv[0]+" -i <inputfile> -o <outputfile>")
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile" ):
			outputfile = arg
	shuffle_data(inputfile, outputfile)

	
def shuffle_data(inputfile, outputfile):
	data = []
	with open(inputfile, 'r') as f:
		
		for line in f:
			data.append(line)
		f.close()
	random.shuffle(data)
	with open(outputfile, 'w') as f:
		f.writelines(data)
		f.close()


if __name__ == "__main__":
	main(sys.argv)

