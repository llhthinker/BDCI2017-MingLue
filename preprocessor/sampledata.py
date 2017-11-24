#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys, getopt

"""
从原始文本从取前count个样本
"""

def main(argv):
	inputfile = ''
	outputfile = ''
	count = 0
	if len(argv) < 2:
		print(argv[0]+" -i <inputfile> -o <outputfile> -c <count>")

	try:
		opts, args = getopt.getopt(argv[1:], "hi:o:c:",
				 ["ifile=", "ofile", "count"])
	except getopt.GetoptError:
		print(argv[0]+" -i <inputfile> -o <outputfile> -c <count>")
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':	
			print(argv[0]+" -i <inputfile> -o <outputfile> -c <count>")
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile" ):
			outputfile = arg
		elif opt in ("-c", "--count"):
			count = int(arg)
	sampledata(inputfile, outputfile, count)

	
def sampledata(inputfile, outputfile, count):
	sample_data = []
	with open(inputfile, 'r') as f:
		i = 0
		for line in f:
			i += 1
			if i > count:
				break
			sample_data.append(line)
		f.close()

	with open(outputfile, 'w') as f:
		f.writelines(sample_data)
		f.close()


if __name__ == "__main__":
	main(sys.argv)

