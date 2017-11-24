# -*- coding: UTF-8 -*-

import sys, getopt
import jieba 
"""
使用结巴对案情描述进行分词
可考虑优化
"""

def main(argv):
	inputfile = ''
	outputfile = ''
	if len(argv) < 2:
		print("Usage: "+argv[0]+" -i <inputfile> -o <outputfile>")
		sys.exit(2)
	try:
		opts, args = getopt.getopt(argv[1:], "hi:o:",
				 ["ifile=", "ofile="])
	except getopt.GetoptError:
		print("Usage: "+argv[0]+" -i <inputfile> -o <outputfile>")
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':	
			print("Usage: "+argv[0]+" -i <inputfile> -o <outputfile>")
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile" ):
			outputfile = arg
	if inputfile != '' and outputfile != '':
		seg(inputfile, outputfile)
	else:
		print("Usage: "+argv[0]+" -i <inputfile> -o <outputfile>")
	
def seg(inputfile, outputfile):
	seg_text = []	
	with open(inputfile, 'r') as f:
		for line in f:
			line_list = line.split('\t')
			line_list[1] = ' '.join(jieba.cut(line_list[1]))
			seg_text.append('\t'.join(line_list))
		f.close()

	with open(outputfile, 'w') as f:
		f.writelines(seg_text)
		f.close()


if __name__ == "__main__":
	jieba.enable_parallel(8)
	main(sys.argv)

