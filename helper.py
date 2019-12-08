import random
import codecs
import math
# import gensim
# import numpy as np
from collections import Counter
# from hparams import params

def make_batches(size, *args):
	'''
	info:
		sampling/make batches for train
	args:
		size: sampling size
		args: org data
	returns:
		every sampling/batches data
	'''
	for times in range(len(args[0])//size):
		start = times * size
		yield tuple([item[start:start+size] for item in args])
def sampling(size, *args):
	'''
	info:
		sampling from org data
	args:
		size: sampling size
		args: org data
	returns:
		sampling data
	'''
	return tuple([item[:size] for item in args])
def make_suffle(*args):
	'''
	info:
		shffle the dataset
	args:
		org datasets
	returns:
		shffled dataset
	'''
	dataset = list(map(list, zip(*args)))
	random.shuffle(dataset)
	return list(map(list, zip(*dataset)))
def pad(sentences, max_length, symbol):
	'''
	info:
		pad sentence
	args:
		max_length: max length of sentence
		symbol: pad symbol
	returns:
		padded sentence
	'''
	result = []
	for sentence in sentences:
		if len(sentence) < max_length:
			sentence += [symbol for _ in range(max_length)][:max_length - len(sentence)]
		else:
			sentence = sentence[:max_length]
		result.append(sentence)
	return result
def reader(source, target, ner, length, max_length):
	'''
	info:
		read data
	args:
		source: source path
		target: target path
		length: length path
	returns:
		paded source data
	'''
	with codecs.open(source, 'r', 'utf-8') as fr:
		lines = fr.read().strip()
		source_data = []
		for line in lines.split('\n'):
			items = [int(item) for item in line.split(' ')]
			source_data.append(items)

	with codecs.open(target, 'r', 'utf-8') as fr:
		lines = fr.read().strip()
		target_data = []
		for line in lines.split('\n'):
			items = [float(item) for item in line.split(' ')]
			target_data.append(items)

	with codecs.open(ner, 'r', 'utf-8') as fr:
		lines = fr.read().strip()
		ner_data = []
		for line in lines.split('\n'):
			items = [float(item) for item in line.split(' ')]
			ner_data.append(items)

	with codecs.open(length, 'r', 'utf-8') as fr:
		lines = fr.read().strip()
		length_data = [int(float(item)) if int(float(item)) <= max_length else max_length for item in lines.split('\n')]
	
	return pad(source_data, max_length, 0), pad(target_data, max_length, 0), pad(ner_data, max_length, 0), length_data

def writer(path, data_lines):
	'''
	info:
		write data
	args:
		path: write path
		data_lines: datas defaut type []
	'''
	try:
		with codecs.open(path, 'w', 'utf-8') as fw:
			for line in data_lines:
				fw.write(line + '\n')
	except Exception as e:
		raise e
# ######################
# ######################

# def seq_reader(source, target, length, max_length):
# 	'''
# 	info:
# 		read data
# 	args:
# 		source: source path
# 		target: target path
# 		length: length path
# 	returns:
# 		paded source data
# 	'''
# 	with codecs.open(source, 'r', 'utf-8') as fr:
# 		lines = fr.read().strip()
# 		source_data = []
# 		for line in lines.split('\n'):
# 			items = [int(item) for item in line.split(' ')]
# 			source_data.append(items)

# 	with codecs.open(target, 'r', 'utf-8') as fr:
# 		lines = fr.read().strip()
# 		target_data = []
# 		for line in lines.split('\n'):
# 			items = [float(item) for item in line.split(' ')]
# 			target_data.append(items)

# 	targets = []
# 	target_len = []
# 	for source_items, target_items in zip(source_data, target_data):
# 		items = [item for index,item in enumerate(source_items) if target_items[index]]
# 		items.append(params.eos)
# 		targets.append(items)
# 		target_len.append(len(items))

# 	with codecs.open(length, 'r', 'utf-8') as fr:
# 		lines = fr.read().strip()
# 		length_data = [int(float(item)) if int(float(item)) <= max_length else max_length for item in lines.split('\n')]
# 	return pad(source_data, max_length, 0), pad(targets, max_length, 0), length_data, target_len

# class TFIDF(object):
# 	"""docstring for TFIDF"""
# 	def __init__(self, TRAIN_TOKEN, TEST_TOKEN, MAX_LEN):
# 		super(TFIDF, self).__init__()
# 		self.TRAIN_TOKEN = TRAIN_TOKEN
# 		self.TEST_TOKEN = TEST_TOKEN
# 		self.MAX_LEN = MAX_LEN
# 		self.train_lines, self.test_lines = self.read()
# 		self.Num = len(self.train_lines) + len(self.test_lines)
# 		self.vocab = {}
# 	def read(self):
# 		'''
# 		info:
# 			read file
# 		'''
# 		with codecs.open(self.TRAIN_TOKEN, 'r', 'utf-8') as fr:
# 			train_lines = fr.read().strip().split('\n')
# 		with codecs.open(self.TEST_TOKEN, 'r', 'utf-8') as fr:
# 			test_lines = fr.read().strip().split('\n')
# 		return train_lines, test_lines 
# 	def union(self, train_count, test_count):
# 		'''
# 		info:
# 			union
# 		'''
# 		return dict(Counter(train_count)+Counter(test_count))
# 	def count_train(self):
# 		'''
# 		info:
# 			count 
# 		'''
# 		# org_index = index
# 		# index = index*self.train_inc

# 		new_line = self.train_lines#[index:index+self.train_inc]
# 		new_line = (' '.join(new_line)).split()
# 		dic_vocb = Counter(new_line)
# 		return dic_vocb

# 	def count_test(self):
# 		'''
# 		info:
# 			count 
# 		'''
# 		new_line = self.test_lines
# 		new_line = (' '.join(new_line)).split()
# 		dic_vocb = Counter(new_line)
# 		return dic_vocb

# 	def tfidf_train(self):
# 		'''
# 		info:
# 			...
# 		'''
# 		res = []
# 		for line in self.train_lines:
# 			value = []
# 			words = line.split()

# 			for word in words:
# 				tf = 1/len(words)
# 				idf = math.log(self.Num/(1+self.vocab[str(word)]))
# 				tfidf = tf*idf
# 				value.append([tf,idf,tfidf])
# 			if len(value) <= self.MAX_LEN:
# 				for _ in range(len(value), self.MAX_LEN):
# 					value.append([0.,0.,0.])
# 			else:
# 				value = value[:self.MAX_LEN]
# 			res.append(value)
# 		return res 
# 	def tfidf_test(self):
# 		'''
# 		info:
# 			...
# 		'''
# 		res = []
# 		for line in self.test_lines:
# 			value = []
# 			words = line.split()
# 			for word in words:
# 				tf = 1/len(words)
# 				idf = math.log(self.Num/(1+self.vocab[str(word)]))
# 				tfidf = tf*idf
# 				value.append([tf,idf,tfidf])
# 			if len(value) <= self.MAX_LEN:
# 				for _ in range(len(value), self.MAX_LEN):
# 					value.append([0.,0.,0.])
# 			else:
# 				value = value[:self.MAX_LEN]
# 			res.append(value)
# 		return res 
# 	def run(self):
# 		'''
# 		info:
# 			...
# 		'''
# 		train_count = self.count_train()
# 		test_count = self.count_test()
# 		self.vocab = self.union(train_count, test_count)
# 		res_train = self.tfidf_train()
# 		res_test = self.tfidf_test()
# 		return res_train, res_test
# def FE_reader(train_org_path, test_org_path, train_int_path, test_int_path, train_ner_path, test_ner_path, v2v_path, max_length, embed_size):
# 	'''
# 	info:
# 		reader for Automatic Generation of Chinese Short Product Titiles for Mobile Display
# 		the model called FE_LSTM
# 	args:
# 		source: source path
# 		target: target path
# 	returns:
# 		...
# 	'''
# 	### tfidf
# 	tfidf = TFIDF(train_int_path, test_int_path, max_length)
# 	train_tfidf, test_tfidf = tfidf.run()
# 	### ner
# 	with codecs.open(train_ner_path, 'r', 'utf-8') as fr:
# 		lines = fr.read().strip()
# 		ner_data = []
# 		for line in lines.split('\n'):
# 			items = [int(item) for item in line.split(' ')]
# 			ner_data.append(items)
# 	train_ner = pad(ner_data, max_length, 0)

# 	with codecs.open(test_ner_path, 'r', 'utf-8') as fr:
# 		lines = fr.read().strip()
# 		ner_data = []
# 		for line in lines.split('\n'):
# 			items = [int(item) for item in line.split(' ')]
# 			ner_data.append(items)
# 	test_ner = pad(ner_data, max_length, 0)

# 	model = gensim.models.Word2Vec.load(v2v_path)
# 	with codecs.open(train_org_path, 'r', 'utf-8') as fr:
# 		lines = fr.read().strip().split('\n')
# 		vec_lines = []
# 		for line in lines:
# 			length = len(line.split())
# 			vec_line = []
# 			for data in line.split():
# 				try:
# 					vec = model[data].tolist()
# 				except:
# 					vec = [0.] * embed_size
# 				vec_line.append(vec)
# 			if length<15:
# 				for _ in range(length,15):
# 					vec_line.append([0.] * embed_size)
# 			vec_lines.append(vec_line)
# 		train_vec = vec_lines

# 	with codecs.open(test_org_path, 'r', 'utf-8') as fr:
# 		lines = fr.read().strip().split('\n')
# 		vec_lines = []
# 		for line in lines:
# 			length = len(line.split())
# 			vec_line = []
# 			for data in line.split():
# 				try:
# 					vec = model[data].tolist()
# 				except:
# 					vec = [0.] * embed_size
# 				vec_line.append(vec)
# 			if length<15:
# 				for _ in range(length,15):
# 					vec_line.append([0.] * embed_size)
# 			vec_lines.append(vec_line)
# 		test_vec = vec_lines

# 	return train_vec, test_vec, train_tfidf, test_tfidf, train_ner, test_ner

# def FE_reader_ada(train_int_path, test_int_path, train_ner_path, test_ner_path, v2v_path, max_length, embed_size):
# 	'''
# 	info:
# 		reader for Automatic Generation of Chinese Short Product Titiles for Mobile Display
# 		the model called FE_LSTM
# 	args:
# 		source: source path
# 		target: target path
# 	returns:
# 		...
# 	'''
# 	### tfidf
# 	tfidf = TFIDF(train_int_path, test_int_path, max_length)
# 	train_tfidf, test_tfidf = tfidf.run()
# 	### ner
# 	with codecs.open(train_ner_path, 'r', 'utf-8') as fr:
# 		lines = fr.read().strip()
# 		ner_data = []
# 		for line in lines.split('\n'):
# 			items = [int(item) for item in line.split(' ')]
# 			ner_data.append(items)
# 	train_ner = pad(ner_data, max_length, 0)

# 	with codecs.open(test_ner_path, 'r', 'utf-8') as fr:
# 		lines = fr.read().strip()
# 		ner_data = []
# 		for line in lines.split('\n'):
# 			items = [int(item) for item in line.split(' ')]
# 			ner_data.append(items)
# 	test_ner = pad(ner_data, max_length, 0)

# 	return train_tfidf, test_tfidf, train_ner, test_ner

# def ptr_reader(source, target, ner, length, max_length):
# 	'''
# 	info:
# 		read data
# 	args:
# 		source: source path
# 		target: target path
# 		length: length path
# 	returns:
# 		paded source data
# 	'''
# 	with codecs.open(source, 'r', 'utf-8') as fr:
# 		lines = fr.read().strip()
# 		source_data = []
# 		for line in lines.split('\n'):
# 			items = [int(item) for item in line.split(' ')]
# 			source_data.append(items)

# 	with codecs.open(ner, 'r', 'utf-8') as fr:
# 		lines = fr.read().strip()
# 		ner_data = []
# 		for line in lines.split('\n'):
# 			items = [float(item) for item in line.split(' ')]
# 			ner_data.append(items)

# 	with codecs.open(target, 'r', 'utf-8') as fr:
# 		lines = fr.read().strip()
# 		target_data = []
# 		for line in lines.split('\n'):
# 			items = [index+1 for index in range(len(line.split(' '))) if line.split(' ')[index] == '1.0']
# 			target_data.append(items)

# 	with codecs.open(length, 'r', 'utf-8') as fr:
# 		lines = fr.read().strip()
# 		length_data = [int(float(item)) if int(float(item)) <= max_length else max_length for item in lines.split('\n')]
# 	return pad(source_data, max_length, 0), pad(target_data, max_length, 0),  pad(ner_data, max_length, 0), length_data#, target_len

# # def FE_reader(source_path, target_path, v2v_path, max_length):
# # 	### train vec

# # 	target_vec = vec_lines

# # 	return source_vec, target_vec
# if __name__ == '__main__':
# 	TRAIN = 'Datasets/Compress/middle/train_title_int.txt' 
# 	TAG = 'Datasets/Compress/middle/train_tag.txt' 
# 	LEN = 'Datasets/Compress/middle/train_length.txt' 
# 	train_inputs, train_target, train_length = ptr_reader(TRAIN, TAG, LEN, 15)
# 	print (train_target[:10], train_length[:10])

