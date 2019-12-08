import tensorflow as tf 
import numpy as np
class BASIC(object):
	"""docstring for BASIC"""
	def __init__(self, args):
		'''
		args:
			...
		'''
		# super(BASIC, self).__init__()
		self.embed_size = args.embed_size
		self.hidden_size = args.hidden_size
		self.batch_size = args.batch_size
		# self.keep_prob = args.keep_prob
		self.learning_rate = args.lr
		self.max_length = args.max_length
		self.vocab_size = args.vocab_size
		self.votrg_size = args.votrg_size
		self.max_grad_norm = args.max_grad_norm
		self.label_smoothing = args.label_smoothing
		self.graph = tf.Graph()
	def get_inputs(self):
		'''
		info: 
			get the inputs data
		returns:
			inputs: input data
			target: target data
			sequence_len: the length of the input/target data
		'''
		# inputs = tf.placeholder(tf.float32, [None, self.max_source_len])
		# target = tf.placeholder(tf.float32, [None, self.max_source_len])
		# inputs = tf.placeholder(tf.int32, [None, None], name = 'inputs')
		# inputs = tf.placeholder(tf.int32, [None, self.max_length], name = 'inputs')
		inputs = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name = 'inputs')
		target = tf.placeholder(tf.int32, [None, self.max_length], name = 'target')
		ner = tf.placeholder(tf.int32, [None, self.max_length], name = 'ner')
		length = tf.placeholder(tf.int32, (None, ), name = 'length')
		keep_prob = tf.placeholder(tf.float32, None, name = 'keep_prob')
		return inputs, target, ner, length, keep_prob
	def input_layer(self, input_data):
		'''
		info:
			input layer for embedding sequence
		args:
			input_data: input data
		returns:
			...
		'''
		with tf.variable_scope('input_layer'):
			# with tf.device("/cpu:0"):
			embed_sequence = tf.contrib.layers.embed_sequence(ids = input_data, 
															vocab_size = self.vocab_size, 
															embed_dim = self.embed_size, 
															initializer = tf.random_uniform_initializer(-0.25, 0.25, seed = 1),
															scope = 'embed')
		return embed_sequence
	def positional_encoding(self, input_data):
		'''
		info:
			PE
		'''
		N = self.batch_size
		T = self.max_length
		# N,T = input_data.get_shape().as_list()
		num_units = self.embed_size
		with tf.variable_scope('positional_encoding'):
			position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

		# First part of the PE function: sin and cos argument
		sinusoid_enc = np.array([
			[pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
			for pos in range(T)])

		# Second part, apply the cosine to even columns and sin to odds.
		sinusoid_enc[:, 0::2] = np.sin(sinusoid_enc[:, 0::2])  # dim 2i
		sinusoid_enc[:, 1::2] = np.cos(sinusoid_enc[:, 1::2])  # dim 2i+1

		# Convert to a tensor
		# lookup_table = tf.convert_to_tensor(position_enc)
		lookup_table = tf.cast(tf.convert_to_tensor(sinusoid_enc),tf.float32)
		# if zero_pad:
		# 	lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
		# 							lookup_table[1:, :]), 0)
		pe = tf.nn.embedding_lookup(lookup_table, position_ind)
		return pe
	def output_layer(self, input_data):
		'''
		info:
			the output of rnn layer
		args:
			input_data: the input data of this layer
		returns:
			...
		'''
		with tf.variable_scope('output_layer'):
			final_state = tf.layers.dense(inputs=input_data, 
									units = self.votrg_size*self.max_length,
									kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
		return tf.nn.softmax(tf.reshape(final_state, [-1, self.max_length, self.votrg_size]))
	def data_smoothing(self, input_data):
		'''
		info:
			label smoothing for target label
		args:
			input_data: ...
		returns:
			...
		'''
		return tf.cast((1.0-self.label_smoothing)*tf.cast(input_data, tf.float32) + self.label_smoothing/self.votrg_size, tf.float32)
	def metrics(self, logits, target, masks):
		'''
		info: 
			metrics the model
		args:
			logits: predict data
			target: ground truth
		returns:
			precision: count_correct/count_output
			recall: ount_correct/count_manual
			fmeasure: 2*precision*recall/(precision+recall)
		'''
		logits = tf.multiply(tf.cast(tf.argmax(logits, 2), tf.float32), masks)
		target = tf.cast(target,tf.float32)
		intersection = tf.multiply(logits, target)
		# modify by fuy 2019/04/15 begin
		# precision = tf.reduce_sum(intersection)/tf.reduce_sum(logits)
		precision = tf.cond(tf.reduce_sum(logits)>0, lambda:tf.reduce_sum(intersection)/tf.reduce_sum(logits),lambda:tf.reduce_sum(logits))
		recall = tf.reduce_sum(intersection)/tf.reduce_sum(target)
		# fmeasure = 0 if precision+recall == 0 else 2*precision*recall/(precision+recall)
		fmeasure = tf.cond(precision+recall>0, lambda: 2*precision*recall/(precision+recall), lambda:precision+recall)
		# modify by fuy 2019/04/15 end
		return precision, recall, fmeasure, logits, intersection