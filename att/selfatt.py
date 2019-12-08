import tensorflow as tf 
import numpy as np
from basic import BASIC
import sys
sys.path.append('..')
# from capsule import capsnet
class SELF_ATT(BASIC):
	"""docstring for SELF_ATT"""
	def __init__(self, args):
		super(SELF_ATT, self).__init__(args)
		self.key_size = args.key_size
		self.value_size = args.value_size
		self.num_heads = args.num_heads
		self.batch_size = args.batch_size
		# self.pe = self.positional_encoding()
	def __call__(self, queries, keys, values, keep_prob=None):
		'''
		info:
			...
		args:
			embed_data: embed_data
			keep_prob: keep_prob
		returns:
			multi-head self-att layer
			shape with []
		'''
		# if value_size is not None:
		# 	self.value_size = value_size
		# output = self.multihead_attention(embed_data, keep_prob)
		# logits = self.foward_layer(output)
		# # logits = self.output_layer(flayer)
		# return logits
		q = self.split_heads(queries)
		k = self.split_heads(keys)
		v = self.split_heads(values)
		res = self.dot_product_attention(q, k, v, keep_prob)
		res = self.combine_heads(res)
		return res

	def dot_product_attention(self, q, k, v, keep_prob, bias=None):
		'''
		args:
			q: a tensor with shape [batch, length, channels]
			k: a tensor with shape [batch, length, channels]
			v: a tensor with shape [batch, length, channels]
			bias: bias
			keep_prob: keep_prob
		returns:
			a tensor
		'''
		with tf.variable_scope('dot_product_attention'):
			scalar = tf.rsqrt(tf.to_float(k.get_shape().as_list()[-1]))
			logits = tf.matmul(q * scalar, k, transpose_b = True)
			# logits = tf.matmul(q, k, transpose_b=True)
			if bias is not None:
				logits += bias
			weights = tf.nn.softmax(logits)
			if keep_prob is not None:
				 weights = tf.nn.dropout(weights, keep_prob)
			return tf.matmul(weights, v)
	def split_heads(self, inputs):
		'''
		args:
			inputs: a tensor with shape [batch, length, channels]
		returns:
			a tensor with shape [batch, num_heads, length, channels / num_heads] 
		'''
		data_shape = inputs.get_shape().as_list()
		# data = tf.reshape(inputs, data_shape[:-1] + [self.num_heads, data_shape[-1] // self.num_heads])
		# data = tf.reshape(inputs, [-1] + data_shape[1:-1] + [self.num_heads, data_shape[-1] // self.num_heads])
		data = tf.reshape(inputs, [-1] + data_shape[1:-1] + [self.num_heads, data_shape[-1] // self.num_heads])
		return tf.transpose(data, [0, 2, 1, 3])
	def combine_heads(self, inputs):
		'''
		args:
			inputs: a tensor with shape [batch, num_heads, length, channels / num_heads] 
		returns:
			a tensor with shape [batch, length, channels]
		'''
		data = tf.transpose(inputs, [0, 2, 1, 3])
		data_shape = data.get_shape().as_list()
		a, b = data_shape[-2:]
		# return tf.reshape(data, data_shape[:-2] + [a * b])
		return tf.reshape(data, [-1] + data_shape[1:-2] + [a * b])
	def compute_qkv(self, inputs_data):
		'''
		args:
			inputs: 
			key_size:
			value_size:
		returns:
			split q,k,v
		'''
		# shape = [1, inputs.get_shape().as_list()[-1], self.key_size * 2 + self.value_size]
		# filer = tf.get_variable("filer", shape)
		# res = tf.nn.conv1d(inputs, filer, stride=1, padding='VALID')
		with tf.variable_scope('compute_qkv'):
			q = tf.layers.conv1d(inputs=inputs_data, 
							filters=self.key_size, 
							kernel_size=1, 
							strides=1, 
							padding='valid', 
							kernel_initializer=tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
			k = tf.layers.conv1d(inputs=inputs_data, 
							filters=self.key_size, 
							kernel_size=1, 
							strides=1, 
							padding='valid', 
							kernel_initializer=tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
			v = tf.layers.conv1d(inputs=inputs_data, 
							filters=self.value_size, 
							kernel_size=1, 
							strides=1, 
							padding='valid', 
							kernel_initializer=tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
		return q,k,v
		# 	res = tf.layers.conv1d(inputs=inputs_data, 
		# 					filters=self.key_size * 2 + self.value_size, 
		# 					kernel_size=1, 
		# 					strides=1, 
		# 					padding='valid', 
		# 					kernel_initializer=tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
		# return tf.split(res, [self.key_size, self.key_size, self.value_size], axis=-1)
	def multihead_attention(self, queries, keep_prob):
		'''
		info:
			multihead_attention
		args:
			queries: a tensor with shape [batch, length, depth] 
			keep_prob: keep_prob
		return:
			a tensor with shape [batch, length, depth] 
		'''
		# print (queries)
		queries, keys, values = self.compute_qkv(queries)
		q = self.split_heads(queries)
		k = self.split_heads(keys)
		v = self.split_heads(values)
		res = self.dot_product_attention(q, k, v, keep_prob)
		com_res = self.combine_heads(res)
		return com_res
	def foward_layer(self, input_data):
		'''
		info:
			passing information between two layers
		args:
			input_data: the input data of this layer
		returns:
			...
		'''
		# return tf.reshape(input_data, [-1, self.max_length * self.hidden_size])
		# return tf.reshape(input_data, [-1, input_data.shape[-2] * input_data.shape[-1]])
		res = tf.layers.dense(inputs=input_data, 
						units = self.votrg_size,
						# activation = tf.nn.sigmoid,
						kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
		return res
	def graph_build(self):
		'''
		info:
			model graph
		'''
		with self.graph.as_default():
			inputs, target, ner, length, keep_prob = self.get_inputs()
			embed_data = self.input_layer(inputs)
			#embed_data = embed_data + self.positional_encoding(inputs)
			logits = self.multihead_attention(embed_data, keep_prob)
			logits = self.foward_layer(logits)
			# output = self.multihead_attention(embed_data, keep_prob)
			# logits = self.output_layer(self.foward_layer(output))
			# targets = self.data_smoothing(target)
			masks = tf.sequence_mask(length, self.max_length, dtype=tf.float32, name='masks')
			loss = tf.contrib.seq2seq.sequence_loss(logits, target, masks)
			# loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = tf.reshape(output,[480,2]), labels = target)
			# correct_prediction = tf.equal(tf.argmax(target,1),tf.argmax(logits,0))
			# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
			precision, recall, fmeasure, logit, intersection = self.metrics(logits, target, masks)
			# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=target))

			with tf.name_scope("optimization"):
				optimizer = tf.train.AdamOptimizer()
				gradients = optimizer.compute_gradients(loss)
				# gradients = optimizer.compute_gradients(regularization_cost)
				# print ('-----------')
				# print (len(gradients))
				# for g, v in gradients:
				grads = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in gradients if g is not None]
				train_op = optimizer.apply_gradients(grads)

		return inputs, target, ner, length, keep_prob, train_op, loss, precision, recall, fmeasure#, logits'

class SELF_ATT_PE(SELF_ATT):
	"""docstring for SELF_ATT"""
	# def __init__(self, args):
	# 	super(SELF_ATT, self).__init__(args)
	def graph_build(self):
		'''
		info:
			model graph
		'''
		with self.graph.as_default():
			inputs, target, ner, length, keep_prob = self.get_inputs()
			embed_data = self.input_layer(inputs)
			embed_data = embed_data + self.positional_encoding(inputs)
			logits = self.multihead_attention(embed_data, keep_prob)
			logits = self.foward_layer(logits)
			masks = tf.sequence_mask(length, self.max_length, dtype=tf.float32, name='masks')
			loss = tf.contrib.seq2seq.sequence_loss(logits, target, masks)
			precision, recall, fmeasure, logit, intersection = self.metrics(logits, target, masks)
			with tf.name_scope("optimization"):
				optimizer = tf.train.AdamOptimizer()
				gradients = optimizer.compute_gradients(loss)
				grads = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in gradients if g is not None]
				train_op = optimizer.apply_gradients(grads)

		return inputs, target, ner, length, keep_prob, train_op, loss, precision, recall, fmeasure#, logits'
