import tensorflow as tf 
from basic import BASIC
import sys
sys.path.append('..')
from att import selfatt as att

class RNN(BASIC):
	"""docstring for RNN"""
	def __init__(self, args):
		super(RNN, self).__init__(args)
	def graph_build(self):
		'''
		info:
			model graph
		'''
		raise NotImplementedError

class GRU_SATT(RNN):
	"""docstring for GRU"""
	def __init__(self, args):
		super(GRU_SATT, self).__init__(args)
		self.satt = att.SELF_ATT(args)
		# self.arg = arg

	def get_inputs(self):
		'''
		info: 
			get the inputs data
		returns:
			inputs: input data
			target: target data
			sequence_len: the length of the input/target data
		'''
		inputs = tf.placeholder(tf.int32, [None, self.max_length], name = 'inputs')
		target = tf.placeholder(tf.int32, [None, self.max_length], name = 'target')
		ner = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name = 'ner')
		length = tf.placeholder(tf.int32, (None, ), name = 'length')
		keep_prob = tf.placeholder(tf.float32, None, name = 'keep_prob')
		return inputs, target, ner, length, keep_prob

	def ner_feature(self, ner):
		'''
		info:
			semantic feature
		args:
			ner: ner
		'''
		v_ner = tf.one_hot(ner, 41)
		f_ner = tf.layers.dense(inputs=v_ner, 
						units = v_ner.shape[-1],
						kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

		# return tf.concat([f_tfidf,f_ner], axis=-1)
		return f_ner
	def get_gru_cell(self, keep_prob):
		'''
		info:
			get gru cell
		args:
			keep_prob: ...
		returns:
			gru cell
		'''	
		gru_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.hidden_size//2, kernel_initializer = tf.random_uniform_initializer(-0.1, 0.1, seed = 1)), 
																		input_keep_prob = keep_prob, 
																		output_keep_prob = keep_prob)
		return gru_cell
	def gru_layer(self, embed_input, source_sequence_length, keep_prob):
		'''
		inf: 
			rnn layer
		args:
			input_data: input data
			source_vocab_size: source_vocab_size
			source_sequence_length: source_sequence_length
		returns:
			final_state: the state of the rnn output, a tensor with shape [batch_size, word_depth]
		'''
		with tf.variable_scope('bigru_layer'):
			fw_cell = self.get_gru_cell(keep_prob)
			bw_cell = self.get_gru_cell(keep_prob)
			_output, _state = tf.nn.bidirectional_dynamic_rnn(
				fw_cell,
				bw_cell,
				embed_input,
				sequence_length = source_sequence_length,
				dtype = tf.float32,
				time_major = False)  
			return _output, _state

	def att_layer(self, embed_input, ner_input, source_sequence_length, keep_prob):
		'''
		info:
			all attention
		args:
			embed_input:
			ner_input:
			source_sequence_length:
			keep_prob:
		returns:
			logit results
		'''
		logits, _ = self.gru_layer(embed_input, source_sequence_length, keep_prob)
		res = self.satt(logits[0],logits[1], tf.concat(logits,-1))

		ner = self.ner_feature(ner_input)
		res = tf.layers.dense(inputs=tf.concat([res,ner], axis=-1), 
						units = self.hidden_size,
						activation = tf.nn.tanh,
						kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
		res = tf.layers.dense(inputs=res, 
						units = self.votrg_size,
						activation = tf.nn.softmax,
						kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
		return res

	def graph_build(self):
		'''
		info:
			model graph
		'''
		with self.graph.as_default():
			inputs, target, ner, length, keep_prob= self.get_inputs()
			embed_data = self.input_layer(inputs)
			logits = self.att_layer(embed_data, ner, length, keep_prob)

			masks = tf.sequence_mask(length, self.max_length, dtype=tf.float32, name='masks')
			loss = tf.contrib.seq2seq.sequence_loss(logits, target, masks)
			precision, recall, fmeasure, logit, intersection = self.metrics(logits, target, masks)

			with tf.name_scope("optimization"):
				optimizer = tf.train.AdamOptimizer()
				gradients = optimizer.compute_gradients(loss)
				grads = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in gradients if g is not None]
				train_op = optimizer.apply_gradients(grads)
			saver = tf.train.Saver()
		return inputs, target, ner, length, keep_prob, train_op, loss, precision, recall, fmeasure, saver

class GRU_SATT_SMD(GRU_SATT):
	"""docstring for GRU_SATT_2"""
	def __init__(self, args):
		super(GRU_SATT_SMD, self).__init__(args)
		self.tau = args.tau
	def metrics(self, logits, target):
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
		logits = tf.cast(logits, tf.float32)
		target = tf.cast(target, tf.float32)
		intersection = tf.multiply(logits, target)
		# modify by fuy 2019/04/15 begin
		# precision = tf.reduce_sum(intersection)/tf.reduce_sum(logits)
		precision = tf.cond(tf.reduce_sum(logits)>0, lambda:tf.reduce_sum(intersection)/tf.reduce_sum(logits),lambda:tf.reduce_sum(logits))
		recall = tf.reduce_sum(intersection)/tf.reduce_sum(target)
		# fmeasure = 0 if precision+recall == 0 else 2*precision*recall/(precision+recall)
		fmeasure = tf.cond(precision+recall>0, lambda: 2*precision*recall/(precision+recall), lambda:precision+recall)
		# modify by fuy 2019/04/15 end
		return precision, recall, fmeasure, logits, intersection

	def att_layer(self, embed_input, ner_input, source_sequence_length, keep_prob):
		'''
		info:
			all attention
		args:
			embed_input:
			ner_input:
			source_sequence_length:
			keep_prob:
		returns:
			logit results
		'''
		logits, _ = self.gru_layer(embed_input, source_sequence_length, keep_prob)
		res = self.satt(logits[0],logits[1], tf.concat(logits,-1))
		ner = self.ner_feature(ner_input)
		res = tf.layers.dense(inputs=tf.concat([res,ner], axis=-1), 
						units = self.hidden_size,
						activation = tf.nn.tanh,
						kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
		res = tf.layers.dense(inputs=res, 
						units = 1,
						# activation = tf.nn.sigmoid,
						kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
		res = tf.squeeze(res)
		return res
	def graph_build(self):
		'''
		info:
			model graph
		'''
		with self.graph.as_default():
			inputs, target, ner, length, keep_prob= self.get_inputs()
			embed_data = self.input_layer(inputs)
			logits = self.att_layer(embed_data, ner, length, keep_prob)

			loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(target,tf.float32)))

			zero = tf.zeros_like(logits)
			one = tf.ones_like(logits)
			logits = tf.where(tf.nn.sigmoid(logits) < self.tau, x=zero, y=one)
			precision, recall, fmeasure, logit, intersection = self.metrics(logits, target)

			# masks = tf.sequence_mask(length, self.max_length, dtype=tf.float32, name='masks')
			# loss = tf.contrib.seq2seq.sequence_loss(logits, target, masks)
			# precision, recall, fmeasure, logit, intersection = self.metrics(logits, target, masks)

			with tf.name_scope("optimization"):
				optimizer = tf.train.AdamOptimizer()
				gradients = optimizer.compute_gradients(loss)
				grads = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in gradients if g is not None]
				train_op = optimizer.apply_gradients(grads)
			saver = tf.train.Saver()
		return inputs, target, ner, length, keep_prob, train_op, loss, precision, recall, fmeasure, saver

class GRU(GRU_SATT):
	"""docstring for GRU"""
	def __init__(self, args):
		super(GRU, self).__init__(args)

	def logit_layer(self, embed_input, source_sequence_length, keep_prob):
		'''
		info:
			bigru
		args:
			embed_input:
			ner_input:
			source_sequence_length:
			keep_prob:
		'''
		logits, _ = self.gru_layer(embed_input, source_sequence_length, keep_prob)
		res = tf.layers.dense(inputs=tf.concat(logits,axis=-1), 
						units = self.hidden_size,
						activation = tf.nn.tanh,
						kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
		res = tf.layers.dense(inputs=res, 
						units = self.votrg_size,
						activation = tf.nn.softmax,
						kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
		return res
	def graph_build(self):
		'''
		info:
			model graph
		'''
		with self.graph.as_default(): 
			inputs, target, ner, length, keep_prob= self.get_inputs()
			embed_data = self.input_layer(inputs)
			logits = self.logit_layer(embed_data, length, keep_prob)

			masks = tf.sequence_mask(length, self.max_length, dtype=tf.float32, name='masks')
			loss = tf.contrib.seq2seq.sequence_loss(logits, target, masks)
			precision, recall, fmeasure, logit, intersection = self.metrics(logits, target, masks)

			with tf.name_scope("optimization"):
				optimizer = tf.train.AdamOptimizer()
				gradients = optimizer.compute_gradients(loss)
				grads = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in gradients if g is not None]
				train_op = optimizer.apply_gradients(grads)
		return inputs, target, ner, length, keep_prob, train_op, loss, precision, recall, fmeasure