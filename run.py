import tensorflow as tf 
import helper
import hparams 
import time
from rnn import rnn as rnns
from att import selfatt as atts
class runbasic(object):
	"""docstring for runbasic"""
	def __init__(self, args):
		super(runbasic, self).__init__()
		self.scale = args.scale
		self.task = args.task
		self.epoch = args.epoch
		self.batch_size = args.batch_size
		self.train_prob = args.train_prob
		self.test_prob = args.test_prob
		self.sampling_size = args.sampling_size * args.sampling_scale
		self.config = tf.ConfigProto()
		self.result = [str(args)]
		self.model_name = args.model
		self.dataset = hparams.datasets(self.scale, self.task)
		self.train_inputs, self.train_target, self.train_ner, self.train_length = helper.reader(self.dataset.train_source(), 
																				self.dataset.train_target(),
																				self.dataset.train_ner(),
																				self.dataset.train_length(), 
																				args.max_length)

		self.test_inputs, self.test_target, self.test_ner, self.test_length = helper.reader(self.dataset.test_source(), 
																			self.dataset.test_target(), 
																			self.dataset.test_ner(),
																			self.dataset.test_length(), 
																			args.max_length)
	def run():
		'''
		info:
			train & test
		'''
		raise NotImplementedError
class rnn(runbasic):
	"""docstring for rnn"""
	def __init__(self, args):
		super(rnn, self).__init__(args)
		# self.args = args
		if self.model_name == 'GRU':
			self.model = rnns.GRU(args)
		elif self.model_name == 'GRU_SATT':
			self.model = rnns.GRU_SATT(args)
		elif self.model_name == 'GRU_SATT_SMD':
			self.model = rnns.GRU_SATT_SMD(args)
		else:
			raise Exception("Err:chema does not match model")
		self.test_size = len(self.test_length)
		self.train_size = len(self.train_length)
		self.baseline = 0
	def run(self):
		'''
		ingo:
			train & test
		'''
		inputs, target, ner, length, keep_prob, train_op, loss, precision, recall, fmeasure, saver = self.model.graph_build()
		self.train_inputs, self.train_target, self.train_ner, self.train_length = helper.sampling(self.sampling_size,
																							self.train_inputs, 
																							self.train_target,
																							self.train_ner, 
																							self.train_length)
		self.config.gpu_options.allow_growth=True
		# tf.reset_default_graph()
		
		self.saveb_aseline = 0
		with tf.Session(graph=self.model.graph, config=self.config) as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(self.epoch):
				train_inputs, train_target, train_ner, train_length = helper.make_suffle(self.train_inputs, 
																			self.train_target,
																			self.train_ner, 
																			self.train_length)
				train_start_time = time.time()
				for train_inputs_batch, train_target_batch, train_ner_batch, train_length_batch in helper.make_batches(self.batch_size,
																									train_inputs, 
																									train_target, 
																									train_ner,
																									train_length):
					_ = sess.run([train_op],
						feed_dict={
							inputs: train_inputs_batch,
							target: train_target_batch,
							ner: train_ner_batch,
							length: train_length_batch,
							keep_prob: self.train_prob
						})
				train_end_time = time.time()
				if self.test_size > self.batch_size:
					sp = self.test_size/self.batch_size
					cl = 0
					cp = 0
					cr = 0
					cf = 0
					for test_inputs_batch, test_target_batch, test_ner_batch, test_length_batch in helper.make_batches(self.batch_size, 
																									self.test_inputs, 
																									self.test_target,
																									self.test_ner, 
																									self.test_length):
						_l, _p, _r, _f = sess.run([loss, precision, recall, fmeasure],
							feed_dict={
								inputs: test_inputs_batch,
								target: test_target_batch,
								ner: test_ner_batch,
								length: test_length_batch,
								keep_prob: self.test_prob
							})
						cl += _l
						cp += _p
						cr += _r
						cf += _f
					_l = cl/sp
					_p = cp/sp
					_r = cr/sp
					_f = 2*_p*_r/(_p+_r)
				else:
					_l, _p, _r, _f = sess.run([loss, precision, recall, fmeasure],
						feed_dict={
							inputs: self.test_inputs,
							target: self.test_target,
							ner: self.test_ner,
							length: self.test_length,
							keep_prob: self.test_prob
						})
				test_end_time = time.time()
				train_time = train_end_time - train_start_time
				test_time = test_end_time - train_end_time
				self.result.append('epoch {:>3}, test_loss: {:>4.4f}, test_precision: {:>4.4f}, test_recall: {:>4.4f}, test_fmeasure: {:>4.4f}, train_time: {:>4.4f}, test_time: {:>4.4f}'.format(epoch, _l, _p, _r, _f, train_time, test_time))
				print(self.result[epoch+1])
			res_path = 'Result/' + str(self.task) + '/' + str(self.model) + '_' + str(self.scale) + '_' + time.strftime('%y%m%d%H%M',time.localtime(time.time())) + '.txt'
			helper.writer(res_path,result)

class att(runbasic):
	"""docstring for att"""
	def __init__(self, args):
		super(att, self).__init__(args)
		# self.args = args
		if self.model_name == 'SELF_ATT':
			self.model = atts.SELF_ATT(args)
		elif self.model_name == 'SELF_ATT_PE':
			self.model = atts.SELF_ATT_PE(args)
		else:
			raise Exception("Err:chema does not match model")
		self.test_size = len(self.test_length)
		self.train_size = len(self.train_length)
	def run(self):
		'''
		ingo:
			train & test
		'''
		inputs, target, ner, length, keep_prob, train_op, loss, precision, recall, fmeasure = self.model.graph_build()
		self.train_inputs, self.train_target, self.train_ner, self.train_length = helper.sampling(self.sampling_size,
																							self.train_inputs, 
																							self.train_target,
																							self.train_ner, 
																							self.train_length)
		self.config.gpu_options.allow_growth=True
		# tf.reset_default_graph()
		with tf.Session(graph=self.model.graph, config=self.config) as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(self.epoch):
				train_inputs, train_target, train_ner, train_length = helper.make_suffle(self.train_inputs, 
																			self.train_target,
																			self.train_ner, 
																			self.train_length)
				train_start_time = time.time()
				for train_inputs_batch, train_target_batch, train_ner_batch, train_length_batch in helper.make_batches(self.batch_size,
																									train_inputs, 
																									train_target, 
																									train_ner,
																									train_length):
					_ = sess.run([train_op],
						feed_dict={
							inputs: train_inputs_batch,
							target: train_target_batch,
							ner: train_ner_batch,
							length: train_length_batch,
							keep_prob: self.train_prob
						})
				train_end_time = time.time()
				if self.test_size > self.batch_size:
					sp = self.test_size/self.batch_size
					cl = 0
					cp = 0
					cr = 0
					cf = 0
					for test_inputs_batch, test_target_batch, test_ner_batch, test_length_batch in helper.make_batches(self.batch_size, 
																									self.test_inputs, 
																									self.test_target,
																									self.test_ner, 
																									self.test_length):
						_l, _p, _r, _f = sess.run([loss, precision, recall, fmeasure],
							feed_dict={
								inputs: test_inputs_batch,
								target: test_target_batch,
								ner: test_ner_batch,
								length: test_length_batch,
								keep_prob: self.test_prob
							})
						cl += _l
						cp += _p
						cr += _r
						cf += _f
					_l = cl/sp
					_p = cp/sp
					_r = cr/sp
					_f = 2*_p*_r/(_p+_r)
				else:
					_l, _p, _r, _f = sess.run([loss, precision, recall, fmeasure],
						feed_dict={
							inputs: self.test_inputs,
							target: self.test_target,
							ner: self.test_ner,
							length: self.test_length,
							keep_prob: self.test_prob
						})

				test_end_time = time.time()
				train_time = train_end_time - train_start_time
				test_time = test_end_time - train_end_time
				self.result.append('epoch {:>3}, test_loss: {:>4.4f}, test_precision: {:>4.4f}, test_recall: {:>4.4f}, test_fmeasure: {:>4.4f}, train_time: {:>4.4f}, test_time: {:>4.4f}'.format(epoch, _l, _p, _r, _f, train_time, test_time))
				print(self.result[epoch+1])
			res_path = 'Result/' + str(self.task) + '/' + str(self.model) + '_' + str(self.scale) + '_' + time.strftime('%y%m%d%H%M',time.localtime(time.time())) + '.txt'
			helper.writer(res_path,result)
