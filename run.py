import tensorflow as tf 
import helper
import hparams 
import time
# from lstm import lstms as lstm
from rnn import rnn as rnns
from att import selfatt as atts
# from gan import gans as gan
# from seq import nmt as nmt
# from seq import ptrnet as ptr
# from seq import msptr as ms
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
				if self.baseline < _f:
					self.baseline = _f
					saver.save(sess, "./model/" + self.model_name.lower() + "/" + self.task + "/model")
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
		
# class basic(object):
# 	"""docstring for basic"""
# 	def __init__(self, args):
# 		'''
# 		args:
# 			...
# 		'''
# 		# super(basic, self).__init__()
# 		self.scale = args.scale
# 		self.task = args.task
# 		self.epoch = args.epoch
# 		self.batch_size = args.batch_size
# 		# self.test_batch = args.test_batch
# 		self.train_prob = args.train_prob
# 		self.test_prob = args.test_prob
# 		self.sampling_size = args.sampling_size * args.sampling_scale
# 		self.config = tf.ConfigProto()
# 		self.result = [str(args)]
		
# 		self.dataset = hparams.datasets(self.scale, self.task)
# 		self.train_inputs, self.train_target, self.train_ner, self.train_length = helper.reader(self.dataset.train_source(), 
# 																				self.dataset.train_target(),
# 																				self.dataset.train_ner(),
# 																				self.dataset.train_length(), 
# 																				args.max_length)

# 		self.test_inputs, self.test_target, self.test_ner, self.test_length = helper.reader(self.dataset.test_source(), 
# 																			self.dataset.test_target(), 
# 																			self.dataset.test_ner(),
# 																			self.dataset.test_length(), 
# 																			args.max_length)
# 		self.test_size = len(self.test_length)
# 		self.train_size = len(self.train_length)

# 		if args.model == 'LSTM':
# 			self.model = lstm.LSTM(args)
# 		elif args.model == 'GRU':
# 			self.model = lstm.GRU(args)
# 		elif args.model == 'GRU_SATT':
# 			self.model = lstm.GRU_SATT(args)
# 		elif args.model == 'GRU_SATT_SINGMOID':
# 			self.model = lstm.GRU_SATT_SINGMOID(args)
# 		elif args.model == 'GRU_SATT_BI':
# 			self.model = lstm.GRU_SATT_BI(args)
# 		elif args.model == 'GRU_SATT_DOT':
# 			self.model = lstm.GRU_SATT_DOT(args)
# 		elif args.model == 'LSTM_ATT':
# 			self.model = lstm.LSTM_ATT(args)
# 		elif args.model == 'SELF_ATT':
# 			self.model = att.SELF_ATT(args)
# 		elif args.model == 'SELF_ATT_PE':
# 			self.model = att.SELF_ATT_PE(args)
# 		elif args.model == 'SELF_ATT_CAPS':
# 			self.model = att.SELF_ATT_CAPS(args)
# 		elif args.model == 'VOTE_SELF_ATT':
# 			self.model = att.VOTE_SELF_ATT(args)
# 		elif args.model == 'SELF_ATT_VOTE':
# 			self.model = att.SELF_ATT_VOTE(args)
# 		elif args.model == 'SVOTE_SELF_ATT':
# 			self.model = att.SVOTE_SELF_ATT(args)
# 		elif args.model == 'SELF_ATT_SVOTE':
# 			self.model = att.SELF_ATT_SVOTE(args)
# 		elif args.model == 'SELF_ATT_VOTE_CAPS_FAST':
# 			self.model = att.SELF_ATT_VOTE_CAPS_FAST(args)
# 		elif args.model == 'SELF_ATT_SVOTE_CAPS_FAST':
# 			self.model = att.SELF_ATT_SVOTE_CAPS_FAST(args)
# 		elif args.model == 'DOT_ATT':
# 			self.model = att.DOT_ATT(args)
		
# 	def run(self):
# 		'''
# 		ingo:
# 			train & test
# 		'''
# 		tmp = 0
# 		inputs, target, ner, length, keep_prob, train_op, loss, precision, recall, fmeasure = self.model.graph_build()
# 		self.train_inputs, self.train_target, self.train_ner, self.train_length = helper.sampling(self.sampling_size,
# 																							self.train_inputs, 
# 																							self.train_target,
# 																							self.train_ner, 
# 																							self.train_length)
# 		self.config.gpu_options.allow_growth=True
# 		# tf.reset_default_graph()
# 		with tf.Session(graph=self.model.graph, config=self.config) as sess:
# 			sess.run(tf.global_variables_initializer())
# 			for epoch in range(self.epoch):
# 				train_inputs, train_target, train_ner, train_length = helper.make_suffle(self.train_inputs, 
# 																			self.train_target,
# 																			self.train_ner, 
# 																			self.train_length)
# 				train_start_time = time.time()
# 				for train_inputs_batch, train_target_batch, train_ner_batch, train_length_batch in helper.make_batches(self.batch_size,
# 																									train_inputs, 
# 																									train_target, 
# 																									train_ner,
# 																									train_length):
# 					_ = sess.run([train_op],
# 						feed_dict={
# 							inputs: train_inputs_batch,
# 							target: train_target_batch,
# 							ner: train_ner_batch,
# 							length: train_length_batch,
# 							keep_prob: self.train_prob
# 						})
# 					tmp = tmp + 1
# 					# if tmp == 5:
# 					# 	print (res_logit)
# 				train_end_time = time.time()
# 				if self.test_size > self.batch_size:
# 					sp = self.test_size/self.batch_size
# 					cl = 0
# 					cp = 0
# 					cr = 0
# 					cf = 0
# 					for test_inputs_batch, test_target_batch, test_ner_batch, test_length_batch in helper.make_batches(self.batch_size, 
# 																									self.test_inputs, 
# 																									self.test_target,
# 																									self.test_ner, 
# 																									self.test_length):
# 						_l, _p, _r, _f = sess.run([loss, precision, recall, fmeasure],
# 							feed_dict={
# 								inputs: test_inputs_batch,
# 								target: test_target_batch,
# 								ner: test_ner_batch,
# 								length: test_length_batch,
# 								keep_prob: self.test_prob
# 							})
# 						cl += _l
# 						cp += _p
# 						cr += _r
# 						cf += _f

# 						# print (self.test_inputs)
# 						# print (self.test_target)
# 						# print (_logit)
# 					_l = cl/sp
# 					_p = cp/sp
# 					_r = cr/sp
# 					_f = 2*_p*_r/(_p+_r)
# 				else:
# 					_l, _p, _r, _f = sess.run([loss, precision, recall, fmeasure],
# 						feed_dict={
# 							inputs: self.test_inputs,
# 							target: self.test_target,
# 							ner: self.test_ner,
# 							length: self.test_length,
# 							keep_prob: self.test_prob
# 						})
# 					# print (self.test_inputs)
# 					# print (self.test_target)
# 					# print (_logit)
# 				test_end_time = time.time()
# 				train_time = train_end_time - train_start_time
# 				test_time = test_end_time - train_end_time
# 				self.result.append('epoch {:>3}, test_loss: {:>4.4f}, test_precision: {:>4.4f}, test_recall: {:>4.4f}, test_fmeasure: {:>4.4f}, train_time: {:>4.4f}, test_time: {:>4.4f}'.format(epoch, _l, _p, _r, _f, train_time, test_time))
# 				print(self.result[epoch+1])
# 			res_path = 'Result/' + str(self.task) + '/' + str(self.model) + '_' + str(self.scale) + '_' + time.strftime('%y%m%d%H%M',time.localtime(time.time())) + '.txt'
# 			helper.writer(res_path,result)
# class gans(basic):
# 	"""docstring for gans"""
# 	def __init__(self, args):
# 		super(gans, self).__init__(args)
# 		self.mode = args.model
# 		self.pre_epoch = args.pre_epoch
# 		if self.mode == 'GAN_BASIC':
# 			self.model = gan.GAN_BASIC(args)
# 		elif self.mode == 'GAN_SELF':
# 			self.model = gan.GAN_SELF(args)
# 	def run(self):
# 		'''
# 		info:
# 			for run
# 		'''
# 		if self.mode == 'GAN_BASIC':
# 			self.run_basic()
# 		elif self.mode == 'GAN_SELF':
# 			self.run_self()
# 	def run_basic(self):
# 		'''
# 		info:
# 			train & test for gans
# 		'''
# 		inputs, target, length, keep_prob, dis_op, gen_op, dis_loss, gen_loss, precision, recall, fmeasure, gen_data = self.model.graph_build()
# 		self.config.gpu_options.allow_growth=True

# 		with tf.Session(graph=self.model.graph, config=self.config) as sess:
# 			sess.run(tf.global_variables_initializer())
# 			# coord = tf.train.Coordinator()  
# 			# threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
# 			for epoch in range(self.epoch):
# 				train_inputs, train_target, train_length = helper.make_suffle(self.train_inputs, 
# 																			self.train_target, 
# 																			self.train_length)
# 				for train_inputs_batch, train_target_batch, train_length_batch in helper.make_batches(self.batch_size,
# 																									train_inputs, 
# 																									train_target, 
# 																									train_length):
# 					_ = sess.run([dis_op, gen_op],
# 						feed_dict={
# 							inputs: train_inputs_batch,
# 							target: train_target_batch,
# 							length: train_length_batch,
# 							keep_prob: self.train_prob
# 						})
# 				if self.test_size > self.batch_size:
# 					sp = self.test_size/self.batch_size
# 					cd = 0
# 					cg = 0
# 					cp = 0
# 					cr = 0
# 					cf = 0
# 					for test_inputs_batch, test_target_batch, test_length_batch in helper.make_batches(self.batch_size, 
# 																									self.test_inputs, 
# 																									self.test_target, 
# 																									self.test_length):
# 						_d, _g, _p, _r, _f, _gen_data = sess.run([dis_loss, gen_loss, precision, recall, fmeasure, gen_data],
# 							feed_dict={
# 								inputs: test_inputs_batch,
# 								target: test_target_batch,
# 								length: test_length_batch,
# 								keep_prob: self.test_prob
# 							})
# 						cd += _d
# 						cg += _g
# 						cp += _p
# 						cr += _r
# 					_d = cd/sp
# 					_g = cg/sp
# 					_p = cp/sp
# 					_r = cr/sp
# 					_f = 2*_p*_r/(_p+_r)
# 				else:
# 					_d, _g, _p, _r, _f, _gen_data = sess.run([dis_loss, gen_loss, precision, recall, fmeasure, gen_data],
# 						feed_dict={
# 							inputs: self.test_inputs,
# 							target: self.test_target,
# 							length: self.test_length,
# 							keep_prob: self.test_prob
# 						})
# 				self.result.append('epoch {:>3}, dis_loss: {:>4.4f}, gen_loss: {:>4.4f}, test_precision: {:>4.4f}, test_recall: {:>4.4f}, test_fmeasure: {:>4.4f}'.format(epoch, _d, _g, _p, _r, _f))
# 				print(self.result[epoch+1])
# 			# coord.request_stop()  
# 			# coord.join(threads) 
# 			res_path = 'Result/' + self.model + '_' + self.scale + '_' + time.strftime('%y%m%d%H%M',time.localtime(time.time())) + '.txt'
# 			helper.writer(res_path,result)
# 	def run_self(self):
# 			'''
# 			info:
# 				train & test for gans
# 			'''
# 			inputs, target, length, keep_prob, dis_op, gen_op, dis_loss, gen_loss, precision, recall, fmeasure, gen_data, pgen_op = self.model.graph_build()
# 			self.config.gpu_options.allow_growth=True

# 			with tf.Session(graph=self.model.graph, config=self.config) as sess:
# 				sess.run(tf.global_variables_initializer())
# 				# pretrain
# 				for prepoch in range(self.pre_epoch):
# 					train_inputs, train_target, train_length = helper.make_suffle(self.train_inputs, 
# 																				self.train_target, 
# 																				self.train_length)
# 					for train_inputs_batch, train_target_batch, train_length_batch in helper.make_batches(self.batch_size,
# 																										train_inputs, 
# 																										train_target, 
# 																										train_length):
# 						_ = sess.run([pgen_op,dis_op],
# 							feed_dict={
# 								inputs: train_inputs_batch,
# 								target: train_target_batch,
# 								length: train_length_batch,
# 								keep_prob: self.train_prob
# 							})
# 					if self.test_size > self.batch_size:
# 						sp = self.test_size/self.batch_size
# 						cd = 0
# 						cg = 0
# 						cp = 0
# 						cr = 0
# 						cf = 0
# 						for test_inputs_batch, test_target_batch, test_length_batch in helper.make_batches(self.batch_size, 
# 																										self.test_inputs, 
# 																										self.test_target, 
# 																										self.test_length):
# 							_d, _g, _p, _r, _f, _gen_data = sess.run([dis_loss, gen_loss, precision, recall, fmeasure, gen_data],
# 								feed_dict={
# 									inputs: test_inputs_batch,
# 									target: test_target_batch,
# 									length: test_length_batch,
# 									keep_prob: self.test_prob
# 								})
# 							cd += _d
# 							cg += _g
# 							cp += _p
# 							cr += _r
# 						_d = cd/sp
# 						_g = cg/sp
# 						_p = cp/sp
# 						_r = cr/sp
# 						_f = 2*_p*_r/(_p+_r)
# 					else:
# 						_d, _g, _p, _r, _f, _gen_data = sess.run([dis_loss, gen_loss, precision, recall, fmeasure, gen_data],
# 							feed_dict={
# 								inputs: self.test_inputs,
# 								target: self.test_target,
# 								length: self.test_length,
# 								keep_prob: self.test_prob
# 							})
# 					self.result.append('prepoch {:>3}, dis_loss: {:>4.4f}, gen_loss: {:>4.4f}, test_precision: {:>4.4f}, test_recall: {:>4.4f}, test_fmeasure: {:>4.4f}'.format(prepoch, _d, _g, _p, _r, _f))
# 					print(self.result[prepoch+1])
# 				# train
# 				for epoch in range(self.epoch):
# 					train_inputs, train_target, train_length = helper.make_suffle(self.train_inputs, 
# 																				self.train_target, 
# 																				self.train_length)
# 					for train_inputs_batch, train_target_batch, train_length_batch in helper.make_batches(self.batch_size,
# 																										train_inputs, 
# 																										train_target, 
# 																										train_length):
# 						_ = sess.run([gen_op, dis_op],
# 							feed_dict={
# 								inputs: train_inputs_batch,
# 								target: train_target_batch,
# 								length: train_length_batch,
# 								keep_prob: self.train_prob
# 							})
# 					if self.test_size > self.batch_size:
# 						sp = self.test_size/self.batch_size
# 						cd = 0
# 						cg = 0
# 						cp = 0
# 						cr = 0
# 						cf = 0
# 						for test_inputs_batch, test_target_batch, test_length_batch in helper.make_batches(self.batch_size,
# 																										self.test_inputs, 
# 																										self.test_target, 
# 																										self.test_length):
# 							_d, _g, _p, _r, _f, _gen_data = sess.run([dis_loss, gen_loss, precision, recall, fmeasure, gen_data],
# 								feed_dict={
# 									inputs: test_inputs_batch,
# 									target: test_target_batch,
# 									length: test_length_batch,
# 									keep_prob: self.test_prob
# 								})
# 							cd += _d
# 							cg += _g
# 							cp += _p
# 							cr += _r
# 						_d = cd/sp
# 						_g = cg/sp
# 						_p = cp/sp
# 						_r = cr/sp
# 						_f = 2*_p*_r/(_p+_r)
# 					else:
# 						_d, _g, _p, _r, _f, _gen_data = sess.run([dis_loss, gen_loss, precision, recall, fmeasure, gen_data],
# 							feed_dict={
# 								inputs: self.test_inputs,
# 								target: self.test_target,
# 								length: self.test_length,
# 								keep_prob: self.test_prob
# 							})
# 					self.result.append('epoch {:>3}, dis_loss: {:>4.4f}, gen_loss: {:>4.4f}, test_precision: {:>4.4f}, test_recall: {:>4.4f}, test_fmeasure: {:>4.4f}'.format(epoch, _d, _g, _p, _r, _f))
# 					print(self.result[epoch+self.pre_epoch+1])
# 				# coord.request_stop()  
# 				# coord.join(threads) 
# 				res_path = 'Result/' + self.model + '_' + self.scale + '_' + time.strftime('%y%m%d%H%M',time.localtime(time.time())) + '.txt'
# 				helper.writer(res_path,result)
# class seq2seq(basic):
# 	"""docstring for basic"""
# 	def __init__(self, args):
# 		'''
# 		args:
# 			...
# 		'''
# 		self.name = 'Nomal'
# 		super(seq2seq, self).__init__(args)

# 		# self.train_inputs, self.train_target, self.train_length = helper.reader(self.dataset.train_source(), 
# 		# 																		self.dataset.train_target(), 
# 		# 																		self.dataset.train_length(), 
# 		# 																		args.max_length)
# 		# self.test_inputs, self.test_target, self.test_length = helper.reader(self.dataset.test_source(), 
# 		# 																	self.dataset.test_target(), 
# 		# 																	self.dataset.test_length(), 
# 		# 																	args.max_length)

# 		# self.test_size = len(self.test_length)
# 		# self.train_size = len(self.train_length)

# 		if args.model == 'NMT':
# 			self.model = nmt.NMT(args)
# 		elif args.model == 'PTR_NET':
# 			self.model = ptr.PTR_NET(args)
# 			self.train_inputs, self.train_target, _, self.train_length = helper.ptr_reader(self.dataset.train_source(), 
# 																					self.dataset.train_target(), 
# 																					self.dataset.train_ner(),
# 																					self.dataset.train_length(), 
# 																					args.max_length)
# 			self.test_inputs, self.test_target, _, self.test_length = helper.ptr_reader(self.dataset.test_source(), 
# 																				self.dataset.test_target(), 
# 																				self.dataset.test_ner(),
# 																				self.dataset.test_length(), 
# 																				args.max_length)
# 		elif args.model == 'MS_PTR': 
# 			self.name = 'MS'
# 			self.model = ms.MS_PTR(args)
# 			self.train_inputs, self.train_target, self.train_ner, self.train_length = helper.ptr_reader(self.dataset.train_source(), 
# 																					self.dataset.train_target(), 
# 																					self.dataset.train_ner(),
# 																					self.dataset.train_length(), 
# 																					args.max_length)
# 			self.test_inputs, self.test_target, self.test_ner, self.test_length = helper.ptr_reader(self.dataset.test_source(), 
# 																				self.dataset.test_target(), 
# 																				self.dataset.test_ner(),
# 																				self.dataset.test_length(), 
# 																				args.max_length)
		
# 	def run(self):
# 		'''
# 		ingo:
# 			train & test
# 		'''
# 		if self.name != 'MS':
# 			inputs, target, length, keep_prob, train_op, loss, precision, recall, fmeasure = self.model.graph_build()
# 			# inputs, target, input_length, target_length, keep_prob, train_op, loss, precision, recall, fmeasure = self.model.graph_build()
# 			self.config.gpu_options.allow_growth=True
# 			# tf.reset_default_graph()
			
# 			with tf.Session(graph=self.model.graph, config=self.config) as sess:
# 				sess.run(tf.global_variables_initializer())
# 				for epoch in range(self.epoch):
# 					train_inputs, train_target, train_length = helper.make_suffle(self.train_inputs, 
# 																				self.train_target, 
# 																				self.train_length)
# 					train_start_time = time.time()
# 					for train_inputs_batch, train_target_batch, train_length_batch in helper.make_batches(self.batch_size,
# 																										train_inputs, 
# 																										train_target, 
# 																										train_length):
# 						_, _loss, _precision, _recall, _fmeasure = sess.run([train_op, loss, precision, recall, fmeasure],
# 							feed_dict={
# 								inputs: train_inputs_batch,
# 								target: train_target_batch,
# 								length: train_length_batch,
# 								keep_prob: self.train_prob
# 							})
# 						# print ('train:')
# 						# print (_loss, _precision, _recall, _fmeasure)
# 						# print (_targets)
# 					train_end_time = time.time()
# 					if self.test_size > self.batch_size:
# 						sp = self.test_size/self.batch_size
# 						cl = 0
# 						cp = 0
# 						cr = 0
# 						cf = 0
# 						for test_inputs_batch, test_target_batch, test_length_batch in helper.make_batches(self.batch_size, 
# 																										self.test_inputs, 
# 																										self.test_target, 
# 																										self.test_length):
# 							_l, _p, _r, _f = sess.run([loss, precision, recall, fmeasure],
# 								feed_dict={
# 									inputs: test_inputs_batch,
# 									target: test_target_batch,
# 									length: test_length_batch,
# 									keep_prob: self.train_prob
# 								})
# 							# print ('test:')
# 							# print (_l, _p, _r, _f)
# 							cl += _l
# 							cp += _p
# 							cr += _r
# 							cf += _f
# 							# print (_predict)
# 						_l = cl/sp
# 						_p = cp/sp
# 						_r = cr/sp
# 						_f = 2*_p*_r/(_p+_r)
# 					else:
# 						_l, _p, _r, _f = sess.run([loss, precision, recall, fmeasure],
# 							feed_dict={
# 								inputs: self.test_inputs,
# 								target: self.test_target,
# 								length: self.test_length,
# 								keep_prob: self.test_prob
# 							})
# 					test_end_time = time.time()
# 					train_time = train_end_time - train_start_time
# 					test_time = test_end_time - train_end_time
# 					self.result.append('epoch {:>3}, test_loss: {:>4.4f}, test_precision: {:>4.4f}, test_recall: {:>4.4f}, test_fmeasure: {:>4.4f}, train_time: {:>4.4f}, test_time: {:>4.4f}'.format(epoch, _l, _p, _r, _f, train_time, test_time))
# 					print(self.result[epoch+1])
# 				res_path = 'Result/' + str(self.task) + '/' + str(self.model) + '_' + str(self.scale) + '_' + time.strftime('%y%m%d%H%M',time.localtime(time.time())) + '.txt'
# 				helper.writer(res_path,result)
# 		else:
# 			# inputs, target, length, keep_prob, train_op, loss, precision, recall, fmeasure, targets = self.model.graph_build()
# 			inputs, target, ner, length, keep_prob, train_op, loss, precision, recall, fmeasure = self.model.graph_build()
# 			# inputs, target, input_length, target_length, keep_prob, train_op, loss, precision, recall, fmeasure = self.model.graph_build()
# 			self.config.gpu_options.allow_growth=True
# 			# tf.reset_default_graph()
# 			with tf.Session(graph=self.model.graph, config=self.config) as sess:
# 				sess.run(tf.global_variables_initializer())
# 				for epoch in range(self.epoch):
# 					train_inputs, train_target, train_ner, train_length = helper.make_suffle(self.train_inputs, 
# 																				self.train_target, 
# 																				self.train_ner,
# 																				self.train_length)
# 					train_start_time = time.time()
# 					for train_inputs_batch, train_target_batch, train_ner_batch, train_length_batch in helper.make_batches(self.batch_size,
# 																										train_inputs, 
# 																										train_target, 
# 																										train_ner,
# 																										train_length):
# 						# print (length)
# 						# print (train_length_batch)
# 						_, _loss, _precision, _recall, _fmeasure = sess.run([train_op, loss, precision, recall, fmeasure],
# 							feed_dict={
# 								inputs: train_inputs_batch,
# 								target: train_target_batch,
# 								ner: train_ner_batch,
# 								length: train_length_batch,
# 								keep_prob: self.train_prob
# 							})
# 						# print ('train:')
# 						# print (_loss, _precision, _recall, _fmeasure)
# 						# print (_targets)
# 					train_end_time = time.time()
# 					if self.test_size > self.batch_size:
# 						sp = self.test_size/self.batch_size
# 						cl = 0
# 						cp = 0
# 						cr = 0
# 						cf = 0
# 						for test_inputs_batch, test_target_batch, test_ner_batch, test_length_batch in helper.make_batches(self.batch_size, 
# 																										self.test_inputs, 
# 																										self.test_target, 
# 																										self.test_ner,
# 																										self.test_length):
# 							_l, _p, _r, _f = sess.run([loss, precision, recall, fmeasure],
# 								feed_dict={
# 									inputs: test_inputs_batch,
# 									target: test_target_batch,
# 									ner: test_ner_batch,
# 									length: test_length_batch,
# 									keep_prob: self.test_prob
# 								})
# 							# print ('test:')
# 							# print (_l, _p, _r, _f)
# 							cl += _l
# 							cp += _p
# 							cr += _r
# 							cf += _f
# 							# print (_predict)
# 						_l = cl/sp
# 						_p = cp/sp
# 						_r = cr/sp
# 						_f = 2*_p*_r/(_p+_r)
# 					else:
# 						_l, _p, _r, _f = sess.run([loss, precision, recall, fmeasure],
# 							feed_dict={
# 								inputs: self.test_inputs,
# 								target: self.test_target,
# 								ner: self.test_ner,
# 								length: self.test_length,
# 								keep_prob: self.test_prob
# 							})
# 					test_end_time = time.time()
# 					train_time = train_end_time - train_start_time
# 					test_time = test_end_time - train_end_time
# 					self.result.append('epoch {:>3}, test_loss: {:>4.4f}, test_precision: {:>4.4f}, test_recall: {:>4.4f}, test_fmeasure: {:>4.4f}, train_time: {:>4.4f}, test_time: {:>4.4f}'.format(epoch, _l, _p, _r, _f, train_time, test_time))
# 					print(self.result[epoch+1])
# 				res_path = 'Result/' + str(self.task) + '/' + str(self.model) + '_' + str(self.scale) + '_' + time.strftime('%y%m%d%H%M',time.localtime(time.time())) + '.txt'
# 				helper.writer(res_path,result)
# class felstm(basic):
# 	"""docstring for fe_lstm"""
# 	def __init__(self, args):
# 		super(felstm, self).__init__(args)
# 		# self.arg = arg
# 		if args.model == 'FE_LSTM_SOFT_ADA':
# 			self.train_tfidf, self.test_tfidf, self.train_ner, self.test_ner = helper.FE_reader_ada(self.dataset.train_source(),
# 																									self.dataset.test_source(),
# 																									self.dataset.train_ner(),
# 																									self.dataset.test_ner(),
# 																									self.dataset.v2v_path(),
# 																									args.max_length,
# 																									args.embed_size)
# 		else:
# 			self.train_inputs, self.test_inputs, self.train_tfidf, self.test_tfidf, self.train_ner, self.test_ner = helper.FE_reader(self.dataset.train_org(),
# 																																	self.dataset.test_org(),
# 																																	self.dataset.train_source(),
# 																																	self.dataset.test_source(),
# 																																	self.dataset.train_ner(),
# 																																	self.dataset.test_ner(),
# 																																	self.dataset.v2v_path(),
# 																																	args.max_length,
# 																																	args.embed_size)
# 		# print(self.train_tfidf[0])
# 		# print (self.train_ner[0])
# 		if args.model == 'FE_LSTM':
# 			self.model = lstm.FE_NET(args)
# 		elif args.model == 'FE_LSTM_SOFT':
# 			self.model = lstm.FE_NET_SOFT(args)
# 		elif args.model == 'FE_LSTM_SOFT_ADA':
# 			self.model = lstm.FE_NET_SOFT_ADA(args)
# 	def run(self):
# 		'''
# 		ingo:
# 			train & test
# 		'''
# 		inputs, target, length, tfidf, ner, keep_prob, train_op, loss, precision, recall, fmeasure, logits = self.model.graph_build()
		
# 		self.train_inputs, self.test_inputs, self.train_tfidf, self.test_tfidf, self.train_ner, self.test_ner = helper.sampling(self.sampling_size,
# 																																self.train_inputs, 
# 																																self.test_inputs, 
# 																																self.train_tfidf, 
# 																																self.test_tfidf, 
# 																																self.train_ner, 
# 																																self.test_ner)

# 		self.config.gpu_options.allow_growth=True
# 		# tf.reset_default_graph()
# 		with tf.Session(graph=self.model.graph, config=self.config) as sess:
# 			sess.run(tf.global_variables_initializer())
# 			for epoch in range(self.epoch):
# 				train_inputs, train_target, train_length, train_tfidf, train_ner = helper.make_suffle(self.train_inputs, 
# 																			self.train_target, 
# 																			self.train_length, 
# 																			self.train_tfidf,
# 																			self.train_ner)
# 				# print (len(train_inputs))
# 				train_start_time = time.time()
# 				for train_inputs_batch, train_target_batch, train_length_batch, train_tfidf_batch, train_ner_batch in helper.make_batches(self.batch_size,
# 																																			train_inputs, 
# 																																			train_target, 
# 																																			train_length,
# 																																			train_tfidf,
# 																																			train_ner):
# 					# print (inputs)
# 					# print (target)
# 					# print (length)
# 					# print (tfidf)
# 					# print (ner)
# 					# print (keep_prob)
# 					_ = sess.run([train_op],
# 						feed_dict={
# 							inputs: train_inputs_batch,
# 							target: train_target_batch,
# 							length: train_length_batch,
# 							tfidf: train_tfidf_batch,
# 							ner: train_ner_batch,
# 							keep_prob: self.train_prob
# 						})
# 				train_end_time = time.time()
# 				if self.test_size > self.batch_size:
# 					sp = self.test_size/self.batch_size
# 					cl = 0
# 					cp = 0
# 					cr = 0
# 					cf = 0
# 					for test_inputs_batch, test_target_batch, test_length_batch, test_tfidf_batch, test_ner_batch in helper.make_batches(self.batch_size, 
# 																																		self.test_inputs, 
# 																																		self.test_target, 
# 																																		self.test_length,
# 																																		self.test_tfidf,
# 																																		self.test_ner):
# 						_l, _p, _r, _logits = sess.run([loss, precision, recall, logits],
# 							feed_dict={
# 								inputs: test_inputs_batch,
# 								target: test_target_batch,
# 								length: test_length_batch,
# 								tfidf: test_tfidf_batch,
# 								ner: test_ner_batch,
# 								keep_prob: self.test_prob
# 							})
# 						# print ('----')
# 						# print (_logits)
# 						# print ('--')
# 						# print (test_target_batch)
# 						cl += _l
# 						cp += _p
# 						cr += _r
# 					_l = cl/sp
# 					_p = cp/sp
# 					_r = cr/sp
# 					_f = 2*_p*_r/(_p+_r)
# 				else:
# 					_l, _p, _r, _f = sess.run([loss, precision, recall, fmeasure],
# 						feed_dict={
# 							inputs: self.test_inputs,
# 							target: self.test_target,
# 							length: self.test_length,
# 							tfidf: test_tfidf_batch,
# 							ner: test_ner_batch,
# 							keep_prob: self.test_prob
# 						})
# 				test_end_time = time.time()
# 				train_time = train_end_time - train_start_time
# 				test_time = test_end_time - train_end_time
# 				self.result.append('epoch {:>3}, test_loss: {:>4.4f}, test_precision: {:>4.4f}, test_recall: {:>4.4f}, test_fmeasure: {:>4.4f}, train_time: {:>4.4f}, test_time: {:>4.4f}'.format(epoch, _l, _p, _r, _f, train_time, test_time))
# 				print(self.result[epoch+1])
# 			res_path = 'Result/' + str(self.task) + '/' + str(self.model) + '_' + str(self.scale) + '_' + time.strftime('%y%m%d%H%M',time.localtime(time.time())) + '.txt'
# 			helper.writer(res_path,result)
