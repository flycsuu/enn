import tensorflow as tf 
import hparams
import helper
import time
import argparse
class load(object):
	"""docstring for load"""
	def __init__(self, args):
		super(load, self).__init__()

		self.max_length = args.max_length
		self.scale = args.scale
		self.task = args.task
		self.model_name = args.model
		self.dataset = hparams.datasets(self.scale, self.task)
		self.test_inputs, self.test_target, self.test_ner, self.test_length = helper.reader(self.dataset.test_source(), 
																			self.dataset.test_target(), 
																			self.dataset.test_ner(),
																			self.dataset.test_length(), 
																			self.max_length)
		self.test_size = len(self.test_length)
		self.batch_size = args.batch_size
		self.test_prob = args.test_prob
	def run(self):
		'''
		info:
			train & test
		'''
		with tf.Session() as sess:
			graph_path = 'model/{}/{}/model.meta'.format(self.model_name.lower(),self.task)
			restore_path = 'model/{}/{}/'.format(self.model_name.lower(),self.task)
			saver = tf.train.import_meta_graph(graph_path)
			saver.restore(sess,tf.train.latest_checkpoint(restore_path))
			graph= tf.get_default_graph()
			# inputs
			inputs = graph.get_tensor_by_name('inputs:0')
			target = graph.get_tensor_by_name('target:0')
			ner = graph.get_tensor_by_name('ner:0')
			length = graph.get_tensor_by_name('length:0')
			keep_prob = graph.get_tensor_by_name('keep_prob:0')
			# outputs
			#loss = graph.get_tensor_by_name('sequence_loss/truediv:0')
			precision = graph.get_tensor_by_name('cond/Merge:0')
			recall = graph.get_tensor_by_name('truediv:0')
			fmeasure = graph.get_tensor_by_name('cond_1/Merge:0')

			test_begin_time = time.time()
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
					_p, _r, _f = sess.run([precision, recall, fmeasure],
						feed_dict={
							inputs: test_inputs_batch,
							target: test_target_batch,
							ner: test_ner_batch,
							length: test_length_batch,
							keep_prob: self.test_prob
						})
					cp += _p
					cr += _r
					cf += _f
				_p = cp/sp
				_r = cr/sp
				_f = 2*_p*_r/(_p+_r)
			else:
				_p, _r, _f = sess.run([precision, recall, fmeasure],
					feed_dict={
						inputs: self.test_inputs,
						target: self.test_target,
						ner: self.test_ner,
						length: self.test_length,
						keep_prob: self.test_prob
					})

			test_end_time = time.time()
			test_time = test_end_time - test_begin_time 
			print('test_precision: {:>4.4f}, test_recall: {:>4.4f}, test_fmeasure: {:>4.4f}, test_time: {:>4.4f}'.format(_p, _r, _f, test_time))
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='keyphrase args')
	# defaut setting
	parser.add_argument('-model', type=str, default='GRU_SATT_SMD', help='Model name [default: {rnn: GRU_SATT/GRU_SATT_SINGMOID/GRU, att: SELF_ATT/SELF_ATT_PE}]')
	parser.add_argument('-max_length', type=int, default=15, help='sequence maximum length [default: 30]')
	parser.add_argument('-batch_size', type=int, default=64 , help='initial batch size [default: 64]')
	parser.add_argument('-task', type=str, default='Compress', help='default task phrase [default: Phrase/Phrase5/Phrase7/Original/Compress]' )
	parser.add_argument('-scale', type=str, default='middle', help='default scale middle [default: small/middle]' )
	parser.add_argument('-test_prob', type=float, default=1, help='the probability for dropout [default: 1]')

	args = parser.parse_args()
	load(args).run()