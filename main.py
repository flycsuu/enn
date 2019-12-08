import run
import argparse
def main(args):
	'''
	args：
		...
	'''
	if args.schema == 'rnn':
		op = run.rnn(args)
	elif args.schema == 'att':
		op = run.att(args)
	elif args.schema == 'load':
		op = load.load(args)
	op.run()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='keyphrase args')
	# learning
	parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
	parser.add_argument('-max_grad_norm', type=int, default=5, help='the max_grad_norm [default: 5]')
	parser.add_argument('-epsilon', type=float, default=1e-9, help='the epsilon [default: 1e-9]')
	parser.add_argument('-tau', type=float, default=0.4, help='the tau [default: 0.4]')

	# data
	parser.add_argument('-embed_size', type=int, default=100, help='initial embedding size [default: 300]')
	parser.add_argument('-max_length', type=int, default=15, help='sequence maximum length [default: 30]')
	# parser.add_argument('-vocab_size', type=int, default=50000, help='initial vocabulary size [default:50000]')
	parser.add_argument('-vocab_size', type=int, default=150000, help='initial vocabulary size [default:50000]')
	parser.add_argument('-votrg_size', type=int, default=2, help='initial target vocabulary size [default:2]')

	# model
	parser.add_argument('-schema', type=str, default='rnn', help='schema name [default: rnn/att]')
	parser.add_argument('-model', type=str, default='GRU_SATT', help='Model name [default: {rnn: GRU_SATT/GRU_SATT_SMD/GRU, att: SELF_ATT/SELF_ATT_PE}]')
	parser.add_argument('-hidden_size', type=int, default=512, help='initial hidden size [default: 512]')

	# train
	parser.add_argument('-epoch', type=int, default=25, help='train epoch [default: 25]')
	parser.add_argument('-batch_size', type=int, default=64 , help='initial batch size [default: 64]')
	parser.add_argument('-task', type=str, default='Phrase', help='default task phrase [default: Phrase/Phrase5/Phrase7/Original/Compress]' )
	parser.add_argument('-scale', type=str, default='middle', help='default scale middle [default: small/middle]' )
	parser.add_argument('-train_prob', type=float, default=0.8, help='the probability for dropout [default: 0.8]')
	parser.add_argument('-test_prob', type=float, default=1, help='the probability for dropout [default: 1]')
	parser.add_argument('-label_smoothing', type=float, default=0.1, help='initial label smoothing [default: 0.1]')
	parser.add_argument('-sampling_size', type=int, default=50000, help='quantity of sampling unit [default: 50000]')
	parser.add_argument('-sampling_scale', type=int, default=10, help='scale of sampling unit [default: 1]')

	#self_attenton
	parser.add_argument('-key_size', type=int, default=512, help='query and key size [default: 300]')
	parser.add_argument('-value_size', type=int, default=512, help='value size [default: 300]')
	parser.add_argument('-num_heads', type=int, default=8, help='initial model heads [default: 8]')

	args = parser.parse_args()
	print (str(args))
	main(args)