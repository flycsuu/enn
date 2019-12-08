class datasets(object):
	"""docstring for datasets"""
	def __init__(self, scale, task):
		super(datasets, self).__init__()
		self.scale = scale
		self.task = task
	# train
	def train_org(self):
		return 'Datasets/' + self.task + '/' + self.scale + '/train_title.txt'
	def train_source(self):
		return 'Datasets/' + self.task + '/' + self.scale + '/train_title_int.txt'
	def train_target(self):
		if self.task == 'Compress':
			return 'Datasets/' + self.task + '/' + self.scale + '/train_tag.txt'
		elif self.task == 'Original':
			return 'Datasets/' + self.task + '/' + self.scale + '/train_orgtag.txt'
		else:
			return 'Datasets/' + self.task + '/' + self.scale + '/train_phrase.txt'
	def train_length(self):
		return 'Datasets/' + self.task + '/' + self.scale + '/train_length.txt'
	def train_ner(self):
		return 'Datasets/' + self.task + '/' + self.scale + '/train_ner_int.txt'
	# test
	def test_org(self):
		return 'Datasets/' + self.task + '/' + self.scale + '/test_title.txt'
	def test_source(self):
		return 'Datasets/' + self.task + '/' + self.scale + '/test_title_int.txt'
	def test_target(self):
		if self.task == 'Compress':
			return 'Datasets/' + self.task + '/' + self.scale + '/test_tag.txt'
		elif self.task == 'Original':
			return 'Datasets/' + self.task + '/' + self.scale + '/test_orgtag.txt'
		else:
			return 'Datasets/' + self.task + '/' + self.scale + '/test_phrase.txt'
	def test_length(self):
		return 'Datasets/' + self.task + '/' + self.scale + '/test_length.txt'
	def test_ner(self):
		return 'Datasets/' + self.task + '/' + self.scale + '/test_ner_int.txt'

	def v2v_path(self):
		return 'Datasets/' + self.task + '/' + self.scale + '/v2vocab.model'
	
# class params(object):
# 	"""docstring for params"""
# 	# def __init__(self, arg):
# 	# 	super(params, self).__init__()
# 	pad = 0
# 	unk = 1
# 	sos = 2
# 	eos = 3