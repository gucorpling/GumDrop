import re, io, os, sys, time
from datetime import timedelta
from argparse import ArgumentParser
from glob import glob
# Allow package level imports in module
script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "..")
ncrf_dir = script_dir + os.sep + "NCRFpp"
vecdir = lib + os.sep + ".." + os.sep + "vec"
sys.path.append(lib)
sys.path.append(ncrf_dir)
from conll_reader import read_conll, feats2rnn
from tune import get_best_params
from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, STATUS_FAIL
from hyperopt.pyll.base import scope
from copy import deepcopy
import numpy as np
import random
import torch

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic=True

if __name__ =="__main__":
	from NCRFpp.main import dispatch as ncrf
	from NCRFpp.utils.data import Data
else:
	from .NCRFpp.main import dispatch as ncrf
	from .NCRFpp.utils.data import Data


class RNNSegmenter:

	def __init__(self,model="eng.rst.gum",genre_pat="^(..)",load_params=True,auto="",conn=False):
		self.name = "RNNSegmenter"
		self.model = model
		self.corpus_dir = ncrf_dir + os.sep + "data" + os.sep + model
		self.auto = auto
		self.conn = conn
		if conn:
			self.segtype="conn"
		else:
			self.segtype="seg"
		if not os.path.isdir(self.corpus_dir):
			os.mkdir(self.corpus_dir)
		self.make_configs(corpus=model,load_params=load_params,conn=conn)
		self.ext = "bmes"  # use bio for bio encoding
		if "gum" in model:
			genre_pat = "GUM_([^_]+)_"
		self.genre_pat = genre_pat

	def make_configs(self,corpus,load_params=False,conn=False):

		lang = self.model.split(".")[0]

		if lang == "zho":
			vec = "cc.zho.300.vec_trim.vec"
		elif lang == "eng":
			vec = "glove.6B.300d_trim.vec"
		else:
			vec = "wiki.**lang**.vec_trim.vec".replace("**lang**",lang)

		decode = io.open(script_dir + os.sep + self.segtype + ".decode.config").read()
		train = io.open(script_dir + os.sep + self.segtype + ".train.config").read()

		if load_params:
			if self.auto != "":
				sys.stderr.write("! WARN: loading gold best params but processing auto parse data\n")
			_, params, _ = get_best_params(corpus,self.name)
			decode = self.set_params(params,decode)
			train = self.set_params(params,train)

		if self.auto == "_auto":
			train = train.replace("lstmcrf","lstmcrf_auto")
			decode = decode.replace("lstmcrf","lstmcrf_auto")

		with io.open(ncrf_dir + os.sep + self.segtype + ".train"+self.auto+".config",'w',newline="\n") as f:
			f.write(train.replace("**corpus**",corpus).replace("**vec**",vec).replace("**vecdir**",vecdir).replace("**ncrfdir**",ncrf_dir))

		with io.open(ncrf_dir + os.sep + self.segtype + ".decode"+self.auto+".config",'w',newline="\n") as f:
			f.write(decode.replace("**corpus**",corpus).replace("**vec**",vec).replace("**vecdir**",vecdir).replace("**ncrfdir**",ncrf_dir))

	def set_params(self,params,config):
		output = []
		lines = config.split("\n")
		for line in lines:
			for key,val in params.items():
				if line.startswith(key+"="):
					line = line.split("=")[0] + "=" + str(val)
					break
			output.append(line)

		return "\n".join(output)

	def train(self,trainfile, devfile, multifolds=1,as_text=False):

		p = StdOutFilter()
		p.start()

		train_feats, _, _, _, _ = read_conll(trainfile,genre_pat=self.genre_pat,as_text=as_text)
		train_for_rnn, scalers = feats2rnn(train_feats)

		dev_feats, _, _, _, _ = read_conll(devfile,genre_pat=self.genre_pat,as_text=as_text)
		dev_for_rnn, scalers = feats2rnn(dev_feats)

		with io.open(self.corpus_dir + os.sep + "dev."+self.ext,'w',encoding="utf8",newline="\n") as f:
			f.write(dev_for_rnn)

		# NCRFpp expects a test file, we reuse dev
		with io.open(self.corpus_dir + os.sep + "test."+self.ext,'w',encoding="utf8",newline="\n") as f:
			f.write(dev_for_rnn)

		if multifolds > 1:
			train_chunks, test_chunks = self.split_dataset(train_for_rnn,multifolds=multifolds)
			all_preds = []
			all_labs = []
			for i in range(opts.multifolds):
				with io.open(self.corpus_dir + os.sep + "train."+self.ext,'w',encoding="utf8",newline="\n") as f:
					f.write(train_chunks[i])

				with io.open(self.corpus_dir + os.sep + "raw."+self.ext,'w',encoding="utf8",newline="\n") as f:
					f.write(test_chunks[i])

				# TRAIN ON FOLD
				sys.stderr.write("\no Training on fold " + str(i+1)+"/" + str(multifolds) + "\n")
				config = ncrf_dir + os.sep + self.segtype + ".train"+self.auto+".config"
				ncrf(config)

				# PREDICT
				config = ncrf_dir + os.sep + self.segtype + ".decode"+self.auto+".config"
				ncrf(config,status="decode")

				labs, scores = self.read_preds()
				all_labs += labs
				all_preds += scores

			# SERIALIZE MULTITRAIN PREDS
			with io.open(self.corpus_dir + os.sep + corpus + self.auto + "_multitrain.tab",'w',newline='\n') as f:
				for j, pred in enumerate(all_preds):
					if self.conn:
						lab = all_labs[j]
					else:
						lab = 1 if pred >= 0.15 else 0  # Arbitrary threshold, we will only use prob as feature for metalearner
					f.write(str(lab) + "\t" + str(pred) + "\n")

		# TRAIN MODEL ON ALL TRAIN
		sys.stderr.write("\no Training on full train set\n")
		with io.open(self.corpus_dir + os.sep + "train."+self.ext,'w',encoding="utf8",newline="\n") as f:
			f.write(train_for_rnn)

		config = ncrf_dir + os.sep + self.segtype + ".train"+self.auto+".config"
		ncrf(config)

		p.end()

	def tune(self, trainfile, devfile, max_evals=10, as_text=False):

		train_feats, _, _, _, _ = read_conll(trainfile,genre_pat=self.genre_pat,as_text=as_text)
		train_for_rnn, scalers = feats2rnn(train_feats)

		dev_feats, _, _, _, _ = read_conll(devfile,genre_pat=self.genre_pat,as_text=as_text)
		dev_for_rnn, scalers = feats2rnn(dev_feats)

		with io.open(self.corpus_dir + os.sep + "dev."+self.ext,'w',encoding="utf8",newline="\n") as f:
			f.write(dev_for_rnn)

		# NCRFpp expects a test file, we reuse dev
		with io.open(self.corpus_dir + os.sep + "test."+self.ext,'w',encoding="utf8",newline="\n") as f:
			f.write(dev_for_rnn)

		# TRAIN MODEL ON ALL TRAIN
		with io.open(self.corpus_dir + os.sep + "train."+self.ext,'w',encoding="utf8",newline="\n") as f:
			f.write(train_for_rnn)

		config = ncrf_dir + os.sep + self.segtype + ".train"+self.auto+".config"

		def objective(params):

			# sys.stderr.write(str(params))

			data = Data()
			data.read_config(config)

			data.HP_batch_size = int(params['batch_size'])
			data.HP_lr = float(params['lr'])
			# data.word_emb_dim=int(params['word_emb_dim'])
			data.char_emb_dim=int(params['char_emb_dim'])
			data.word_feature_extractor=params['word_seq_feature']
			data.char_feature_extractor=params['char_seq_feature']
			#data.optimizer=params['optimizer']
			data.HP_cnn_layer=int(params['cnn_layer'])
			#data.HP_char_hidden_dim=int(params['char_hidden_dim'])
			data.HP_hidden_dim=int(params['hidden_dim'])
			data.HP_dropout=float(params['dropout'])
			data.HP_lstm_layer=int(params['lstm_layer'])
			data.average_batch_loss=str2bool(params['ave_batch_loss'])

			p = StdOutFilter4Tune()
			p.start()

			ret, best_dev = ncrf(config=None, data=data)

			p.end()
			sys.stdout.write("F1 {:.3f} params {}".format(-best_dev, params))


			if ret == 1:
				return {'loss': -best_dev, 'status': STATUS_OK }
			else:
				return {'status': STATUS_FAIL }

		space = {
			'batch_size': scope.int(hp.quniform('batch_size', 10, 100, 10)),
			'lr': hp.quniform('lr', 0.003, 0.18, 0.001),
			# 'word_emb_dim': scope.int(hp.quniform('word_emb_dim', 100, 300, 10)),
			'char_emb_dim': scope.int(hp.quniform('char_emb_dim', 30, 70, 10)),
			'word_seq_feature': hp.choice('word_seq_feature', ["LSTM","CNN"]),
			'char_seq_feature': hp.choice('char_seq_feature', ["LSTM","CNN"]),
			#'optimizer': hp.choice('optimizer', ["SGD","AdaGrad","AdaDelta","RMSProp","Adam"]),
			'optimizer': hp.choice('optimizer', ["AdaGrad"]),
			'cnn_layer': scope.int(hp.quniform('cnn_layer', 1, 8, 1)),
			'char_hidden_dim': scope.int(hp.quniform('char_hidden_dim', 50, 200, 10)),
			'hidden_dim': scope.int(hp.quniform('hidden_dim', 100, 300, 20)),
			'dropout': hp.quniform('dropout', 0.2, 0.8, 0.1),
			'lstm_layer': scope.int(hp.quniform('lstm_layer', 1, 5, 1)),
			'ave_batch_loss': hp.choice('ave_batch_loss', ["True","False"])
		}

		best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals)

		best_params = space_eval(space,best_params)

		with io.open(script_dir + os.sep + "params" + os.sep + "RNNSegmenter"+self.auto+"_best_params.tab",'a',encoding="utf8") as bp:
			corpus = os.path.basename(trainfile).split("_")[0]
			for k, v in best_params.items():
				bp.write("\t".join([corpus, 'RNNClassifier', k, str(v)])+"\n")
		return best_params



	def predict(self,testfile,as_text=True):

		test_feats, _, _, _, _ = read_conll(testfile,genre_pat=self.genre_pat,as_text=as_text)
		test_for_rnn, scalers = feats2rnn(test_feats)

		with io.open(self.corpus_dir + os.sep + "raw."+self.ext,'w',encoding="utf8",newline="\n") as f:
			f.write(test_for_rnn)

		p = StdOutFilter()
		p.start()

		config = ncrf_dir + os.sep + self.segtype+".decode"+self.auto+".config"
		ncrf(config,status="decode")

		p.end()

		labs, probas = self.read_preds()
		return zip(labs,probas)

	def predict_cached(self,train=None):
		output = []
		with io.open(self.corpus_dir + os.sep + self.model + self.auto + "_multitrain.tab",encoding="utf8") as f:
			for line in f.readlines():
				if '\t' in line:
					lab, pred = line.strip().split("\t")
					lab = float(lab)
					pred = float(pred)
					output.append((lab,pred))
		return output

	@staticmethod
	def rm_label(data):
		lines = data.split("\n")
		output = []
		for line in lines:
			line = line.strip()
			if " " in line:
				output.append(" ".join(line.split()[:-1]))
			else:
				output.append(line)
		return "\n".join(output)

	@staticmethod
	def split_dataset(data,multifolds=3):
		def chunk_list(data, multifolds=3):
			k, m = divmod(len(data), multifolds)
			return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(multifolds)]

		sequences = data.strip().split("\n\n")
		all_trainsets = []
		all_testsets = []

		chunks = chunk_list(sequences,multifolds)
		chunks = ["\n\n".join(seq_list) for seq_list in chunks]

		for i in range(multifolds):
			train = [chunks[j] for j in range(multifolds) if j!=i]
			heldout = chunks[i]
			#heldout = rm_label(heldout)
			train = "\n\n".join(train)
			all_trainsets.append(train)
			all_testsets.append(heldout)

		return all_trainsets, all_testsets

	def read_preds(self,nbest=5):
		lines = io.open(self.corpus_dir + os.sep + "raw.out",encoding="utf8").readlines()

		outprobs = []
		outlabs = []
		for line in lines:
			if line.startswith("#"):
				if re.match(r'# [01]',line) is not None:
					probas = line.split()[1:]
					probas = [float(p) for p in probas]
					continue
			if " " in line:
				fields = line.strip().split()
				labels = fields[-nbest:]
				score = 0.0
				for i, lab in enumerate(labels):  # Scale b-best label probabilities with harmonic series as a total score
					if i == 0:
						if self.conn:
							if "B" in lab:
								outlabs.append("B-Conn")
							elif "I" in lab:
								outlabs.append("I-Conn")
							else:
								outlabs.append("O")
						elif lab == "O":
							outlabs.append(0)
						else:
							outlabs.append(1)
					if lab != "O":
						score += probas[i]/(i+1)
				outprobs.append(score)
		return outlabs,outprobs


class StdOutFilter(object):
	def __init__(self): #, strings_to_filter, stream):
		self.stream = sys.stdout
		self.stdout = sys.stdout

	def __getattr__(self, attr_name):
		return getattr(self.stream, attr_name)

	def write(self, data):
		output = []
		lines = data.split("\n")
		for line in lines:
			if "Epoch: " in line or (" f:" in line and "Test" not in line):
				output.append(line)
		if len(output)>0:
			data = "\n".join(output) + "\n"
			self.stream.write("RNN log - " + data.strip() + "\n")
			self.stream.flush()

	def flush(self):
		self.stream.flush()

	def start(self):
		sys.stdout = self

	def end(self):
		sys.stdout = self.stdout


class StdOutFilter4Tune(object):
	def __init__(self): #, strings_to_filter, stream):
		self.stream = sys.stdout
		self.stdout = sys.stdout

	def __getattr__(self, attr_name):
		return getattr(self.stream, attr_name)

	def write(self, data):
		self.stream.flush()

	def flush(self):
		self.stream.flush()

	def start(self):
		sys.stdout = self

	def end(self):
		sys.stdout = self.stdout


def str2bool(string):
	if string == "True" or string == "true" or string == "TRUE":
		return True
	else:
		return False


if __name__ == "__main__":
	start_time = time.time()
	corpus_start_time = start_time
	table_out = []

	p = ArgumentParser()
	p.add_argument("-c","--corpus",default="spa.rst.sctb",help="Corpus to train on")
	p.add_argument("-d","--data_dir",default=os.path.normpath("../../../data"),help="Path to shared task data folder")
	p.add_argument("-m","--multifolds",default=5,type=int,help="Folds for multitraining")
	p.add_argument("--mode",action="store",default="train-test",choices=["train-test","test","tune"])
	p.add_argument("-b","--best_params",action="store_true",help="Load best parameters from file")
	p.add_argument("--eval_test",action="store_true",help="Evaluate on test instead of dev")
	p.add_argument("--auto",action="store_true",help="Evaluate on automatic parse")
	p.add_argument("--conn",action="store_true",help="Evaluate connective detection")

	opts = p.parse_args()

	corpora = opts.corpus

	data_dir = opts.data_dir

	if corpora == "all":
		corpora = os.listdir(data_dir)
		if opts.conn:
			corpora = [c for c in corpora if (os.path.isdir(os.path.join(data_dir, c)) and "pdtb" in c)]  # PDTB
		else:
			corpora = [c for c in corpora if (os.path.isdir(os.path.join(data_dir, c)) and "pdtb" not in c)]  # No PDTB
	else:
		corpora = [corpora]

	auto = "_auto" if opts.auto else ""
	if opts.auto:
		data_dir += "_parsed"

	for corpus in corpora:

		corpus_start_time = time.time()

		train = glob(data_dir + os.sep + corpus + os.sep + corpus + "_train.conll")[0]
		dev = glob(data_dir + os.sep + corpus + os.sep + corpus + "_dev.conll")[0]
		test = glob(data_dir + os.sep + corpus + os.sep + corpus + "_test.conll")[0]

		seg = RNNSegmenter(model=corpus,load_params=opts.best_params,auto=auto,conn=opts.conn)

		if "train" in opts.mode:
			sys.stderr.write("\no Training on corpus " + corpus + "\n")
			seg.train(train,dev,multifolds=opts.multifolds)
		elif "tune" in opts.mode:
			# TUNE
			sys.stderr.write("\no Tuning on corpus " + corpus + "\n")
			best_params = seg.tune(train,dev)
			sys.stderr.write(str(best_params))
		else:
			# PREDICT
			if opts.eval_test:
				seg.predict(test,as_text=False)
			else:
				seg.predict(dev,as_text=False)

		elapsed = time.time() - corpus_start_time
		sys.stderr.write("\nTime training on corpus:\n")
		sys.stderr.write(str(timedelta(seconds=elapsed)) + "\n\n")

	sys.stderr.write("\nTotal time:\n")
	elapsed = time.time() - start_time
	sys.stderr.write(str(timedelta(seconds=elapsed)) + "\n\n")
