from sklearn.linear_model import LogisticRegressionCV, RidgeCV, LassoCV, ElasticNetCV
import numpy as np
import sys, os, io
from argparse import ArgumentParser
from sklearn.externals import joblib

# Allow package level imports in module
script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "..")
sys.path.append(lib)

from bow_reader import read_bow
import warnings

class BOWSegCounter:

	def __init__(self,lang="eng",model="eng.rst.gum",auto=""):
		self.name = "BOWSegCounter"
		self.clf = None
		self.vocab = None
		self.lang = lang
		self.model = model
		self.auto = auto

	def train(self,training_file,model_path=None,as_text=True,multifolds=5):

		if model_path is None:  # Try default model location
			model_path = script_dir + os.sep + ".." + os.sep + ".." + os.sep + "models" + os.sep + self.model + self.auto + "_bowcount_seg.pkl"

		rows, vocab = read_bow(training_file,as_text=as_text)
		s_lens = []

		X = []
		y = []
		for row in rows:
			label = row[-1]
			s_len = row[-2]
			feats = row[:-2]
			s_lens.append(int(s_len))
			y.append(label)
			X.append(feats)

		X = np.array(X).astype(np.float64)
		y = np.array(y).astype(np.int)

		if multifolds < 2:
			#clf = LogisticRegressionCV(cv=3,random_state=42,solver="liblinear",penalty="l1")
			clf = RidgeCV(cv=3)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				clf.fit(X,y)
		else:
			#clf = LogisticRegressionCV(cv=3,random_state=42,solver="liblinear",penalty="l1")
			clf = RidgeCV(cv=3)
			all_preds, all_probas = [], []
			X_folds = np.array_split(X, multifolds)
			y_folds = np.array_split(y, multifolds)
			s_folds = np.array(s_lens)
			s_folds = np.array_split(s_folds, multifolds)
			for i in range(multifolds):
				sys.stderr.write("o Multitrain fold " + str(i+1)+"/"+str(multifolds) + "\n")
				X_train = np.vstack(tuple([X_folds[j] for j in range(multifolds) if j!=i]))
				y_train = np.concatenate(tuple([y_folds[j] for j in range(multifolds) if j!=i]))
				X_heldout = X_folds[i]
				s_heldout = list(s_folds[i])
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					clf.fit(X_train,y_train)
				#probas = clf.predict_proba(X_heldout)
				probas = clf.predict(X_heldout)
				preds, probas = self.duplicate_sent_preds(probas,s_heldout)
				all_preds += preds
				all_probas += probas

			with io.open(script_dir + os.sep + "multitrain" + os.sep + self.name + self.auto + "_" + self.model,'w',encoding="utf8") as f:
				for pred, proba in zip(all_preds,all_probas):
					f.write(str(pred) + "\t" + str(proba) + "\n")

			# Now fit classifier on whole data for predict time
			#clf = LogisticRegressionCV(cv=3,random_state=42,solver="liblinear",penalty="l1")
			clf = RidgeCV(cv=3)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				sys.stderr.write("o Training on full train set\n\n")
				clf.fit(X,y)

		self.clf = clf
		self.vocab = vocab

		joblib.dump((clf, vocab), model_path, compress=3)

	def predict_cached(self,filename,model_path=None,as_text=True,eval_gold=False):
		return self.predict(filename,model_path=model_path,as_text=as_text,eval_gold=eval_gold,cache=True)

	def predict(self,filename,model_path=None,as_text=True,eval_gold=False,cache=False):

		if model_path is None:  # Try default model location
			model_path = script_dir + os.sep + ".." + os.sep + ".." + os.sep + "models" + os.sep + self.model + self.auto + "_bowcount_seg.pkl"

		self.clf, self.vocab = joblib.load(model_path)

		rows = read_bow(filename,test_vocab=self.vocab,as_text=as_text)

		X = []
		y = []
		s_lens = []
		for row in rows:
			s_len = row[-2]
			label = row[-1]
			feats = row[:-2]
			y.append(label)
			s_lens.append(int(s_len))
			X.append(feats)

		X = np.array(X).astype(np.float64)

		if cache:
			# Load predictions from a file
			cache_file = script_dir + os.sep + "multitrain" + os.sep + self.name + self.auto + "_" + self.model
			lines = io.open(cache_file).readlines()
			dup_preds, dup_probas = [],[]
			for line in lines:
				if "\t" in line:
					pred, proba = line.split("\t")
					dup_preds.append(int(pred))
					dup_probas.append(float(proba))
			#_, _, dup_y = self.duplicate_sent_preds(list(zip(s_lens,s_lens)),s_lens,y)
			_, _, dup_y = self.duplicate_sent_preds(s_lens,s_lens,y)
		else:
			#probas = self.clf.predict_proba(X)
			probas = self.clf.predict(X)
			dup_preds, dup_probas, dup_y = self.duplicate_sent_preds(probas,s_lens,y)

		if eval_gold:
			dup_y = np.array(dup_y).astype(np.int)
			predicted = np.array([round(p) for p in dup_probas])
			#predicted = np.array([int(p>0.5) for p in dup_probas])
			print("Accuracy: ")
			print(sum([predicted[i] == dup_y[i] for i in range(len(dup_y))])/len(dup_y))
			predicted = np.ones(len(dup_y))
			print("Negative baseline: ")
			print(sum([predicted[i] == dup_y[i] for i in range(len(dup_y))])/len(dup_y))
		else:
			return zip(dup_preds,dup_probas)

	@staticmethod
	def duplicate_sent_preds(probas, s_lens, y=None):
		dup_probas = []
		dup_preds = []
		dup_y = []
		for i, prob in enumerate(probas):
			p_positive = prob#[1]
			for j in range(s_lens[i]):
				dup_probas.append(p_positive)
				dup_preds.append(1) if p_positive > 0.5 else dup_preds.append(0)
				if y is not None:
					#dup_y.append(1) if y[i] == 1 else dup_y.append(0)
					dup_y.append(y[i])
		if y is not None:
			return dup_preds, dup_probas, dup_y
		else:
			return dup_preds, dup_probas


if __name__ == "__main__":

	p = ArgumentParser()
	p.add_argument("-c","--corpus",default="eng.rst.gum",help="Corpus to train on in data_dir or 'all'")
	p.add_argument("-d","--data_dir",default=os.path.normpath("../../../data"),help="Path to shared task data folder")
	p.add_argument("-m","--multitrain",type=int,default=5,help="Perform k-fold multitraining and save predictions for ensemble training")
	p.add_argument("--auto",action="store_true",help="Evaluate on automatic parse")
	opts = p.parse_args()

	corpora  = opts.corpus

	data_dir = opts.data_dir
	if opts.auto:
		data_dir += "_parsed"
		auto = "_auto"
	else:
		auto = ""

	if corpora == "all":
		corpora = os.listdir(data_dir )
		corpora = [c for c in corpora if (os.path.isdir(os.path.join(data_dir, c)) and "pdtb" not in c)]  # No PDTB
	else:
		corpora = [corpora]

	for corpus in corpora:

		if "." in corpus:
			lang = corpus.split(".")[0]
		else:
			lang = "eng"

		seg = BOWSegCounter(lang=lang,model=corpus,auto=auto)

		sys.stderr.write("\no Training on corpus " + corpus + "\n\n")
		seg.train(data_dir + os.sep + corpus + os.sep + corpus+ "_train.conll",as_text=False,multifolds=opts.multitrain)
		if opts.multitrain > 1:
			# Get accuracy on train via out-of-fold predictions
			seg.predict(data_dir + os.sep + corpus + os.sep + corpus+ "_train.conll",eval_gold=True,as_text=False,cache=opts.multitrain>1)
		else:
			# Get accuracy on separate dev data
			seg.predict(data_dir + os.sep + corpus + os.sep + corpus+ "_dev.conll",eval_gold=True,as_text=False)

