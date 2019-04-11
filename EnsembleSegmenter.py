#!/usr/bin/python
# -*- coding: utf-8 -*-

import io, os, sys, re, copy
from collections import Counter
import numpy as np
import pandas as pd
from glob import glob
from argparse import ArgumentParser

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from lib.segmenters.subtree_segmenter import SubtreeSegmenter
from lib.segmenters.bow_seg_counter import BOWSegCounter
from lib.segmenters.rnn_segmenter import RNNSegmenter
#, ... some other estimators
from lib.conll_reader import read_conll
from lib.tune import get_best_params, hyper_optimize, get_best_score

from lib.exec import exec_via_temp
from collections import defaultdict
from random import seed

seed(42)
np.random.seed(42)

script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "lib")
segmenters_dir = lib + os.sep + "segmenters"
models = lib + os.sep + ".." + os.sep + "models"

DEFAULTCLF = RandomForestClassifier(random_state=42)

class EnsembleSegmenter:

	def __init__(self,lang="eng",model="eng.rst.gum",multitrain=False,auto=""):
		self.name = "EnsembleSegmenter"
		self.corpus = model
		self.lang = lang
		self.auto = auto
		lang_map = {"deu":"german","eng":"english","spa":"spanish","fra":"french","nld":"dutch","rus":"russian","eus":"basque","por":"portuguese","zho":"chinese","tur":"turkish"}
		self.long_lang = lang_map[lang] if lang in lang_map else lang
		self.estimators = []
		# Special genre patterns
		if "gum" in model:
			self.genre_pat = "GUM_(.+)_.*"
		else:
			self.genre_pat = "^(..)"
		try:
			self.udpipe_model = glob(os.path.abspath(os.path.join(lib,"udpipe",self.long_lang+"*.udpipe")))[0]
		except:
			sys.stderr.write("! Model not found for language " + self.long_lang + "*.udpipe in " + os.path.abspath(os.path.join([lib,"udpipe",self.long_lang+"*.udpipe"]))+"\n")
			sys.exit(0)
		self.udpipe_path = os.path.abspath(os.path.join(lib,"udpipe")) + os.sep

		self.estimators.append(SubtreeSegmenter(lang=lang,model=model,multifolds=1,auto=self.auto))
		self.estimators.append(BOWSegCounter(lang=lang,model=model,auto=self.auto))
		self.estimators.append(RNNSegmenter(model=model,load_params=True,genre_pat=self.genre_pat,auto=self.auto))
		#self.estimators.append(LRSegmenter(lang=lang,model=model))
		self.model = models + os.sep +model + self.auto + "_ensemble_seg.pkl"
		self.multitrain = multitrain

	def optimize(self, train, rare_thresh=100, size=5000, tune_mode="paramwise",as_text=False, cached_params=False):

		# Estimate useful features on a random sample of |size| instances
		sys.stderr.write("o Tuning hyperparameters\n\n")

		# Optimize hyperparameters via grid search
		if cached_params:
			clf, best_params, _ = get_best_params(self.corpus, self.name)
			sys.stderr.write("\no Using cached best hyperparameters\n")
		else:
			clf, best_params = self.train(train,rare_thresh=rare_thresh,tune_mode=tune_mode,size=size,as_text=as_text)
			sys.stderr.write("\no Found best hyperparameters\n")
		for key, val in best_params.items():
			sys.stderr.write(key + "\t" + str(val) + "\n")
		sys.stderr.write("\n")

		return clf, [], best_params

	def read_data(self,training_file, as_text=True, rare_thresh=100, no_cache=False):
		if as_text:
			train = training_file
		else:
			train = io.open(training_file,encoding="utf8").read().strip().replace("\r","") + "\n"

		cat_labels = ["word","genre","deprel","s_type","morph"]#,"depchunk"]#,"first","last"]#,"pos"]#,"first","last"]
		num_labels = ["tok_len","tok_id","quote","bracket","sent_doc_percentile","s_len"]

		train_feats, vocab, toks, firsts, lasts = read_conll(train,genre_pat=self.genre_pat,mode="seg",as_text=True,char_bytes=self.lang=="zho")
		gold_feats, _, _, _, _ = read_conll(train,mode="seg",as_text=True)
		gold_feats = [{"wid":0,"label":"_"}] + gold_feats + [{"wid":0,"label":"_"}]  # Add dummies to gold

		# Ensure that "_" is in the possible values of first/last for OOV chars at test time
		oov_item = train_feats[-1]
		oov_item["first"] = "_"
		oov_item["last"] = "_"
		oov_item["lemma"] = "_"
		oov_item["word"] = "_"
		oov_item["deprel"] = "_"
		oov_item["pos"] = "_"
		oov_item["cpos"] = "_"
		oov_item["genre"] = "_"
		oov_item["depchunk"] = "_"
		train_feats.append(oov_item)
		train_feats = [oov_item] + train_feats
		toks.append("_")
		toks = ["_"] + toks

		vocab = Counter(vocab)
		top_n_words = vocab.most_common(rare_thresh)
		top_n_words, _ = zip(*top_n_words)

		headers = sorted(list(train_feats[0].keys()))
		data = []

		preds = {}

		for e in self.estimators:
			if self.multitrain and not no_cache:
				pred = e.predict_cached(train)
			else:
				pred = e.predict(train)
			_, preds[e.name + "_prob"] = [list(x) for x in zip(*pred)]
			preds[e.name + "_prob"] = [0.0] + preds[e.name + "_prob"] + [0.0]  # Add dummy wrap for items -1 and +1
			headers.append(e.name + "_prob")
			num_labels.append(e.name + "_prob")

		for i, item in enumerate(train_feats):
			if item["word"] not in top_n_words:
				item["word"] = item["pos"]
			for e in self.estimators:
				item[e.name + "_prob"] = preds[e.name + "_prob"][i]

			feats = []
			for k in headers:
				feats.append(item[k])

			data.append(feats)

		data, headers, cat_labels, num_labels = self.n_gram(data, headers, cat_labels, num_labels)

		# No need for n_gram feats for the following:
		if "BOWSegCounter_prob_min1" in num_labels:
			num_labels.remove("BOWSegCounter_prob_min1")
			num_labels.remove("BOWSegCounter_prob_pls1")
		if "RNNSegmenter_prob_min1" in num_labels:
			num_labels.remove("RNNSegmenter_prob_min1")
			num_labels.remove("RNNSegmenter_prob_pls1")
		if "LRSegmenter_prob_min1" in num_labels:
			num_labels.remove("LRSegmenter_prob_min1")
			num_labels.remove("LRSegmenter_prob_pls1")
		if "SubtreeSegmenter_prob_min1" in num_labels:
			num_labels.remove("SubtreeSegmenter_prob_min1")
			num_labels.remove("SubtreeSegmenter_prob_pls1")
		if "tok_id_min1" in num_labels:
			num_labels.remove("tok_id_min1")
			num_labels.remove("tok_id_pls1")
		if "genre_min1" in cat_labels:
			cat_labels.remove("genre_min1")
			cat_labels.remove("genre_pls1")
		if "s_type_min1" in cat_labels:
			cat_labels.remove("s_type_min1")
			cat_labels.remove("s_type_pls1")
		if "morph_min1" in cat_labels:
			cat_labels.remove("morph_min1")
			cat_labels.remove("morph_pls1")
		if "s_len_min1" in num_labels:
			num_labels.remove("s_len_min1")
			num_labels.remove("s_len_pls1")
		if "sent_doc_percentile_min1" in num_labels:
			num_labels.remove("sent_doc_percentile_min1")
			num_labels.remove("sent_doc_percentile_pls1")

		data = pd.DataFrame(data, columns=headers)
		data_encoded, multicol_dict = self.multicol_fit_transform(data, pd.Index(cat_labels))

		data_x = data_encoded[cat_labels+num_labels].values
		data_y = [int(t['label'] != "_") for t in gold_feats]

		return data_encoded, data_x, data_y, cat_labels, num_labels, multicol_dict, firsts, lasts, top_n_words

	def train(self,training_file,rare_thresh=100,as_text=True,tune_mode=None,size=None,clf_params=None,chosen_clf=None):
		"""
		Train the EnsembleSentencer. Note that the underlying estimators are assumed to be pretrained already.

		:param training_file: File in DISRPT shared task .conll format
		:param rare_thresh: Rank of rarest word to include (rarer items are replace with POS)
		:param genre_pat: Regex pattern with capturing group to extract genre from document names
		:param as_text: Boolean, whether the input is a string, rather than a file name to read
		:return:
		"""

		if tune_mode is not None and size is None and tune_mode != "hyperopt":
			size = 5000
			sys.stderr.write("o No sample size set - setting size to 5000\n")
		if clf_params is None:
			# Default classifier parameters
			clf_params = {"n_estimators":100,"min_samples_leaf":3,"random_state":42}
		if chosen_clf is None:
			chosen_clf = DEFAULTCLF

		data_encoded, data_x, data_y, cat_labels, num_labels, multicol_dict, firsts, lasts, top_n_words = self.read_data(training_file,rare_thresh=rare_thresh,as_text=as_text)

		sys.stderr.write("o Learning...\n")

		if tune_mode == "hyperopt":
			from hyperopt import hp
			from hyperopt.pyll.base import scope
			dev_file = training_file.replace("_train","_dev")
			_, val_x, val_y, _, _, _, _, _, _ = self.read_data(dev_file,rare_thresh=rare_thresh,as_text=False,no_cache=True)
			space = {
				'n_estimators': scope.int(hp.quniform('n_estimators', 50, 150, 10)),
				'max_depth': scope.int(hp.quniform('max_depth', 5, 40, 1)),
				'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
				'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
				'max_features': hp.choice('max_features', ["sqrt", None, 0.5, 0.6, 0.7, 0.8]),
				'clf': hp.choice('clf', ["rf","et","gbm"])
			}
			space = {
				'n_estimators': scope.int(hp.quniform('n_estimators', 50, 150, 10)),
				'max_depth': scope.int(hp.quniform('max_depth', 3, 35, 1)),
				'eta': scope.float(hp.quniform('eta', 0.01, 0.2, 0.01)),
				'gamma': scope.float(hp.quniform('gamma', 0.01, 0.2, 0.01)),
				'colsample_bytree': hp.choice('colsample_bytree', [0.4,0.5,0.6,0.7,0.8,1.0]),
				'subsample': hp.choice('subsample', [0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
				'clf': hp.choice('clf', ["xgb"])
			}
			#best_clf, best_params = hyper_optimize(data_x,data_y,val_x=val_x,val_y=val_y,space=space,max_evals=75)
			best_clf, best_params = hyper_optimize(data_x,data_y,val_x=None,val_y=None,space=space,max_evals=75)
			return best_clf, best_params
		elif tune_mode is not None:
			# Randomly select |size| samples for training and leave rest for validation, max |size| samples
			data_x = data_encoded[cat_labels+num_labels+["label"]].sample(frac=1,random_state=42)
			data_y = np.where(data_x['label'] == "_", 0, 1)
			data_x = data_x[cat_labels+num_labels]
			if len(data_y) > 2*size:
				val_x = data_x[size:2*size]
				val_y = data_y[size:2*size]
			else:
				val_x = data_x[size:]
				val_y = data_y[size:]
			data_x = data_x[:size]
			data_y = data_y[:size]

			best_params = {}
			best_params_by_clf = defaultdict(dict)
			# Tune individual params separately for speed, or do complete grid search if building final model
			params_list = [{"n_estimators":[75,100,125]},
						   {'max_depth': [7,10,15,None]},
						   {"min_samples_split": [2, 5, 10]},
						   {"min_samples_leaf":[1,2,3]}]
			if tune_mode == "full":
				# Flatten dictionary if doing full CV
				params_list = [{k: v for d in params_list for k, v in d.items()}]
			best_score = -10000
			for clf in [RandomForestClassifier(),ExtraTreesClassifier(),GradientBoostingClassifier()]:
				for params in params_list:
					base_params = copy.deepcopy(clf_params)  # Copy default params
					if clf.__class__.__name__ != "GradientBoostingClassifier":
						base_params.update({"n_jobs":4, "oob_score":True, "bootstrap":True})
					for p in params:
						if p in base_params:  # Ensure base_params don't conflict with grid search params
							base_params.pop(p)
					clf.set_params(**base_params)
					grid = GridSearchCV(clf,params,cv=3,n_jobs=3,error_score="raise",refit=False,scoring="f1")
					grid.fit(data_x,data_y)
					if tune_mode == "full":
						if grid.best_score_ > best_score:
							best_score = grid.best_score_
							best_clf = clf
							for param in params:
								best_params[param] = grid.best_params_[param]
					else:
						if grid.best_score_ > best_score:
							best_clf = clf
						for param in params:
							best_params_by_clf[clf.__class__.__name__][param] = grid.best_params_[param]
			if tune_mode == "paramwise":
				best_params = best_params_by_clf[best_clf.__class__.__name__]
			else:
				best_params["best_score"] = best_score

			clf_name = best_clf.__class__.__name__
			with io.open(segmenters_dir + os.sep + "params" + os.sep + "EnsembleSegmenter"+self.auto+"_best_params.tab",'a',encoding="utf8") as bp:
				corpus = os.path.basename(training_file).split("_")[0]
				for k, v in best_params.items():
					bp.write("\t".join([corpus, clf_name, k, str(v)])+"\n")
			self.clf = best_clf
			return best_clf, best_params
		else:
			clf = chosen_clf
			clf.set_params(**clf_params)
			if clf.__class__.__name__ != "GradientBoostingClassifier":
				clf.set_params(**{"n_jobs":3,"oob_score":True,"bootstrap":True})
			clf.set_params(**{"random_state":42})
			clf.fit(data_x,data_y)
			self.clf = clf

		feature_names = cat_labels + num_labels

		zipped = zip(feature_names, clf.feature_importances_)
		sorted_zip = sorted(zipped, key=lambda x: x[1], reverse=True)
		sys.stderr.write("o Feature importances:\n\n")
		for name, importance in sorted_zip:
			sys.stderr.write(name + "=" + str(importance) + "\n")

		#sys.stderr.write("\no OOB score: " + str(clf.oob_score_)+"\n")

		sys.stderr.write("\no Serializing model...\n")

		joblib.dump((clf, num_labels, cat_labels, multicol_dict, top_n_words, firsts, lasts), self.model, compress=3)

	def predict(self, conllu, eval_gold=False, as_text=True, serialize=False):
		"""
		Predict sentence splits using an existing model

		:param infile: File in DISRPT shared task *.conll format
		:param eval_gold: Whether to score the prediction; only applicable if using a gold .conll file as input
		:param as_text: Boolean, whether the input is a string, rather than a file name to read
		:return: tokenwise binary prediction vector if eval_gold is False, otherwise prints evaluation metrics and diff to gold
		"""

		clf, num_labels, cat_labels, multicol_dict, vocab, firsts, lasts = joblib.load(self.model)
		self.clf = clf

		if not as_text:
			conllu = io.open(conllu,encoding="utf8").read()

		train_feats, _, toks, _, _ = read_conll(conllu,genre_pat=self.genre_pat,mode="seg",as_text=True)
		headers = sorted(list(train_feats[0].keys()))

		data = []

		preds = {}
		for e in self.estimators:
			pred = e.predict(conllu)
			_, preds[e.name + "_prob"] = [list(x) for x in zip(*pred)]
			headers.append(e.name + "_prob")

		temp = []
		headers_with_oov = ["deprel","pos","cpos","morph","s_type","depchunk"]
		for pref in ["min1","pls1"]:
			temp += [pref + "_" + h for h in headers_with_oov]
		headers_with_oov += temp

		genre_warning = False
		for i, header in enumerate(headers):
			if header in headers_with_oov and header in cat_labels:
				for item in train_feats:
					if item[header] not in multicol_dict["encoder_dict"][header].classes_:
						item[header] = "_"
		for i, item in enumerate(train_feats):
			item["first"] = item["word"][0] if item["word"][0] in firsts else "_"
			item["last"] = item["word"][-1] if item["word"][-1] in lasts else "_"
			if "genre" in cat_labels:
				if item["genre"] not in multicol_dict["encoder_dict"]["genre"].classes_:  # New genre not in training data
					if not genre_warning:
						sys.stderr.write("! WARN: Genre not in training data: " + item["genre"] + "; suppressing further warnings\n")
						genre_warning = True
					item["genre"] = "_"
			if item["word"] not in vocab:
				if item["pos"] in multicol_dict["encoder_dict"]["word"].classes_:
					item["word"] = item["pos"]
				else:
					item["word"] = "_"
			for e in self.estimators:
				item[e.name + "_prob"] = preds[e.name + "_prob"][i]

			feats = []
			for k in headers:
				feats.append(item[k])

			data.append(feats)

		data, headers, _, _ = self.n_gram(data,headers,[],[])

		data = pd.DataFrame(data, columns=headers)
		data_encoded = self.multicol_transform(data,columns=multicol_dict["columns"],all_encoders_=multicol_dict["all_encoders_"])

		data_x = data_encoded[cat_labels+num_labels].values
		pred = clf.predict(data_x)

		if serialize:
			self.serialize(conllu,pred)
		if eval_gold:
			gold_feats, _,_,_,_ = read_conll(conllu,genre_pat=self.genre_pat,mode="seg",as_text=True)
			gold = [int(t['label'] != "_") for t in gold_feats]
			conf_mat = confusion_matrix(gold, pred)
			sys.stderr.write(str(conf_mat) + "\n")
			true_positive = conf_mat[1][1]
			false_positive = conf_mat[0][1]
			false_negative = conf_mat[1][0]
			prec = true_positive / (true_positive + false_positive)
			rec = true_positive / (true_positive + false_negative)
			f1 = 2*prec*rec/(prec+rec)
			sys.stderr.write("P: " + str(prec) + "\n")
			sys.stderr.write("R: " + str(rec) + "\n")
			sys.stderr.write("F1: " + str(f1) + "\n")
			with io.open("diff.tab",'w',encoding="utf8") as f:
				for i in range(len(gold)):
					f.write("\t".join([toks[i],str(gold[i]),str(pred[i])])+"\n")
			return conf_mat, prec, rec, f1
		else:
			return pred

	@staticmethod
	def multicol_fit_transform(dframe, columns):
		"""
		Transforms a pandas dataframe's categorical columns into pseudo-ordinal numerical columns and saves the mapping

		:param dframe: pandas dataframe
		:param columns: list of column names with categorical values to be pseudo-ordinalized
		:return: the transformed dataframe and the saved mappings as a dictionary of encoders and labels
		"""

		if isinstance(columns, list):
			columns = np.array(columns)
		else:
			columns = columns

		encoder_dict = {}
		# columns are provided, iterate through and get `classes_` ndarray to hold LabelEncoder().classes_
		# for each column; should match the shape of specified `columns`
		all_classes_ = np.ndarray(shape=columns.shape, dtype=object)
		all_encoders_ = np.ndarray(shape=columns.shape, dtype=object)
		all_labels_ = np.ndarray(shape=columns.shape, dtype=object)
		for idx, column in enumerate(columns):
			# instantiate LabelEncoder
			le = LabelEncoder()
			# fit and transform labels in the column
			dframe.loc[:, column] = le.fit_transform(dframe.loc[:, column].values)
			encoder_dict[column] = le
			# append the `classes_` to our ndarray container
			all_classes_[idx] = (column, np.array(le.classes_.tolist(), dtype=object))
			all_encoders_[idx] = le
			all_labels_[idx] = le

		multicol_dict = {"encoder_dict":encoder_dict, "all_classes_":all_classes_,"all_encoders_":all_encoders_,"columns": columns}
		return dframe, multicol_dict

	def serialize(self,conllu_in,predictions):
		output = []
		i = 0
		for line in conllu_in.split("\n"):
			if "\t" in line:
				fields = line.split("\t")
				if "-" in fields[0]:
					output.append(line)
				else:
					if predictions[i] == 1:
						fields[-1] = "BeginSeg=Yes"
					else:
						fields[-1] = "_"
					i+=1
					output.append("\t".join(fields))
			else:
				output.append(line)
		with io.open(self.corpus + "_pred.out.conll",'w',encoding="utf8") as outfile:
			outfile.write("\n".join(output) + "\n")

	@staticmethod
	def multicol_transform(dframe, columns, all_encoders_):
		"""
		Transforms a pandas dataframe's categorical columns into pseudo-ordinal numerical columns based on existing mapping
		:param dframe: a pandas dataframe
		:param columns: list of column names to be transformed
		:param all_encoders_: same length list of sklearn encoders, each mapping categorical feature values to numbers
		:return: transformed numerical dataframe
		"""
		for idx, column in enumerate(columns):
			dframe.loc[:, column] = all_encoders_[idx].transform(dframe.loc[:, column].values)
		return dframe


	@staticmethod
	def n_gram(data, headers, cat_labels, num_labels):
		"""
		Turns unigram feature list into list of tri-skipgram features by adding features of adjacent tokens

		:param data: List of observations, each an ordered list of feature values
		:param headers: List of all feature names in the data
		:param cat_labels: List of categorical features to be used in model
		:param num_labels: List of numerical features to be used in the model
		:return: Modified data, headers and label lists including adjacent token properties
		"""
		n_grammed = []

		for i, tok in enumerate(data):
			if i == 0:
				n_grammed.append(data[-1]+tok+data[i+1])
			elif i == len(data) - 1:
				n_grammed.append(data[i-1]+tok+data[0])
			else:
				n_grammed.append(data[i-1]+tok+data[i+1])

		n_grammed_headers = [header + "_min1" for header in headers] + headers + [header + "_pls1" for header in headers]
		n_grammed_cat_labels = [lab + "_min1" for lab in cat_labels] + cat_labels + [lab + "_pls1" for lab in cat_labels]
		n_grammed_num_labels = [lab + "_min1" for lab in num_labels] + num_labels + [lab + "_pls1" for lab in num_labels]

		return n_grammed, n_grammed_headers, n_grammed_cat_labels, n_grammed_num_labels


if __name__ == "__main__":

	p = ArgumentParser()
	p.add_argument("-c","--corpus",default="spa.rst.sctb",help="Corpus to use or 'all'")
	p.add_argument("-d","--data_dir",default="../data",help="Path to shared task data folder")
	p.add_argument("-m","--multitrain",action="store_true",help="Load saved predictions from ensemble multitraining",default=True)
	p.add_argument("-t","--tune_mode",default="paramwise",choices=["paramwise","full","hyperopt"])
	p.add_argument("-b","--best_params",action="store_true",help="Load best parameters from file")
	p.add_argument("-o","--outfile",action="store_true",help="Print output file CORPUS.pred.out.conll")
	p.add_argument("--mode",choices=["train","test","train-test","optimize-train-test","optimize-train"],default="train-test")
	p.add_argument("--eval_test",action="store_true",help="Evaluate on test, not dev")
	p.add_argument("--auto",action="store_true",help="Evaluate on automatic parse")
	opts = p.parse_args()

	if not opts.multitrain and "train" in opts.mode:
		sys.stderr.write("! WARN: multitraining is off\n")

	corpus = opts.corpus
	data_dir = opts.data_dir

	if opts.auto:
		data_dir = data_dir + "_parsed"
		sys.stderr.write("o Evaluating on automatically parsed data\n")

	corpora = os.listdir(data_dir)
	if corpus == "all":
		corpora = [c for c in corpora if os.path.isdir(os.path.join(data_dir, c))]
	else:
		corpora = [c for c in corpora if os.path.isdir(os.path.join(data_dir, c)) and c == corpus]

	auto = "_auto" if opts.auto else ""

	for corpus in corpora:
		if "pdtb" in corpus or "stan" in corpus:  # Not scoring PDTB connective detection with segmenter module
			continue

		# Set corpus and file information
		train = os.path.join(data_dir,corpus, corpus + "_train.conll")
		dev = os.path.join(data_dir, corpus, corpus + "_dev.conll")
		test = os.path.join(data_dir, corpus, corpus + "_test.conll")
		model_path = "models" + os.sep + corpus + "_sent.pkl"

		if "." in corpus:
			lang = corpus.split(".")[0]
		else:
			lang = "eng"

		# Predict sentence splits
		e = EnsembleSegmenter(lang=lang,model=corpus,multitrain=opts.multitrain,auto=auto)

		best_params = None
		if "optimize" in opts.mode:
			clf, vars, best_params = e.optimize(train,size=5000,tune_mode=opts.tune_mode,cached_params=opts.best_params)
			if "best_score" in best_params:
				best_params.pop("best_score")
			# Now train on whole training set with those variables
			sys.stderr.write("\no Training best configuration\n")
			e.train(train,rare_thresh=100,clf_params=best_params,as_text=False,chosen_clf=clf)
		elif "train" in opts.mode:
			if opts.best_params:
				best_clf, best_params, _ = get_best_params(corpus,e.name,auto=auto)
			else:
				best_clf=None
			sys.stderr.write("\no Training on corpus "+corpus+"\n")
			tune_mode=None if opts.tune_mode != "hyperopt" else "hyperopt"
			e.train(train,as_text=False,tune_mode=tune_mode,chosen_clf=best_clf,clf_params=best_params)

		if "test" in opts.mode:
			if opts.eval_test:
				conf_mat, prec, rec, f1 = e.predict(test,eval_gold=True,as_text=False,serialize=opts.outfile)
			else:
				conf_mat, prec, rec, f1 = e.predict(dev,eval_gold=True,as_text=False,serialize=opts.outfile)

			# Get prediction performance on dev
			if best_params is not None and "optimize" in opts.mode:  # For optimization check if this is a new best score
				best_clf = e.clf
				prev_best_score = get_best_score(corpus,"EnsembleSegmenter"+e.auto)
				if f1 > prev_best_score:
					sys.stderr.write("o New best F1: " + str(f1) + "\n")
					with io.open(segmenters_dir + os.sep + "params" + os.sep + "EnsembleSegmenter"+e.auto+"_best_params.tab",'a',encoding="utf8") as bp:
						for k, v in best_params.items():
							bp.write("\t".join([corpus, best_clf.__class__.__name__, k, str(v)])+"\n")
						#bp.write("\t".join([corpus, best_clf.__class__.__name__, "features", ",".join(vars)])+"\n")
						bp.write("\t".join([corpus, best_clf.__class__.__name__, "best_score", str(f1)])+"\n\n")

