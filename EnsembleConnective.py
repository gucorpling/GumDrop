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

from lib.segmenters.freq_conn_detector import FreqConnDetector
from lib.segmenters.rnn_segmenter import RNNSegmenter
#, ... some other estimators
from lib.conll_reader import read_conll,read_conll_conn
from lib.tune import get_best_params, hyper_optimize, get_best_score
from lib.seg_eval import get_scores

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

class EnsembleConnective:

	def __init__(self,lang="zho",model="zho.pdtb.cdtb",multitrain=False,auto=""):
		self.name = "EnsembleConnective"
		self.genre_pat = "^(..)"
		self.corpus = model
		self.auto = auto
		self.lang = lang
		lang_map = {"deu":"german","eng":"english","spa":"spanish","fra":"french","nld":"dutch","rus":"russian","eus":"basque","por":"portuguese","zho":"chinese","tur":"turkish"}
		self.long_lang = lang_map[lang] if lang in lang_map else lang
		self.estimators = []
		self.estimators.append(FreqConnDetector(lang=lang,model=model))
		self.estimators.append(RNNSegmenter(model=model,conn=True,load_params=False))
		self.model = models + os.sep +model + "_ensemble_conn.pkl"
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

	def read_data(self,training_file,rare_thresh=100,as_text=True, no_cache=False):

		if as_text:
			train = training_file
		else:
			train = io.open(training_file,encoding="utf8").read().strip().replace("\r","") + "\n"

		cat_labels = ["word","genre","deprel","s_type","morph"]#,"depchunk"]#,"first","last"]#,"pos"]#,"first","last"]
		num_labels = ["tok_len","tok_id","quote","bracket","sent_doc_percentile","s_len"]

		train_feats, vocab, toks, firsts, lasts = read_conll(train,genre_pat=self.genre_pat,mode="seg",as_text=True,char_bytes=self.lang=="zho")
		gold_feats, _, _, _, _ = read_conll_conn(train,mode="seg",as_text=True)
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
			if self.multitrain and e.name in ["RNNSegmenter"] and not no_cache:
				pred = e.predict_cached(train)
			# _, preds[e.name + "_B_prob"], preds[e.e.name + "_I_prob"] = [list(x) for x in zip(*pred)]
			else:
				pred = e.predict(train)
			preds[e.name + "_B_prob"] = []
			preds[e.name + "_I_prob"] = []
			if "Freq" in e.name:
				preds[e.name + "_freq"] = []
			for tup in pred:
				if "RNN" in e.name:
					pred = tup[0]
					probas = tup[1]
					freqs = None
				else:
					pred = tup[1]
					probas = float(tup[2])
					freqs = float(tup[3])
				if "B-Con" in pred:
					preds[e.name + "_B_prob"].append(probas)
					preds[e.name + "_I_prob"].append(0.0)
				elif "I-Con" in pred:
					preds[e.name + "_B_prob"].append(0.0)
					preds[e.name + "_I_prob"].append(probas)
				else:
					preds[e.name + "_B_prob"].append(0.0)
					preds[e.name + "_I_prob"].append(0.0)
				if freqs is not None:
					preds[e.name + "_freq"].append(freqs)

				# _, preds[e.name + "_prob"], ratio, freq = [list(x) for x in zip(*pred)]
			preds[e.name + "_B_prob"] = [0.0] + preds[e.name + "_B_prob"] + [0.0]  # Add dummy wrap for items -1 and +1
			preds[e.name + "_I_prob"] = [0.0] + preds[e.name + "_I_prob"] + [0.0]  # Add dummy wrap for items -1 and +1
			if e.name == "FreqConnDetector":
				preds[e.name + "_freq"] = [0.0] + preds[e.name + "_freq"] + [0.0]  # Add dummy wrap for items -1 and +1
			headers.append(e.name + "_B_prob")
			headers.append(e.name + "_I_prob")
			num_labels.append(e.name + "_B_prob")
			num_labels.append(e.name + "_I_prob")
			if "Freq" in e.name:
				headers.append(e.name + "_freq")
				num_labels.append(e.name + "_freq")

		for i, item in enumerate(train_feats):
			if item["word"] not in top_n_words:
				item["word"] = item["pos"]
			for e in self.estimators:
				item[e.name + "_B_prob"] = preds[e.name + "_B_prob"][i]
				item[e.name + "_I_prob"] = preds[e.name + "_I_prob"][i]
				if e.name == "FreqConnDetector":
					item[e.name + "_freq"] = preds[e.name + "_freq"][i]

			feats = []
			for k in headers:
				feats.append(item[k])

			data.append(feats)

		data, headers, cat_labels, num_labels = self.n_gram(data, headers, cat_labels, num_labels)

		# No need for n_gram feats for the following:
		if "FreqConnDetector_B_prob_min1" in num_labels:
			num_labels.remove("FreqConnDetector_B_prob_min1")
			num_labels.remove("FreqConnDetector_B_prob_pls1")
		if "FreqConnDetector_I_prob_min1" in num_labels:
			num_labels.remove("FreqConnDetector_I_prob_min1")
			num_labels.remove("FreqConnDetector_I_prob_pls1")
		if "FreqConnDetector_freq_min1" in num_labels:
			num_labels.remove("FreqConnDetector_freq_min1")
			num_labels.remove("FreqConnDetector_freq_pls1")
		if "RNNSegmenter_B_prob_min1" in num_labels:
			num_labels.remove("RNNSegmenter_B_prob_min1")
			num_labels.remove("RNNSegmenter_B_prob_pls1")
		if "RNNSegmenter_I_prob_min1" in num_labels:
			num_labels.remove("RNNSegmenter_I_prob_min1")
			num_labels.remove("RNNSegmenter_I_prob_pls1")
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
		data_y = []
		for t in gold_feats:
			if "B-Conn" in t['label']:
				#data_y.append((1,0))
				data_y.append(1)
			elif "I-Conn" in t["label"]:
				#data_y.append((0,1))
				data_y.append(2)
			else:
				#data_y.append((0,0))
				data_y.append(0)

		# data_y = [int(t['label'] != "_") for t in gold_feats]
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

		if tune_mode is not None and size is None:
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
				'average': hp.choice('average',["micro","weighted"]),
				'n_estimators': scope.int(hp.quniform('n_estimators', 50, 150, 10)),
				'max_depth': scope.int(hp.quniform('max_depth', 3, 35, 1)),
				'eta': scope.float(hp.quniform('eta', 0.01, 0.2, 0.01)),
				'gamma': scope.float(hp.quniform('gamma', 0.01, 0.2, 0.01)),
				'colsample_bytree': hp.choice('colsample_bytree', [0.4,0.5,0.6,0.7,0.8,1.0]),
				'subsample': hp.choice('subsample', [0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
				'clf': hp.choice('clf', ["xgb"])
			}
			best_clf, best_params = hyper_optimize(data_x,data_y,val_x=val_x,val_y=val_y,space=space)
			return best_clf, best_params
		elif tune_mode is not None:
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
			with io.open(segmenters_dir + os.sep + "params" + os.sep + "EnsembleConnective"+self.auto+"_best_params.tab",'a',encoding="utf8") as bp:
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

		:param conllu: File in DISRPT shared task *.conll format
		:param eval_gold: Whether to score the prediction; only applicable if using a gold .conll file as input
		:param as_text: Boolean, whether the input is a string, rather than a file name to read
		:param serialize: Whether to serialize prediction as a .conll file
		:return: tokenwise prediction vector if eval_gold is False, otherwise prints evaluation metrics and diff to gold
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
			# _, preds[e.name + "_prob"] = [list(x) for x in zip(*pred)]
			preds[e.name + "_B_prob"] = []
			preds[e.name + "_I_prob"] = []
			if "Freq" in e.name:
				preds[e.name + "_freq"] = []
				headers.append(e.name + "_freq")
			for tup in pred:
				if "RNN" in e.name:
					pred = tup[0]
					probas = tup[1]
					freqs = None
				else:
					pred = tup[1]
					probas = float(tup[2])
					freqs = float(tup[3])
				if "B-Conn" in pred:
					preds[e.name + "_B_prob"].append(probas)
					preds[e.name + "_I_prob"].append(0.0)
				elif "I-Conn" in pred:
					preds[e.name + "_B_prob"].append(0.0)
					preds[e.name + "_I_prob"].append(probas)
				else:
					preds[e.name + "_B_prob"].append(0.0)
					preds[e.name + "_I_prob"].append(0.0)
				if "Freq" in e.name:
					preds[e.name + "_freq"].append(freqs)

			headers.append(e.name + "_B_prob")
			headers.append(e.name + "_I_prob")

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
				item[e.name + "_B_prob"] = preds[e.name + "_B_prob"][i]
				item[e.name + "_I_prob"] = preds[e.name + "_I_prob"][i]
				if e.name == "FreqConnDetector":
					item[e.name + "_freq"] = preds[e.name + "_freq"][i]

			feats = []
			for k in headers:
				feats.append(item[k])

			data.append(feats)

		data, headers, _, _ = self.n_gram(data,headers,[],[])

		data = pd.DataFrame(data, columns=headers)
		data_encoded = self.multicol_transform(data,columns=multicol_dict["columns"],all_encoders_=multicol_dict["all_encoders_"])

		data_x = data_encoded[cat_labels+num_labels].values
		preds = clf.predict(data_x)

		if eval_gold:
			gold_feats, _,_,_,_ = read_conll(conllu,genre_pat=self.genre_pat,mode="seg",as_text=True)

			# Array to keep labels for diff
			gold = []
			for t in gold_feats:
				if "B-Conn" in t['label']:
					gold.append("Seg=B-Conn")
				elif "I-Conn" in t['label']:
					gold.append("Seg=I-Conn")
				else:
					gold.append("_")
			gold = np.asarray(gold)

			# Generate response conllu
			lines = conllu.split("\n")
			processed = []
			pred_labs = []
			i = 0
			for line in lines:
				if "\t" in line:
					fields = line.split('\t')
					if "-" in fields[0]:
						processed.append(line)
						continue
					else:
						if preds[i] == 0:
							pred = "_"
						elif preds[i] == 1:
							pred = "Seg=B-Conn"
						else:
							pred = "Seg=I-Conn"
						pred_labs.append(pred)
						fields[-1]=pred
						processed.append("\t".join(fields))
						i+=1
				else:
					processed.append(line)
			processed = "\n".join(processed) + "\n"

			score_dict = get_scores(conllu,processed,string_input=True)

			print("o Total tokens: " + str(score_dict["tok_count"]))
			print("o Gold " +score_dict["seg_type"]+": " + str(score_dict["gold_seg_count"]))
			print("o Predicted "+score_dict["seg_type"]+": " + str(score_dict["pred_seg_count"]))
			print("o Precision: " + str(score_dict["prec"]))
			print("o Recall: " + str(score_dict["rec"]))
			print("o F-Score: " + str(score_dict["f_score"]))

			if serialize:
				self.serialize(conllu,pred_labs)

			with io.open("diff.tab",'w',encoding="utf8") as f:
				for i in range(len(pred_labs)):
					f.write("\t".join([toks[i],str(gold[i]),str(pred_labs[i])])+"\n")
			return score_dict["f_score"]
		else:
			return preds

	def serialize(self,conllu_in,predictions):
		output = []
		i = 0
		for line in conllu_in.split("\n"):
			if "\t" in line:
				fields = line.split("\t")
				if "-" in fields[0]:
					output.append(line)
				else:
					fields[-1] = predictions[i]
					i+=1
					output.append("\t".join(fields))
			else:
				output.append(line)
		with io.open(self.corpus + "_pred.out.conll",'w',encoding="utf8") as outfile:
			outfile.write("\n".join(output) + "\n")

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
	p.add_argument("-c","--corpus",default="zho.pdtb.cdtb",help="Corpus to use or 'all'")
	p.add_argument("-d","--data_dir",default="../data",help="Path to shared task data folder")
	p.add_argument("-m","--multitrain",action="store_true",help="Load saved predictions from ensemble multitraining")
	p.add_argument("-t","--tune_mode",default="paramwise",choices=["paramwise","full","hyperopt"])
	p.add_argument("-b","--best_params",action="store_true",help="Load best parameters from file")
	p.add_argument("-o","--outfile",action="store_true",help="Print output file CORPUS.pred.out.conll")
	p.add_argument("--mode",choices=["train","test","train-test","optimize-train-test"],default="train-test")
	p.add_argument("--eval_test",action="store_true",help="Evaluate on test, not dev")
	p.add_argument("--auto",action="store_true",help="Evaluate on automatic parse")
	opts = p.parse_args()

	corpus = opts.corpus
	if "pdtb" not in corpus:
		sys.stderr.write("Dataset not in format PDTB.")
		sys.exit()
	data_dir = opts.data_dir

	if opts.auto:
		data_dir+="_parsed"
		auto="_auto"
	else:
		auto=""


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
	e = EnsembleConnective(lang=lang,model=corpus,multitrain=opts.multitrain,auto=auto)

	best_params = None
	if "optimize" in opts.mode:
		clf, vars, best_params = e.optimize(train,size=50000,tune_mode=opts.tune_mode,cached_params=opts.best_params)
		if "best_score" in best_params:
			best_params.pop("best_score")
		# Now train on whole training set with those variables
		sys.stderr.write("\no Training best configuration\n")
		e.train(train,rare_thresh=100,clf_params=best_params,as_text=False,chosen_clf=clf)
	elif "train" in opts.mode:
		sys.stderr.write("\no Training on corpus "+corpus+"\n")
		tune_mode=None if opts.tune_mode != "hyperopt" else "hyperopt"
		e.train(train,as_text=False,tune_mode=tune_mode)

	if "test" in opts.mode:
		if opts.eval_test:
			f1 = e.predict(test,eval_gold=True,as_text=False,serialize=opts.outfile)
		else:
			f1 = e.predict(dev,eval_gold=True,as_text=False,serialize=opts.outfile)


			if best_params is not None and "optimize" in opts.mode:  # For optimization check if this is a new best score
				best_clf = e.clf
				prev_best_score = get_best_score(corpus,"EnsembleConnective"+e.auto)
				if f1 > prev_best_score:
					sys.stderr.write("o New best F1: " + str(f1) + "\n")
					with io.open(segmenters_dir + os.sep + "params" + os.sep + "EnsembleConnective"+e.auto+"_best_params.tab",'a',encoding="utf8") as bp:
						for k, v in best_params.items():
							bp.write("\t".join([corpus, best_clf.__class__.__name__, k, str(v)])+"\n")
						bp.write("\t".join([corpus, best_clf.__class__.__name__, "best_score", str(f1)])+"\n\n")

