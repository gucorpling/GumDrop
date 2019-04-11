"""
SubtreeSegmenter.py

A discourse unit segmentation module based on a moving window capturing parts of
a dependency syntax parse.
"""

import io, sys, os, copy

# Allow package level imports in module
script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "..")
models = os.path.abspath(script_dir + os.sep + ".."+os.sep+".."+os.sep + "models")
sys.path.append(lib)

from collections import defaultdict, Counter
from argparse import ArgumentParser
#os.environ['OMP_NUM_THREADS'] = "1"
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from conll_reader import read_conll, get_multitrain_preds
from tune import permutation_importances, report_correlations, report_theils_u, get_best_params, get_best_score, hyper_optimize, grid_search

np.random.seed(42)
import random
random.seed(42)

DEFAULTCLF = RandomForestClassifier(random_state=42)
DEFAULTCLF = XGBClassifier(random_state=42, max_depth=50, min_child_weight=1,  n_estimators=200, n_jobs=3 , verbose=1,learning_rate=0.16)
DEFAULTPARAMS = {"n_estimators":250,"min_samples_leaf":3,"max_features":10,"random_state":42}



class SubtreeSegmenter:

	def __init__(self,lang="eng",model=None,multifolds=5,auto=""):
		self.name = "SubtreeSegmenter"
		self.genre_pat = "^(..)"  # By default 2 first chars of docname identify genre
		if "gum" in model:
			self.genre_pat = "GUM_([^_]+)_"
		self.lang = lang
		self.multifolds = multifolds
		self.corpus = model
		self.auto = auto
		if model is not None:
			self.model = models + os.sep + model + auto +  "_subtreeseg.pkl"
		else:
			self.model = ".." + os.sep + ".." + os.sep + "models" + os.sep + auto + "subtreeseg.pkl"
		self.corpus_dir = None
		self.clf = DEFAULTCLF

	def read_data(self,infile,size,as_text,rare_thresh,chosen_feats=None):

		cap = 3*size if size is not None else None
		train_feats, vocab, toks, firsts, lasts = read_conll(infile,genre_pat=self.genre_pat,mode="seg",cap=cap,char_bytes=self.lang=="zho",as_text=as_text)
		vocab = Counter(vocab)
		top_n_words = vocab.most_common(rare_thresh)
		top_n_words, _ = zip(*top_n_words)
		for tok in train_feats:
			if tok["word"] not in top_n_words:
				tok["word"] = tok["pos"]

		tokens_by_abs_id = self.traverse_trees(train_feats)

		data, headers = self.n_gram(train_feats,tokens_by_abs_id)

		# Features to use for all n-gram tokens
		num_labels = ["head_dist","left_span","right_span","samepar_left","tok_len"]
		cat_labels = ["case","closest_left","closest_right","deprel","farthest_left","farthest_right","pos","word","morph","cpos","depchunk"]

		pref_cat = []
		pref_num = []
		for pref in ["mn2","mn1","par","par_par","pl1","pl2"]:
			pref_cat += [pref + "_" + h for h in cat_labels]
			pref_num += [pref + "_" + h for h in num_labels]

		# Features only needed for node token
		cat_labels += ["genre"] + pref_cat #+ ["heading_first","heading_last"]#+ ["s_type"]
		num_labels += ["dist2end","sent_doc_percentile","tok_id","wid","quote","rank"] + pref_num # + ["bracket"]
		num_labels += ["par_quote","par_par_quote"]#,"par_bracket","par_par_bracket"]

		# Use specific feature subset
		if chosen_feats is not None:
			new_cat = []
			new_num = []
			for feat in chosen_feats:
				if feat in cat_labels:
					new_cat.append(feat)
				elif feat in num_labels:
					new_num.append(feat)
			cat_labels = new_cat
			num_labels = new_num

		data = pd.DataFrame(data, columns=headers)
		data_encoded, multicol_dict = self.multicol_fit_transform(data, pd.Index(cat_labels))

		data_x = data_encoded[cat_labels+num_labels].values
		data_y = np.where(data_encoded['label'] == "_", 0, 1)

		return data_encoded, data_x, data_y, cat_labels, num_labels, multicol_dict, firsts, lasts, top_n_words


	def train(self,training_file,rare_thresh=200,clf_params=None,chosen_feats=None,tune_mode=None,size=None,as_text=True,multitrain=False,chosen_clf=DEFAULTCLF):
		"""
		:param training_file:
		:param rare_thresh:
		:param clf_params:
		:param chosen_feats: List of feature names to force a subset of selected features to be used
		:param tune_mode: None for no grid search, "paramwise" to tune each hyperparameter separately, or "full" for complete grid (best but slowest)
		:param size: Sample size to optimize variable importance with
		:return:
		"""

		if tune_mode is not None and size is None:
			size = 5000
			sys.stderr.write("o No sample size set - setting size to 5000\n")
		if clf_params is None:
			# Default classifier parameters
			clf_params = {"n_estimators":150,"min_samples_leaf":3, "random_state":42}
			if DEFAULTCLF.__class__.__name__ not in ["GradientBoostingClassifier","CatBoostClassifier","XGBClassifier"]:
				clf_params.update({"n_jobs":4, "oob_score":True, "bootstrap":True})

		data_encoded, data_x, data_y, cat_labels, num_labels, multicol_dict, firsts, lasts, top_n_words = self.read_data(training_file,size,as_text=as_text,rare_thresh=rare_thresh,chosen_feats=chosen_feats)
		sys.stderr.write("o Learning...\n")

		if tune_mode is not None:
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

		if tune_mode == "importances":
			sys.stderr.write("o Measuring correlation of categorical variables\n")
			theil_implications = report_theils_u(val_x,cat_labels)
			for (var1, var2) in theil_implications:
				if var1 in cat_labels and var2 in cat_labels and var2 !="word":
					drop_var = var2
					u = theil_implications[(var1, var2)]
					sys.stderr.write("o Removed feature " + drop_var + " due to Theil's U " + str(u)[:6] + " of " + var1 + "->" + var2 + "\n")
					cat_labels.remove(drop_var)
			sys.stderr.write("o Measuring correlation of numerical variables\n")
			cor_mat = report_correlations(val_x[num_labels],thresh=0.95)
			for (var1, var2) in cor_mat:
				if var1 in num_labels and var2 in num_labels:
					drop_var = var2 # if imp[var1] > imp[var2] else var1
					if drop_var == "word":
						continue
					corr_level = cor_mat[(var1, var2)]
					sys.stderr.write("o Removed feature " + drop_var + " due to correlation " + str(corr_level) + " of " + var1 + ":" + var2 + "\n")
					num_labels.remove(drop_var)

			return cat_labels, num_labels

		if tune_mode in ["paramwise","full"]:  # Grid Search
			best_clf, best_params = grid_search(data_x,data_y,tune_mode,clf_params)
			clf_name = best_clf.__class__.__name__
			self.clf = best_clf
			return best_clf, best_params
		elif tune_mode == "hyperopt":  # TPE guided random search
			from hyperopt import hp
			from hyperopt.pyll.base import scope
			val_x, val_y = None, None
			if self.corpus_dir is not None:
				dev_file = self.corpus_dir + os.sep + self.corpus + "_dev.conll"
				_, val_x, val_y, _, _, _, _, _, _ = self.read_data(dev_file,size,as_text=False,rare_thresh=rare_thresh,chosen_feats=chosen_feats)
			space = {
				'n_estimators': scope.int(hp.quniform('n_estimators', 100, 250, 10)),
				'max_depth': scope.int(hp.quniform('max_depth', 3, 30, 1)),
				'eta': scope.float(hp.quniform('eta', 0.01, 0.2, 0.01)),
				'gamma': scope.float(hp.quniform('gamma', 0.01, 0.2, 0.01)),
				'colsample_bytree': hp.choice('colsample_bytree', [0.4,0.5,0.6,0.7,1.0]),
				'subsample': hp.choice('subsample', [0.5,0.6,0.7,0.8,1.0]),
				'clf': hp.choice('clf', ["xgb"])
			}
			best_clf, best_params = hyper_optimize(data_x.values,data_y,val_x=None,val_y=None,space=space,max_evals=20)
			return best_clf, best_params
		else:  # No hyperparameter optimization
			clf = chosen_clf if chosen_clf is not None else DEFAULTCLF
			sys.stderr.write("o Setting params " + str(clf_params) + "\n")
			clf.set_params(**clf_params)
			if clf.__class__.__name__ not in ["GradientBoostingClassifier","CatBoostClassifier","XGBClassifier"]:
				clf.set_params(**{"n_jobs":3,"oob_score":True,"bootstrap":True})
			if clf.__class__.__name__ in ["XGBClassifier"]:
				clf.set_params(**{"n_jobs":3})
			clf.set_params(**{"random_state":42})
			if multitrain:
				multitrain_preds = get_multitrain_preds(clf,data_x,data_y,self.multifolds)
				multitrain_preds = "\n".join(multitrain_preds.strip().split("\n")[1:-1])  # Remove OOV tokens at start and end
				with io.open(script_dir + os.sep + "multitrain" + os.sep + self.name + self.auto + '_' + self.corpus,'w',newline="\n") as f:
					sys.stderr.write("o Serializing multitraining predictions\n")
					f.write(multitrain_preds)
			if  clf.__class__.__name__ == "CatBoostClassifier":
				clf.fit(data_x,data_y,cat_features=list(range(len(cat_labels))))
			else:
				clf.fit(data_x,data_y)
			self.clf = clf

		feature_names = cat_labels + num_labels
		sys.stderr.write("o Using " + str(len(feature_names)) + " features\n")

		zipped = zip(feature_names, clf.feature_importances_)
		sorted_zip = sorted(zipped, key=lambda x: x[1], reverse=True)
		sys.stderr.write("o Feature Gini importances:\n\n")
		for name, importance in sorted_zip:
			sys.stderr.write(name + "=" + str(importance) + "\n")

		if self.clf.__class__.__name__ not in ["GradientBoostingClassifier","CatBoostClassifier","XGBClassifier"]:
			sys.stderr.write("\no OOB score: " + str(clf.oob_score_)+"\n\n")

		if tune_mode=="permutation":
			# Filter features based on permutation importance score threshold
			imp = permutation_importances(clf,val_x,val_y)
			for var, score in imp.items():
				if score < 0 and var != "word":
					sys.stderr.write("o Dropping feature " + var + " due to low permutation importance of " + str(score) + "\n")
					if var in cat_labels:
						cat_labels.remove(var)
					elif var in num_labels:
						num_labels.remove(var)
			sys.stderr.write("o Measuring correlation of numerical variables\n")
			cor_mat = report_correlations(val_x[num_labels])
			for (var1, var2) in cor_mat:
				if var1 in num_labels and var2 in num_labels:
					drop_var = var2 if imp[var1] > imp[var2] else var1
					if drop_var == "word":
						continue
					corr_level = cor_mat[(var1, var2)]
					sys.stderr.write("o Removed feature " + drop_var + " due to correlation " + str(corr_level) + " of " + var1 + ":" + var2 + "\n")
					num_labels.remove(drop_var)

			return cat_labels, num_labels

		sys.stderr.write("\no Serializing model...\n")

		joblib.dump((clf, num_labels, cat_labels, multicol_dict, top_n_words, firsts, lasts), self.model, compress=3)

	def predict_cached(self,train=None):
		pairs = io.open(script_dir + os.sep + "multitrain" + os.sep + self.name + self.auto + '_' + self.corpus).read().split("\n")
		preds = [(int(pr.split()[0]), float(pr.split()[1])) for pr in pairs if "\t" in pr]
		return preds

	def predict(self, infile, eval_gold=False, as_text=True):
		"""
		Predict sentence splits using an existing model

		:param infile: File in DISRPT shared task *.tok or *.conll format (sentence breaks will be ignored in .conll)
		:param eval_gold: Whether to score the prediction; only applicable if using a gold .conll file as input
		:param genre_pat: A regex pattern identifying the document genre from document name comments
		:param as_text: Boolean, whether the input is a string, rather than a file name to read
		:return: tokenwise binary prediction vector if eval_gold is False, otherwise prints evaluation metrics and diff to gold
		"""

		if self.model is None:  # Try default model location
			model_path = ".." + os.sep + ".." + os.sep + "models" + os.sep + "subtreeseg.pkl"
		else:
			model_path = self.model

		clf, num_labels, cat_labels, multicol_dict, top_n_words, firsts, lasts = joblib.load(model_path)

		feats, _, toks, _, _ = read_conll(infile,genre_pat=self.genre_pat,mode="seg",as_text=as_text,char_bytes=self.lang=="zho")
		tokens_by_abs_id = self.traverse_trees(feats)
		feats, headers = self.n_gram(feats,tokens_by_abs_id,dummies=False)

		temp = []
		headers_with_oov = ["first","last","deprel","closest_left","closest_right","farthest_left","farthest_right",
							"pos","cpos","morph","s_type","depchunk"]
		for pref in ["mn2","mn1","par","par_par","pl1","pl2"]:
			temp += [pref + "_" + h for h in headers_with_oov]
		headers_with_oov += temp

		genre_warning = False
		for i, header in enumerate(headers):
			if header in headers_with_oov and header in cat_labels:
				for item in feats:
					if item[i] not in multicol_dict["encoder_dict"][header].classes_:
						item[i] = "_"
			elif header == "genre" and "genre" in cat_labels:
				for item in feats:
					if item[i] not in multicol_dict["encoder_dict"]["genre"].classes_:  # New genre not in training data
						if not genre_warning:
							sys.stderr.write("! WARN: Genre not in training data: " + item[i] + "; suppressing further warnings\n")
							genre_warning = True
						item[i] = "_"
			elif header.endswith("word") and header in cat_labels:
				for item in feats:
					# Replace rare words and words never seen before in this position with POS
					if item[i] not in top_n_words or item[i] not in multicol_dict["encoder_dict"][header].classes_:
						pos_col = headers.index(header.replace("word","pos"))
						if item[pos_col] in multicol_dict["encoder_dict"][header].classes_:
							item[i] = item[pos_col]
						else:
							item[i] = "_"
		data = feats
		data = pd.DataFrame(data, columns=headers)
		data_encoded = self.multicol_transform(data,columns=multicol_dict["columns"],all_encoders_=multicol_dict["all_encoders_"])

		data_x = data_encoded[cat_labels+num_labels].values

		probas = clf.predict_proba(data_x)
		probas = [p[1] for p in probas]
		preds = [int(p>0.5) for p in probas]

		for i, p in enumerate(preds):
			if data["tok_id"].values[i] == 1:  # Ensure tok_id 1 is always a segment start
				preds[i] = 1

		if eval_gold:
			gold = np.where(data_encoded['label'] == "_", 0, 1)
			conf_mat = confusion_matrix(gold, preds)
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
					f.write("\t".join([toks[i],str(gold[i]),str(preds[i])])+"\n")

			return conf_mat, prec, rec, f1
		else:
			return zip(preds,probas)

	def optimize(self, train, rare_thresh=200, size=5000, tune_mode="paramwise",as_text=False, cached_params=False):

		# Estimate useful features on a random sample of |size| instances
		selected_cat, selected_num = self.train(train,rare_thresh=rare_thresh,tune_mode="importances",size=size,as_text=as_text)
		selected_feats = selected_cat + selected_num
		with io.open(script_dir + os.sep + "SubtreeSegmenter_best_params"+self.auto+".tab",'a',encoding="utf8") as bp:
			bp.write(self.corpus + "\t"+self.clf.__class__.__name__+"\tfeatures\t" + ",".join(selected_feats)+"\n")
		sys.stderr.write("o Chose "+str(len(selected_feats))+" features: " + ",".join(selected_feats)+"\n")
		if tune_mode != "features":
			sys.stderr.write("o Tuning hyperparameters\n\n")

		# Optimize hyperparameters via grid search
		if cached_params:
			best_clf, best_params, _ = get_best_params(self.corpus, self.name)
			sys.stderr.write("\no Using cached best hyperparameters\n")
		elif tune_mode!="features":
			best_clf, best_params = self.train(train,rare_thresh=rare_thresh,tune_mode=tune_mode,size=200000,as_text=as_text, chosen_feats=selected_feats)
			sys.stderr.write("\no Found best hyperparameters\n")
		else:
			best_clf = DEFAULTCLF
			best_params = DEFAULTPARAMS
			sys.stderr.write("\no Using default hyperparameters\n")

		for key, val in best_params.items():
			sys.stderr.write(key + "\t" + str(val) + "\n")
		sys.stderr.write(best_clf.__class__.__name__ + "\n")
		sys.stderr.write("\n")

		return best_clf, selected_feats, best_params

	@staticmethod
	def traverse_trees(tokens):

		tokens_by_abs_id = {}

		def get_descendants(parent_id, children_dict, seen_tokens):
			# Helper function to recursively collect children of children

			my_descendants = []
			my_descendants += children_dict[parent_id]

			for child in children_dict[parent_id]:
				if child["abs_id"] in seen_tokens:
					sys.stderr.write("\nCycle detected in syntax tree in sentence " + str(child["s_id"])+" token: "+child["word"]+"\n")
					sys.exit("Exiting due to invalid input\n")
				else:
					seen_tokens.add(child["abs_id"])
			for child in children_dict[parent_id]:
				child_id = child["abs_id"]
				if child_id in children_dict:
					my_descendants += get_descendants(child_id, children_dict, seen_tokens)
			return my_descendants

		def get_rank(tok, token_dict, rank=0):
			# Helper function to determine tokens' graph depth
			if tok["abs_parent"].endswith("_0"):
				return rank
			else:
				rank+=1
				return get_rank(token_dict[tok["abs_parent"]],token_dict,rank=rank)

		# Make unique ids
		for tok in tokens:
			tok["abs_id"] = str(tok["s_id"]) + "_" + str(tok["wid"])
			tok["abs_parent"] = str(tok["s_id"]) + "_" + str(tok["head"])
			tok["descendants"] = []  # Initialize descendant list
			tokens_by_abs_id[str(tok["s_id"]) + "_" + str(tok["wid"])] = tok

		# Add dist2end feature (=reverse id)
		for tok in tokens:
			tok["dist2end"] = tok["s_len"]-tok["wid"]

		# Make children dict
		children = defaultdict(list)
		for tok in tokens:
			if not tok["abs_parent"].endswith("_0"):
				children[tok["abs_parent"]].append(tok)

		# Recursively get descendants
		for parent_id in children:
			seen_tokens = set()
			parent = tokens_by_abs_id[parent_id]
			parent["descendants"] = get_descendants(parent_id, children, seen_tokens)

		# Compute graph rank for each token
		for tok in tokens:
			tok["rank"] = get_rank(tok, tokens_by_abs_id, 0)

		# Use descendant dictionary to find closest/farthest left/right children's network
		for tok in tokens:
			tok["farthest_left"] = tok
			tok["farthest_right"] = tok
			tok["closest_right"] = tok
			tok["closest_left"] = tok
			tok["right_span"] = 0
			tok["left_span"] = 0
			d_list = sorted(tok["descendants"],key=lambda x: x["tok_id"])
			for d in d_list:
				d_id = d["tok_id"]
				t_id = tok["tok_id"]
				if d_id < t_id:  # Left child
					if d_id < int(tok["farthest_left"]["tok_id"]):
						tok["farthest_left"] = d
						# tok["left_span"] = self.bin_numbers(tok["left_span"] ,bin_splits=[-6,-3,-1,0,1,2,4,7])
						tok["left_span"] = int(tok["tok_id"]) - int(d["tok_id"])
					if (d_id > int(tok["closest_left"]["tok_id"]) and d_id < tok["tok_id"]) or (d_id < tok["tok_id"] and tok["closest_left"] == tok):
						tok["closest_left"] = d
				else:  # Right child
					if d_id > int(tok["farthest_right"]["tok_id"]):
						tok["farthest_right"] = d
						tok["right_span"] = int(d["tok_id"]) - int(tok["tok_id"])
					if (d_id < tok["closest_right"]["tok_id"] and d_id > tok["tok_id"]) or (d_id > tok["tok_id"] and tok["closest_right"] == tok):
						tok["closest_right"] = d

			# Represent child network as deprels
			for prop in ["closest_right","closest_left","farthest_right","farthest_left"]:
				if tok[prop] == tok:
					tok[prop] = "_"
				else:
					tok[prop] = tok[prop]["deprel"]

		# Add same parent features (whether a token has the same parent as its right/left neighbors)
		tokens[0]["samepar_left"] = 0
		tokens[-1]["samepar_right"] = 0
		for i in range(1,len(tokens)-1):
			prev, tok, next = tokens[i-1], tokens[i], tokens[i+1]
			if prev["abs_parent"] == tok["abs_parent"]:
				prev["samepar_right"] = 1
				tok["samepar_left"] = 1
			else:
				prev["samepar_right"] = 0
				tok["samepar_left"] = 0
			if next["abs_parent"] == tok["abs_parent"]:
				tok["samepar_right"] = 1
				next["samepar_left"] = 1
			else:
				tok["samepar_right"] = 0
				next["samepar_left"] = 0

		return tokens_by_abs_id

	@staticmethod
	def bin_numbers(number,bin_splits=None):
		if bin_splits is None:
			return 1  # Single bin
		else:
			for i in bin_splits:
				if number >= i:
					return i
			return bin_splits[0]  # If number not greater than any split, it belongs in minimum bin

	@staticmethod
	def n_gram(data, tokens_by_abs_id, dummies=True):
		"""
		Turns unigram list of feature dicts into list of five-skipgram+parent features by adding features of adjacent tokens

		:param data: input tokens as a list of dictionaries, each filled with token property key-values
		:param tokens_by_abs_id: dictionary of absolute sent+word IDs to the corresponding token property dictionary
		:param dummies: Boolean, whether to wrap data with dummy -2, -1 ... +1 +2 tokens for training (should be False when predicting)
		:return: n_grammified token list without feature names, and list of header names
		"""
		n_grammed = []

		# Remove unneeded features
		del_props = ["descendants","lemma","docname","head"]
		for tok in data:
			for prop in del_props:
				tok.pop(prop)

		base_headers = sorted(data[0].keys())
		headers = copy.deepcopy(base_headers)
		# Create fake root token to represent parent of root tokens
		root_type = copy.deepcopy(data[0])
		root_type.update({"word":"_","deprel":"_","first":"_","last":"_","genre":"_","closest_left":"_",
						  "closest_right":"_","farthest_left":"_","farthest_right":"_","pos":"_","cpos":"_","morph":"_"})
		# Also use this token to introduce "_" as possible feature value for OOV cases
		oov_type = copy.deepcopy(root_type)
		oov_type["abs_id"] = "OOV"
		oov_type["abs_parent"] = "OOV"
		tokens_by_abs_id["OOV"] = oov_type

		for pref in ["mn2","mn1","par","par_par","pl1","pl2"]:
			headers += [pref + "_" + h for h in base_headers]

		# During training, pseudo-wrap extra tokens to enable 5 skip grams
		wrapped = []
		wrapped.append(copy.deepcopy(data[-2]))
		wrapped.append(copy.deepcopy(data[-1]))
		if dummies:
			wrapped.append(oov_type)
		wrapped += data
		if dummies:
			wrapped.append(oov_type)
		wrapped.append(copy.deepcopy(data[0]))
		wrapped.append(copy.deepcopy(data[1]))
		data = wrapped

		for i in range(2,len(data)-2):
			tok = data[i]
			prev_prev = data[i-2]
			prev = data[i-1]
			next = data[i+1]
			next_next = data[i+2]
			if tok["abs_parent"] in tokens_by_abs_id:
				par = tokens_by_abs_id[tok["abs_parent"]]
			else:
				par = root_type
			if par["abs_parent"] in tokens_by_abs_id:
				par_par = tokens_by_abs_id[par["abs_parent"]]
			else:
				par_par = root_type

			prev_prev_props = [prev_prev[k] for k in sorted(prev_prev.keys())]
			prev_props = [prev[k] for k in sorted(prev.keys())]
			tok_props = [tok[k] for k in sorted(tok.keys())]
			par_props = [par[k] for k in sorted(par.keys())]
			par_par_props = [par_par[k] for k in sorted(par_par.keys())]
			next_props = [next[k] for k in sorted(next.keys())]
			next_next_props = [next_next[k] for k in sorted(next_next.keys())]

			n_grammed.append(tok_props + prev_prev_props + prev_props + par_props + par_par_props + next_props + next_next_props)

		return n_grammed, headers

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


if __name__ == "__main__":

	p = ArgumentParser()
	p.add_argument("-c","--corpus",default="spa.rst.sctb",help="corpus to use or 'all'")
	p.add_argument("-d","--data_dir",default=os.path.normpath("../../../data"),help="Path to shared task data folder")
	p.add_argument("-s","--sample_size",type=int,default=5000,help="Sample size to use for feature selection")
	p.add_argument("-t","--tune_mode",default=None,choices=[None,"paramwise","full","hyperopt","features","permutation"])
	p.add_argument("-r","--rare_thresh",type=int,default=200,help="Threshold rank for replacing words with POS tags")
	p.add_argument("-m","--multitrain",action="store_true",help="Perform multitraining and save predictions for ensemble training")
	p.add_argument("-b","--best_params",action="store_true",help="Load best parameters from file")
	p.add_argument("--mode",action="store",default="test",choices=["train","train-test","optimize-train-test","test"])
	p.add_argument("--eval_test",action="store_true",help="Evaluate on test, not dev")
	p.add_argument("--auto",action="store_true",help="Evaluate on automatic parse")
	opts = p.parse_args()

	data_dir = opts.data_dir
	rare_thresh = opts.rare_thresh
	tune_mode = opts.tune_mode

	if opts.auto:
		data_dir = data_dir + "_parsed"
		sys.stderr.write("o Evaluating on automatically parsed data\n")

	corpora = os.listdir(data_dir)
	if opts.corpus == "all":
		corpora = [c for c in corpora if os.path.isdir(os.path.join(data_dir, c))]
	else:
		corpora = [c for c in corpora if os.path.isdir(os.path.join(data_dir, c)) and c == opts.corpus]

	for corpus in corpora:
		if "pdtb" in corpus:
			continue

		sys.stderr.write("o Corpus: " + corpus + "\n")

		train = os.path.join(data_dir,corpus, corpus + "_train.conll")
		dev = os.path.join(data_dir, corpus, corpus + "_dev.conll")
		test = os.path.join(data_dir, corpus, corpus + "_test.conll")

		if "." in corpus:
			lang = corpus.split(".")[0]
		else:
			lang = "eng"

		auto = "" if not opts.auto else "_auto"

		seg = SubtreeSegmenter(lang=lang,model=corpus,auto=auto)
		seg.corpus_dir = data_dir + os.sep + corpus

		# Special genre patterns and feature settings
		if "gum" in corpus:
			seg.genre_pat = "GUM_(.+)_.*"

		best_params = None
		if "optimize" in opts.mode:
			best_clf, vars, best_params = seg.optimize(train,size=opts.sample_size,tune_mode=tune_mode,rare_thresh=rare_thresh,as_text=False, cached_params=opts.best_params)
			# Now train on whole training set with those variables
			if "best_score" in best_params:
				best_params.pop("best_score")
			sys.stderr.write("\no Training best configuration\n")
			seg.train(train,chosen_feats=vars,rare_thresh=rare_thresh,clf_params=best_params,as_text=False,chosen_clf=best_clf)
		elif "train" in opts.mode:
			feats = None
			params = None
			best_clf = None
			if opts.best_params:
				best_clf, params, feats = get_best_params(corpus, "SubtreeSegmenter" + auto)
				if len(feats) == 0:
					feats = None
			seg.train(train,rare_thresh=rare_thresh,as_text=False,multitrain=opts.multitrain,chosen_feats=feats,clf_params=params,chosen_clf=best_clf)
		if "test" in opts.mode:
			if opts.multitrain:
				# Get prediction performance on out-of-fold
				preds = seg.predict_cached()
			else:
				# Get prediction performance on dev
				if opts.eval_test:
					conf_mat, prec, rec, f1 = seg.predict(test,eval_gold=True,as_text=False)
				else:
					conf_mat, prec, rec, f1 = seg.predict(dev,eval_gold=True,as_text=False)
					if best_params is not None and "optimize" in opts.mode:  # For optimization check if this is a new best score
						prev_best_score = get_best_score(corpus,"SubtreeSegmenter" + auto)
						if f1 > prev_best_score:
							sys.stderr.write("o New best F1: " + str(f1) + "\n")
							print(seg.clf.__dict__)
							with io.open(script_dir + os.sep + "params" + os.sep + "SubtreeSegmenter"+auto+"_best_params.tab",'a',encoding="utf8") as bp:
								for k, v in best_params.items():
									bp.write("\t".join([corpus, best_clf.__class__.__name__, k, str(v)])+"\n")
								bp.write("\t".join([corpus, best_clf.__class__.__name__, "features", ",".join(vars)])+"\n")
								bp.write("\t".join([corpus, best_clf.__class__.__name__, "best_score", str(f1)])+"\n\n")

