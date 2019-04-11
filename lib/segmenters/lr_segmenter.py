import re, sys, os, operator, argparse, pickle, io, time, gc
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
from collections import Counter
from glob import glob
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
from datetime import timedelta
script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "..")
sys.path.append(lib)
from conll_reader import read_conll, tt_tag, udpipe_tag, shuffle_cut_conllu, get_multitrain_preds
from random import seed, shuffle

seed(42)

# Allow package level imports in module
model_dir = os.path.abspath(script_dir + os.sep + ".." + os.sep + ".." + os.sep + "models" + os.sep)



class LRSegmenter:

	def __init__(self,lang="eng",model="eng.rst.gum",windowsize=3):
		self.lang = lang
		self.name = "LRSegmenter"
		self.corpus = model
		lang_map = {"deu":"german","eng":"english","spa":"spanish","fra":"french","nld":"dutch","rus":"russian","eus":"basque","por":"portuguese","zho":"chinese","tur":"turkish"}
		self.long_lang = lang_map[lang] if lang in lang_map else lang
		try:
			self.udpipe_model = glob(os.path.abspath(os.path.join(lib,"udpipe",self.long_lang+"*.udpipe")))[0]
		except:
			sys.stderr.write("! Model not found for language " + self.long_lang + "*.udpipe in " + os.path.abspath(os.path.join([lib,"udpipe",self.long_lang+"*.udpipe"]))+"\n")
			sys.exit(0)
		self.udpipe_path = os.path.abspath(os.path.join(lib,"udpipe")) + os.sep
		self.model_path = model_dir + os.sep + model + "_lrseg" + ".pkl"
		self.windowsize=windowsize
		self.verbose = False
		self.readconllmode = "train"
		self.uni_cols = []

	def copyforprevnext(self, col_values, col_names, prevnexttype):
		new_names = [prevnexttype + '_' + x for x in col_names]

		nrows = col_values.shape[0]

		if prevnexttype == 'prev':
			data2 = np.append(col_values[nrows - 1:nrows, ], col_values[0:nrows - 1, ], axis=0)
		elif prevnexttype == 'prevprev':
			data2 = np.append(col_values[nrows - 2:nrows, ], col_values[0:nrows - 2, ], axis=0)
		elif prevnexttype == 'prevprevprev':
			data2 = np.append(col_values[nrows - 3:nrows, ], col_values[0:nrows - 3, ], axis=0)
		elif prevnexttype == 'next':
			data2 = np.append(col_values[1:nrows, ], col_values[0:1, ], axis=0)
		elif prevnexttype == 'nextnext':
			data2 = np.append(col_values[2:nrows, ], col_values[0:2, ], axis=0)
		elif prevnexttype == 'nextnextnext':
			data2 = np.append(col_values[3:nrows, ], col_values[0:3, ], axis=0)
		else:
			print("x wrong prev or next type.")
		# print(data1.head(10))
		return data2, new_names

	def numpyconcat2df(self, df1, colnames1, df2, colnames2):
		dfconcatvals = np.append(df1, df2, axis=1)
		dfconcatcols = colnames1 + colnames2
		return dfconcatvals, dfconcatcols

	# categorical feature to multiple one-hot vectors
	def cattodummy(self, cat_entries, cat_cols, num_entries, num_cols):
		assert cat_entries.shape[1] == len(cat_cols)

		for idv, v in enumerate(cat_cols):
			data1 = pd.DataFrame(cat_entries[:, idv], columns=[v])
			converted_df = pd.get_dummies(data1[v], prefix=v, drop_first=True)

			# remove OOV colnames if predict mode
			if self.readconllmode == "predict":
				converted_colnames = list(converted_df)
				filtered_train_cols = [x for x in self.uni_cols if x.startswith(v + '_')]
				diff_predictors = list(set(filtered_train_cols) - set(converted_colnames))
				for c in diff_predictors:
					converted_df[c] = 0
				revdiff_predictors = list(set(converted_colnames) - set(filtered_train_cols))
				converted_df = converted_df.drop(revdiff_predictors, axis=1)

			# print('dummy converted shape', v ,converted_df.shape)
			converted_df = converted_df[sorted(converted_df.columns)]
			converted_colnames = list(converted_df)
			# if v == "parentclauses":
			# 	print(converted_colnames)
			converted_entries = np.array(converted_df.values, dtype=np.float32)

			num_entries, num_cols = self.numpyconcat2df(num_entries, num_cols, converted_entries, converted_colnames)
			del data1
			del converted_df
			del converted_entries

		del cat_entries
		return num_entries, num_cols

	# categorical feature to one label mapped vector
	def cattolabel(self, cat_entries, cat_cols, num_entries, num_cols, mode='train'):
		assert cat_entries.shape[1] == len(cat_cols)

		from sklearn.preprocessing import LabelEncoder
		lb_make = LabelEncoder()

		for idv, v in enumerate(cat_cols):
			data1 = pd.DataFrame(cat_entries[:, idv], columns=[v])
			data1[v] = lb_make.fit_transform(data1[v])

			data1 = data1[sorted(data1.columns)]
			converted_colnames = list(data1)
			converted_entries = np.array(data1.values, dtype=np.float32)

			num_entries, num_cols = self.numpyconcat2df(num_entries, num_cols, converted_entries, converted_colnames)
			del data1
			del converted_entries
		del cat_entries
		return num_entries, num_cols


	# @profile
	def train(self,train_path,as_text=False,standardization=False,cut=True,multitrain=False):

		sys.stderr.write("o Reading training data...\n")

		if multitrain:
			X_train, colnames, Y_train, uni_cols = self.read_conll_sentbreak(train_path, neighborwindowsize=self.windowsize,as_text=as_text,cut=False,multitrain=multitrain)
		else:
			X_train, colnames, Y_train, uni_cols = self.read_conll_sentbreak(train_path, neighborwindowsize=self.windowsize,as_text=as_text,cut=cut)

		print("X_train size" ,X_train.shape)

		sys.stderr.write("o Building logistic regression now ...\n")

		logmodel = LogisticRegressionCV(cv=3,n_jobs=3,penalty='l1',solver="liblinear",random_state=42)

		if multitrain:
			if X_train.shape[0] <= 95000:
				multitrain_preds = get_multitrain_preds(logmodel,X_train,Y_train,5)
				multitrain_preds = "\n".join(multitrain_preds.strip().split("\n"))
				with io.open(script_dir + os.sep + "multitrain" + os.sep + self.name + '_' + self.corpus,'w',newline="\n") as f:
					sys.stderr.write("o Serializing multitraining predictions\n")
					f.write(multitrain_preds)
			else:
				sys.stderr.write('o Skipping multitrain\n')
		# Fit complete dataset
		logmodel.fit(X_train, Y_train)
		logmodel.sparsify()

		if multitrain and X_train.shape[0] > 95000:
			preds, probas = zip(*self.predict(train_path,as_text=False))
			with io.open(script_dir + os.sep + "multitrain" + os.sep + self.name + '_' + self.corpus,'w',newline="\n") as f:
				sys.stderr.write("o Serializing predictions from partial model\n")
				outlines = [str(preds[i]) + "\t" + str(probas[i]) for i in range(len(probas))]
				outlines = "\n".join(outlines)
				f.write(outlines+"\n")

		pickle_objects = (logmodel, uni_cols)
		pickle.dump(pickle_objects, open(self.model_path, 'wb'))
		sys.stderr.write("o Dumped logistic regression segmenter model ...\n")

		del X_train, colnames, Y_train



	def predict_cached(self,test_data):
		infile = script_dir + os.sep + "multitrain" + os.sep + self.name + '_' + self.corpus
		if os.path.exists(infile):
			pairs = io.open(infile).read().split("\n")
		else:
			sys.stderr.write("o No multitrain file at: " + infile + "\n")
			sys.stderr.write("o Falling back to live prediction for LRSegmenter\n")
			return self.predict(test_data)
		preds = [(int(pr.split()[0]), float(pr.split()[1])) for pr in pairs if "\t" in pr]
		return preds


	# @profile
	def predict(self,test_path,as_text=True,standardization=False,do_tag=False):
		self.readconllmode = "predict"

		if self.verbose:
			sys.stderr.write("o Reading test data...\n")

		sys.stderr.write("o Predicting logistic regression ...\n")

		loaded_model, uni_cols = pickle.load(open(self.model_path, 'rb'))
		self.uni_cols = uni_cols
		X_test, colnames, Y_test, _ = self.read_conll_sentbreak(test_path, neighborwindowsize=self.windowsize,as_text=as_text,cut=False,do_tag=do_tag)

		print("X_test size" ,X_test.shape)

		loaded_model.set_params(**{"n_jobs":3})

		gc.collect()

		probas = loaded_model.predict_proba(X_test)
		probas = [p[1] for p in probas]
		preds = [int(p>0.5) for p in probas]

		del X_test, colnames, Y_test


		# # Verify we are returning as many predictions as we received input tokens
		# print(logmodel.predict(tokens))
		# assert len(tokens) == len(output)
		return zip(preds,probas)


	# @profile
	def read_conll_sentbreak(self, infile, neighborwindowsize=5,as_text=True,cut=True,do_tag=True,multitrain=False):
		global TRAIN_LIMIT
		# read data from conll_reader

		vowels = "AEIOUaeiouéèàáíìúùòóаэыуояеёюи"


		numeric_entries = []
		nonnumeric_entries = []
		goldseg_entries = []
		if as_text:
			conllu_in = infile
		else:
			conllu_in = io.open(infile, 'r', encoding='utf8').read()

		# # Reduce data if too large
		# if not cut and conllu_in.count("\n") > 100000 and multitrain:
		# 	sys.stderr.write("o Data too large; forcing cut and turning off multitraining\n")
		# 	cut = True
		# 	TRAIN_LIMIT = 100000
		# if cut:
		# 	conllu_in = shuffle_cut_conllu(conllu_in,limit=TRAIN_LIMIT)

		train_feats,_,_,_,_ = read_conll(conllu_in,mode="seg",genre_pat=None,as_text=True,cap=None,char_bytes=False)


		featurekeys = ["label", "word", "pos", "cpos", "head", "head_dist", "deprel", "case", "tok_len", "depchunk", "conj", "s_len", "s_type", "sent_doc_percentile", "parentclauses"]

		for lnum, line in enumerate(train_feats):
			lfeatures = [line[x] for x in featurekeys]
			lfeatures[0] = int(lfeatures[0]!="_")


			firstletter = str(lfeatures[1][0].encode("utf8")[0]) if self.lang == "zho" else lfeatures[1][0]
			firstisupper = int (firstletter.upper() == firstletter)
			firstisconsonant = len(re.findall('[^'+vowels+']', firstletter))
			firstisvowel = len(re.findall('['+vowels+']', firstletter))
			firstisdigit = len(re.findall('[0-9]', firstletter))
			firstisspecial = len(re.findall('[^A-Za-z0-9]', firstletter))

			lastletter = str(lfeatures[1][-1].encode("utf8")[-1]) if self.lang == "zho" else lfeatures[1][-1]
			lastisupper = int(lastletter.upper() == lastletter)
			lastisconsonant = len(re.findall('[^'+vowels+']', lastletter))
			lastisvowel = len(re.findall('['+vowels+']', lastletter))
			lastisdigit = len(re.findall('[0-9]', lastletter))
			lastisspecial = len(re.findall('[^A-Za-z0-9]', lastletter))

			numconsonants = len(re.findall('[^'+vowels+']', lfeatures[1]))
			numvowels = len(re.findall('['+vowels+']', lfeatures[1]))
			numdigits = len(re.findall('[0-9]', lfeatures[1]))
			numspecials = len(re.findall('[^A-Za-z0-9]', lfeatures[1]))


			numeric_entries.append(
				[
				 numconsonants, numvowels, numdigits, numspecials,
				  firstisupper, firstisconsonant, firstisvowel, firstisdigit, firstisspecial,
				  lastisupper, lastisconsonant, lastisspecial,
					lfeatures[4], lfeatures[5], lfeatures[8], lfeatures[11], lfeatures[13]
				])

			nonnumeric_entries.append([lfeatures[2],lfeatures[3], firstletter,lastletter,
									   lfeatures[6], lfeatures[7], lfeatures[9], lfeatures[10], lfeatures[12], re.sub(r'^([^\|]*\|[^\|]*)\|.*', r'\1', lfeatures[14])
									   ])

			goldseg_entries.append(lfeatures[0])


		# featurekeys = ["label", "word", "pos", "cpos", "head", "head_dist", "deprel", "case", "tok_len", 9 "depchunk", "conj", 11"s_len", "s_type", "sent_doc_percentile", "parentclauses"]


		numeric_colnames = ['numconsonants', 'numvowels', 'numdigits', 'numspecials',
								    'firstisupper', 'firstisconsonant', 'firstisvowel', 'firstisdigit','firstisspecial',
								    'lastisupper', 'lastisconsonant','lastisspecial',
						   featurekeys[4], featurekeys[5], featurekeys[8], featurekeys[11], featurekeys[13]
							]
		nonnumeric_colnames = ['gold_pos', 'gold_cpos','firstletter', 'lastletter',
							  featurekeys[6], featurekeys[7], featurekeys[9], featurekeys[10], featurekeys[12], featurekeys[14]
							   ]
		numeric_entries = np.array(numeric_entries, dtype=np.float32)
		nonnumeric_entries = np.array(nonnumeric_entries)


		# Dummy multi vectors cattodummy is much better
		unigram_entries, unigram_colnames = self.cattodummy(nonnumeric_entries, nonnumeric_colnames, numeric_entries, numeric_colnames)


		sys.stderr.write("o unigram dataframe ready\n")


		if neighborwindowsize >= 3:
			sys.stderr.write("o duplicating to %d-gram...\n" %neighborwindowsize)

			prev_entries, prev_colnames = self.copyforprevnext(unigram_entries, unigram_colnames, 'prev')
			next_entries, next_colnames = self.copyforprevnext(unigram_entries, unigram_colnames, 'next')
			ngram_entries, ngram_colnames = self.numpyconcat2df(prev_entries, prev_colnames, unigram_entries, unigram_colnames)
			ngram_entries, ngram_colnames = self.numpyconcat2df(ngram_entries,ngram_colnames, next_entries, next_colnames)
			del prev_entries, prev_colnames, next_colnames, next_entries

			if neighborwindowsize >=5:
				prevprev_entries, prevprev_colnames = self.copyforprevnext(unigram_entries, unigram_colnames, 'prevprev')
				nextnext_entries, nextnext_colnames = self.copyforprevnext(unigram_entries, unigram_colnames, 'nextnext')
				ngram_entries, ngram_colnames = self.numpyconcat2df(prevprev_entries, prevprev_colnames, ngram_entries,
																   ngram_colnames)
				ngram_entries, ngram_colnames  = self.numpyconcat2df(ngram_entries, ngram_colnames, nextnext_entries,
																   nextnext_colnames)
				del prevprev_entries, prevprev_colnames, nextnext_colnames, nextnext_entries

				if neighborwindowsize == 7:
					prevprevprev_entries, prevprevprev_colnames = self.copyforprevnext(unigram_entries, unigram_colnames, 'prevprevprev')
					nextnextnext_entries, nextnextnext_colnames = self.copyforprevnext(unigram_entries, unigram_colnames, 'nextnextnext')
					ngram_entries, ngram_colnames = self.numpyconcat2df(prevprevprev_entries, prevprevprev_colnames, ngram_entries,
																   ngram_colnames)
					ngram_entries, ngram_colnames = self.numpyconcat2df(ngram_entries, ngram_colnames, nextnextnext_entries,
																   nextnextnext_colnames)
					del prevprevprev_entries, prevprevprev_colnames, nextnextnext_colnames, nextnextnext_entries

		else:
			ngram_entries, ngram_colnames = unigram_entries, unigram_colnames

		del unigram_entries, numeric_colnames, numeric_entries

		return ngram_entries, ngram_colnames, goldseg_entries, unigram_colnames


if __name__ == "__main__":

	# Argument parser
	parser = argparse.ArgumentParser(description='Input parameters')
	parser.add_argument('--corpus', '-c', action='store', dest='corpus', default="spa.rst.sctb", help='corpus name')
	parser.add_argument('--mode', '-m', action='store', dest='mode', default="train", choices=["train","predict"],help='Please specify train or predict mode')
	parser.add_argument('--windowsize', '-w', action='store', dest='windowsize', default=3, type=int, choices=[1,3,5,7], help='Please specify windowsize which has to be an odd number, i.e. 1, 3, 5, 7 (defaulted to 5).')
	parser.add_argument('--limit', '-l', action='store', default=5000, type=int, help='Subset size of training data to use')
	parser.add_argument("-d","--data_dir",default=os.path.normpath('../../../data'),help="Path to shared task data folder")
	parser.add_argument("-v","--verbose",action="store_true",help="Output verbose messages")
	parser.add_argument('--standardization', '-s', action='store_true', dest='standardization', help='whether to standardize features')
	parser.add_argument('--multitrain', action='store_true', help='whether to perform multitraining')

	args = parser.parse_args()

	start_time = time.time()

	data_folder = args.data_dir
	if data_folder is None:
		data_folder = os.path.normpath(r'./../../../sharedtask2019/data/')
	corpora = args.corpus

	if corpora == "all":
		corpora = os.listdir(data_folder)
		corpora = [c for c in corpora if os.path.isdir(os.path.join(data_folder, c))]
	else:
		corpora = [corpora]

	# Set a global limit to training size document in lines
	# We set a very conservative, small training size to avoid over-reliance of the ensemble, which also
	# uses the same training data to assess the usefulness of this estimator
	TRAIN_LIMIT = args.limit

	for corpusname in corpora:
		if "." in corpusname:
			lang = corpusname.split(".")[0]
		else:
			lang = "eng"

		# Run test
		segmenter = LRSegmenter(lang=lang,model=corpusname,windowsize=args.windowsize)
		if args.verbose:
			segmenter.verbose = True

		if segmenter.verbose:
			sys.stderr.write("o Processing corpus "+corpusname+"\n")


		if args.mode == "train":
			# When running from CLI, we always train (predict mode is done on imported class)
			segmenter.train(data_folder + os.sep+ corpusname + os.sep + corpusname + "_train.conll",as_text=False,standardization=args.standardization,multitrain=args.multitrain)

		# Now evaluate model
		predictions, probas = zip(*segmenter.predict(data_folder + os.sep+ corpusname + os.sep +corpusname + "_dev.conll",
													 as_text=False,standardization=args.standardization,do_tag=True))

		# Get gold labels for comparison
		conllu_in = io.open(data_folder + os.sep+ corpusname + os.sep +corpusname + "_dev.conll", 'r', encoding='utf8').read()
		devfeats,_,_,_,_ = read_conll(conllu_in, mode="seg", genre_pat=None, as_text=True, cap=None, char_bytes=False)
		labels = [int(x["label"]!='_') for x in devfeats]

		# give dev F1 score
		from sklearn.metrics import classification_report, confusion_matrix
		print(classification_report(labels, predictions, digits=6))
		print(confusion_matrix(labels, predictions))

		elapsed = time.time() - start_time
		sys.stderr.write(str(timedelta(seconds=elapsed)) + "\n\n")

