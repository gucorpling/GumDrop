from keras.models import load_model
from gensim.models import KeyedVectors
import io, os, sys, re, requests, time, argparse, gc, wget
from datetime import timedelta
from collections import defaultdict, OrderedDict
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "..")
sent = os.path.abspath(script_dir + os.sep + ".." + os.sep + "sentencers")
sys.path.append(lib)
sys.path.append(sent)
from dnn_sent_wrapper import loadvec, n_gram, unique_embed
from conll_reader import get_stype, get_case
# from depfeatures import DepFeatures

model_dir = os.path.abspath(script_dir + os.sep + ".." + os.sep + ".." + os.sep + "models" + os.sep)
vec_dir = os.path.abspath(script_dir + os.sep + ".." + os.sep + ".." + os.sep + "vec" + os.sep)


# categorical feature to multiple one-hot vectors
def cattodummy(cat_vars, data):
	for var in cat_vars:
		cat_list = pd.get_dummies(data[var], prefix=var, drop_first=True)
		data1 = data.join(cat_list)
		data = data1
	data.drop(cat_vars, axis=1, inplace=True)
	return data


def read_conll(features,infile,mode="seg",genre_pat=None,as_text=False,cap=None,char_bytes=False):
	if as_text:
		lines = infile.split("\n")
	else:
		lines = io.open(infile,encoding="utf8").readlines()
	docname = infile if len(infile) < 100 else "doc1"
	output = []  # List to hold dicts of each observation's features
	cache = []  # List to hold current sentence tokens before adding complete sentence features for output
	toks = []  # Plain list of token forms
	firsts = set([])  # Attested first characters of words
	lasts = set([])  # Attested last characters of words
	vocab = defaultdict(int)  # Attested token vocabulary counts
	sent_start = True
	tok_id = 0  # Track token ID within document
	sent_id = 1
	genre = "_"
	open_quotes = set(['"','«','``','”'])
	close_quotes = set(['"','»','“',"''"])
	open_brackets = set(["(","[","{","<"])
	close_brackets = set([")","]","}",">"])
	used_feats = ["VerbForm","PronType","Person","Mood"]
	in_quotes = 0
	in_brackets = 0
	last_quote = 0
	last_bracket = 0
	total = 0
	total_sents = 0
	doc_sents = 1
	heading_first = "_"
	heading_last = "_"
	for r, line in enumerate(lines):
		if "\t" in line:
			fields = line.split("\t")
			if "-" in fields[0]:  # conllu super-token
				continue
			total +=1
			word, lemma, pos, cpos, feats, head, deprel = fields[1:-2]
			if mode=="seg":
				if "BeginSeg=Yes" in fields[-1]:
					label = 1
				else:
					label = 0
			elif mode == "sent":
				if sent_start:
					label = 1
				else:
					label = 0
			else:
				raise ValueError("read_conll mode must be one of: seg|sent\n")
			# Compose a categorical feature from morphological features of interest
			feats = [f for f in feats.split("|") if "=" in f]
			feat_string = ""
			for feat in feats:
				name, val = feat.split("=")
				if name in used_feats:
					feat_string += val
			if feat_string == "":
				feat_string = "_"
			vocab[word] += 1
			case = get_case(word)
			head_dist = int(fields[0]) - int(head)
			if len(word.strip()) == 0:
				raise ValueError("! Zero length word at line " + str(r) + "\n")
			toks.append(word)
			first_char = word[0]
			last_char = word[-1]
			if char_bytes:
				try:
					first_char = str(first_char.encode("utf8")[0])
					last_char = str(last_char.encode("utf8")[-1])
				except:
					pass
			firsts.add(first_char)
			lasts.add(last_char)

			tent_dict = {"word":word, "lemma":lemma, "pos":pos, "cpos":cpos, "head":head, "head_dist":head_dist, "deprel":deprel,
						   "docname":docname, "case":case,"tok_len":len(word),"label":label,"first":first_char,"last":last_char,
						   "tok_id": tok_id,"genre":genre,"wid":int(fields[0]),"quote":in_quotes,"bracket":in_brackets,"morph":feat_string,
						   "heading_first": heading_first, "heading_last": heading_last,"depchunk":"_","conj":"_"}

			if len(features) <= 1:
				cache.append(tent_dict)
			else:
				cache.append({k:tent_dict[k] for k in features})

			if mode == "seg":
				cache[-1]["s_num"] = doc_sents

			tok_id += 1
			sent_start = False
			if word in open_quotes:
				in_quotes = 1
				last_quote = tok_id
			elif word in close_quotes:
				in_quotes = 0
			if word in open_brackets:
				in_brackets = 1
				last_bracket = tok_id
			elif word in close_brackets:
				in_brackets = 0
			if tok_id - last_quote > 100:
				in_quotes = 0
			if tok_id - last_bracket > 100:
				in_brackets = 0

		elif "# newdoc id = " in line:
			if cap is not None:
				if total > cap:
					break
			docname = re.search(r"# newdoc id = (.+)",line).group(1)
			if genre_pat is not None:
				genre = re.search(genre_pat,docname).group(1)
			else:
				genre = "_"
			doc_sents =1
			tok_id = 1
		elif len(line.strip())==0:
			sent_start = True
			if len(cache)>0:
				if mode == "seg":  # Don't add s_len in sentencer learning mode
					sent = " ".join([t["word"] for t in cache])
					if sent[0] == sent[0].upper() and len(cache) < 6 and sent[-1] not in [".","?","!",";","！","？","。"]:
						# Uppercase short sentence not ending in punctuation - possible heading affecting subsequent data
						heading_first = str(sent.encode("utf8")[0]) if char_bytes else sent[0]
						heading_last = str(sent.encode("utf8")[-1]) if char_bytes else sent[-1]
					# Get s_type features
					s_type = get_stype(cache)
					for tok in cache:
						tok["s_len"] = len(cache)
						tok["s_id"] = sent_id
						tok["heading_first"] = heading_first
						tok["heading_last"] = heading_last
						tok["s_type"] = s_type
					sent_id +=1
					doc_sents += 1
					total_sents += 1
				output += cache
				if mode == "seg":
					if len(output) > 0:
						for t in output[-int(fields[0]):]:
							# Add sentence percentile of document length in sentences
							t["sent_doc_percentile"] = t["s_num"]/doc_sents
				cache = []

	# Flush last sentence if no final newline
	if len(cache)>0:
		if mode == "seg":  # Don't add s_len in sentencer learning mode
			sent = " ".join([t["word"] for t in cache])
			if sent[0] == sent[0].upper() and len(cache) < 6 and sent[-1] not in [".","?","!",";","！","？","。"]:
				# Uppercase short sentence not ending in punctuation - possible heading
				heading_first = str(sent.encode("utf8")[0]) if char_bytes else sent[0]
				heading_last = str(sent.encode("utf8")[-1]) if char_bytes else sent[-1]
			# Get s_type features
			s_type = get_stype(cache)
			for tok in cache:
				tok["s_len"] = len(cache)
				tok["s_id"] = sent_id
				tok["heading_first"] = heading_first
				tok["heading_last"] = heading_last
				tok["s_type"] = s_type

		output += cache
		if mode == "seg":
			for t in output[-int(fields[0]):]:
				# Add sentence percentile of document length in sentences
				t["sent_doc_percentile"] = 1.0

	# df = DepFeatures()
	# output = df.extract_depfeatures(output)
	train_feats = extract_depfeatures(output, vocab, toks, firsts, lasts)

	feats_tups = [tuple([x['s_id'],x['wid'],x['pos'],int(x['head']),x['deprel'],x['word']]) for x in train_feats]
	num_s = sorted(list(set([x[0] for x in feats_tups])))
	heads_list = []
	grand_heads_list = []
	first_tok_list = []
	first_tok_pos_list = []
	# root_list = []
	head_pos_list = []
	for s in num_s:
		feats_s = sorted([x for x in feats_tups if x[0]==s], key=lambda x: x[1])
		sent = [tup[-1] for tup in feats_s]
		first_tok_list += [sent[0]]*len(sent)
		first_tok_pos_list += [feats_s[0][2]]*len(sent)
		heads, grand_heads, head_pos = find_dep_feats(sent, feats_s)
		heads_list += heads
		grand_heads_list += grand_heads
		# root_list += ROOTs
		head_pos_list += head_pos

	for n in range(len(train_feats)):
		train_feats[n]['head'] = heads_list[n]
		train_feats[n]['grand_head'] = grand_heads_list[n]
		train_feats[n]['first_tok'] = first_tok_list[n]
		# train_feats[n]['root'] = root_list[n]
		train_feats[n]['first_tok_pos'] = first_tok_pos_list[n]
		train_feats[n]['head_pos'] = head_pos_list[n]
	return train_feats


def find_dep_feats(sent, feats):
	sent_feats = {(x[0], x[1][1]):x[1] for x in zip(sent, feats)}
	sent_dict = {}
	for tups in feats:
		sent_dict[tups[1]] = tups[-1]
	head_list = []
	head_pos_list = []
	head_index = []
	ROOT = ""
	for count,tok in enumerate(sent):
		head_card = sent_feats[(tok, count+1)][3]
		head_pos_list.append(sent_feats[(tok, count+1)][2])
		if head_card != 0:
			head = sent_dict[head_card]
		else:
			ROOT = sent_feats[(tok, count+1)][-1]
			head = ROOT
		head_list.append(head)
		head_index.append((head, head_card))
	grand_head_list = []
	grand_head_pos_list = []
	for tok, count in head_index:
		if count == 0:
			grand_head = ROOT
		else:
			head_card = sent_feats[(tok, count)][3]
			grand_head = sent_dict[head_card] if head_card != 0 else sent_feats[(tok, count)][-1]
		grand_head_list.append(grand_head)
	return head_list, grand_head_list, head_pos_list#, [ROOT] * len(sent)


def extract_depfeatures(train_feats, vocab, toks, firsts, lasts):
	clausal_deprels = ['csubj', 'ccomp', 'xcomp', 'advcl', 'acl', 'acl:relcl', 'list', 'parataxis', 'appos', 'conj']

	# train_feats, vocab, toks, firsts, lasts = conll_reader.read_conll(extractfile)
	# <class 'dict'>: {'word': 'creada', 'lemma': 'creado', 'pos': 'VERB', 'cpos': '_', 'head': '17', 'head_dist': 2, 'deprel': 'acl', 'docname': 'BMCS_ESP1-GS', 'case': 'l', 'tok_len': 6, 'label': '_', 'first': 'c', 'last': 'a', 'tok_id': 92, 'genre': '_', 'wid': 19, 'quote': 0, 'morph': 'Part', 'heading_first': '_', 'heading_last': '_', 's_num': 2, 's_len': 47, 's_id': 2, 'sent_doc_percentile': 0.6666666666666666}
	# 'word': 'creada' , 'head': '17',  'wid': 19, 'head_dist': 2, 'deprel': 'acl','docname': 'BMCS_ESP1-GS', 's_len': 47, 's_id': 2, 'tok_id': 92, ('s_num': 2, )
	# [s_id, wid, tok_id ,head, deprel, word]
	feats_tups = [tuple([x['s_id'],x['wid'],x['tok_id'],int(x['head']),x['deprel'],x['word']]) for x in train_feats]
	num_s = sorted(list(set([x[0] for x in feats_tups])))
	deprel_s = sorted(list(set([x[4] for x in feats_tups])))
	# print(corpus,'\t' ,deprel_s)

	# looping through sentences
	for s in num_s:
		feats_s = sorted([x for x in feats_tups if x[0]==s], key=lambda x: x[1])
		feats_parents = defaultdict(list)

		sent = [tup[-1] for tup in feats_s]

		# looping through tokens in a sentence
		wid2tokid = {}
		for t in feats_s:

			# finding all (grand)parents (grand-heads)
			wid = t[1]
			head = t[3]
			wid2tokid[wid]=t[2]
			while head != 0:
				head_t = [x for x in feats_s if x[1]==head][0]
				feats_parents[t].append(head_t)
				head = head_t[3]

		# loop through clausal_deprels (non-conj & conj) and create BIO list for sentence tokens
		dr_d = defaultdict(list)

		## finding all tokens in a sentence who or whose parents has a deprel (dr) -- non-conj
		for t in feats_s:
			t_gen = [t] + feats_parents[t]

			# all including conj
			for dr in clausal_deprels:
				in_t_gen = [x for x in t_gen if x[4].startswith(dr)]
				if in_t_gen!=[]:
					dr_d[(in_t_gen[0][1], in_t_gen[0][4])].append(t)


		# sort dictionary values
		dr_dl = defaultdict(list)
		for k,v in dr_d.items():
			if v!=[]:
				dr_dl[k+(len(v),)] = sorted(list(set([x[1] for x in v])))
				# sorted(v, key=lambda x: x[1])

		# collect all BIEO features, for conj and non-conj separately
		feats_l = [[] for x in range(len(feats_s))]
		feats_conjl = [[] for x in range(len(feats_s))]
		for i in range(len(feats_s)):
			for k,v in dr_dl.items():
				# for non-conj
				if not k[1].startswith('conj'):
					if not i+1 in v:
						feats_l[i].append('_')
					elif v[0]==i+1:
						feats_l[i].append(('B'+ k[1], v[0], v[-1]))
					elif v[-1]==i+1:
						feats_l[i].append(('E'+ k[1], v[0], v[-1]))
					else:
						feats_l[i].append(('I'+ k[1], v[0], v[-1]))

				# for conj
				else:
					if not i+1 in v:
						feats_conjl[i].append('_')
					elif v[0]==i+1:
						feats_conjl[i].append(('B'+ k[1], v[0], v[-1]))
					elif v[-1]==i+1:
						feats_conjl[i].append(('E'+ k[1], v[0], v[-1]))
					else:
						feats_conjl[i].append(('I'+ k[1], v[0], v[-1]))


		# Prioritize Bsmall > Blarge > Elarge > Esmall > Ismall > Ilarge > _
		# non-conj
		for id_l, l in enumerate(feats_l):
			Bsub = sorted([x for x in l if x[0].startswith('B')], key=lambda x: x[2]-x[1])
			Esub = sorted([x for x in l if x[0].startswith('E')], key=lambda x: x[2]-x[1], reverse=True)
			Isub = sorted([x for x in l if x[0].startswith('I')], key=lambda x: x[2]-x[1])
			if Bsub!=[]:
				feats_l[id_l]=Bsub[0][0]
			elif Esub!=[]:
				feats_l[id_l] = Esub[0][0]
			elif Isub!=[]:
				feats_l[id_l] = Isub[0][0]
			else:
				feats_l[id_l] = '_'

		# remove sub-deprel after :, e.g. csubj:pass -> csubj (however, acl:relcl stays as acl:relcl)
		feats_l = [re.sub(r':[^r].*$', '', x) for x in feats_l]

		# add non-conj to train_feats
		for id_l, l in enumerate(feats_l):
			train_feats[wid2tokid[id_l+1]-1]['deprel'] = l


		# conj
		for id_l, l in enumerate(feats_conjl):
			Bsub = sorted([x for x in l if x[0].startswith('B')], key=lambda x: x[2]-x[1])
			Esub = sorted([x for x in l if x[0].startswith('E')], key=lambda x: x[2]-x[1], reverse=True)
			Isub = sorted([x for x in l if x[0].startswith('I')], key=lambda x: x[2]-x[1])
			if Bsub!=[]:
				feats_conjl[id_l]=Bsub[0][0]
			elif Esub!=[]:
				feats_conjl[id_l] = Esub[0][0]
			elif Isub!=[]:
				feats_conjl[id_l] = Isub[0][0]
			else:
				feats_conjl[id_l] = '_'

		# add conj to train_feats
		for id_l, l in enumerate(feats_conjl):
			train_feats[wid2tokid[id_l+1]-1]['conj'] = l

		sys.stderr.write('\r Adding deprel BIEO features to train_feats %s ### o Sentence %d' %(corpus, s))

	return train_feats


def mergeWord2Vec(word_grams, uni_embed, categorical_cols):
	fec_vec = []
	count = 0
	for words in word_grams:
		word_vec = []
		for word in words:
			word_vec += uni_embed[word]
		word_vec += categorical_cols[count]
		fec_vec.append(word_vec)
		count += 1
	return np.asarray(fec_vec, dtype=np.float32)


def get_best_params(corpus, model_name):
	infile = script_dir + os.sep + "params" + os.sep + model_name + "_best_params.tab"
	if not os.path.isfile(infile):
		space = {
				'num_layers': 'two',
				'units2': 64,
				'dropout2': .30,
				'units1': 96,
				'dropout1': .40,
				'epoch': 10,
				'batch_size': 128,
				'optimizer': 'adam',
				'activation': 'relu',
				'gram': 7
				}
	else:
		lines = io.open(infile).readlines()
		params = ['num_layers', 'units2', 'dropout2', 'units1', 'dropout1', 'epoch', 'batch_size', 'optimizer', 'gram']
		str_feats = ['one', 'two', 'adadelta', 'adam', 'rmsprop', 'sgd']
		space = {}

		for line in lines:
			if "\t" in line:
				corp, clf_name, param, val = line.split("\t")
				val = val.strip()
				if corp == corpus:
					if param in params:
						if val in str_feats:
							space[param] = val
						elif "0." in val:
							space[param] = float(val)
						else:
							space[param] = int(val)

	return space


class DNNSegmenter:

	def __init__(self,lang="eng",model="eng.rst.gum"):
		self.lang = lang
		self.name = "DNNSegmenter"
		self.corpus = model
		self.model_path = model_dir + os.sep + model + "_dnn_seg" + ".hd5"
		if lang == "zho":
			vec = "cc.zho.300.vec_trim.vec"
		elif lang == "eng":
			vec = "glove.6B.300d_trim.vec"
		else:
			vec = "wiki.**lang**.vec_trim.vec".replace("**lang**",lang)
		self.vec_path = vec_dir + os.sep + vec
		self.space = get_best_params(self.corpus, self.name)


	def process_data(self,path,as_text=False):
		MB = 1024*1024
		word_embed = loadvec(self.vec_path)
		features = []
		output = read_conll(features, path, mode="seg", genre_pat=None, as_text=as_text)
		uni_embed = unique_embed(output, word_embed)
		sys.stdout.write('\n##### Loaded dataset word embeddings.\n')
		n_gram_cols, label = n_gram(output, self.space['gram'])


		dep_f = ['first_tok', 'head', 'grand_head']
		dep_feats = [[word_dict[f] for f in dep_f] for word_dict in output]
		categorical_f = ['pos', 'head_pos']
		categorical_feats = pd.DataFrame([[word_dict[f] for f in categorical_f] for word_dict in output], dtype=np.str, columns=categorical_f)
		categorical_feats = cattodummy(categorical_f, categorical_feats)
		# cateogorical_cols = len(categorical_feats.columns())
		categorical_feats = categorical_feats.to_sparse().values

		feats_cols = [n_gram_cols[n] + dep_feats[n] for n in range(len(n_gram_cols))]
		fc_vec = mergeWord2Vec(feats_cols, uni_embed, categorical_feats.tolist())
		# train_fc = np.vstack((fc_vec, categorical_feats))
		sys.stdout.write("fc_vec %d MB\n" % (sys.getsizeof(fc_vec)/MB))

		X = fc_vec
		y = label
		self.cols = fc_vec.shape[1]
		return X, y


	def get_multitrain_preds(self, X, y, multifolds):
		all_preds = []
		all_probas = []
		X_folds = np.array_split(X, multifolds)
		y_folds = np.array_split(y, multifolds)
		for i in range(multifolds):
			model = self.keras_model()
			X_train = np.vstack(tuple([X_folds[j] for j in range(multifolds) if j != i]))
			y_train = np.concatenate(tuple([y_folds[j] for j in range(multifolds) if j != i]))
			X_heldout = X_folds[i]
			sys.stdout.write("##### Training on fold " + str(i + 1) + " of " + str(multifolds) + "\n")
			model.fit(X_train, y_train, epochs=self.space['epoch'], batch_size=self.space['batch_size'], verbose=2)
			probas = model.predict(X_heldout)
			preds = [str(int(p > 0.5)) for p in probas]
			probas = [str(p[0]) for p in probas]
			all_preds += preds
			all_probas += probas

		pairs = list(zip(all_preds, all_probas))
		pairs = ["\t".join(pair) for pair in pairs]

		return "\n".join(pairs)


	def keras_model(self):
		from keras.models import Sequential
		from keras.layers import Dense, Dropout, Activation
		from keras import metrics

		np.random.seed(11)

		# create model
		model = Sequential()
		model.add(Dense(self.space['units1'], input_dim=self.cols))
		model.add(Activation('relu'))
		model.add(Dropout(self.space['dropout1']))
		if self.space['num_layers'] == 'two':
			model.add(Dense(self.space['units2']))
			model.add(Activation('relu'))
			model.add(Dropout(self.space['dropout2']))
		model.add(Dense(1))
		model.add(Activation('sigmoid'))
		# Compile model
		model.compile(loss='binary_crossentropy',
					  optimizer=self.space['optimizer'],
					  metrics=['accuracy'])
		return model


	def train(self, x_train, y_train, multitrain=False):
		model = self.keras_model()

		if multitrain:
			multitrain_preds = self.get_multitrain_preds(x_train, y_train, 5)
			multitrain_preds = "\n".join(multitrain_preds.strip().split("\n"))
			with io.open(script_dir + os.sep + "multitrain" + os.sep + self.name + '_' + self.corpus, 'w',
						 newline="\n") as f:
				sys.stdout.write("##### Serializing multitraining predictions\n")
				f.write(multitrain_preds)

		# Fit the model
		model.fit(x_train, y_train, epochs=self.space['epoch'], batch_size=self.space['batch_size'], verbose=2)
		model.save(self.model_path)


	def predict(self,test_path,as_text=True):
		# predict the model
		model = load_model(self.model_path)
		X_test, y_test = self.process_data(test_path,as_text=as_text)

		probas = model.predict(X_test)
		preds = [int(p > 0.5) for p in probas]
		probas = [p[0] for p in probas]
		preds[0] = 1
		probas[0] = 0.6

		# give dev F1 score
		if not as_text:
			print(classification_report(y_test, preds, digits=6))
			print(confusion_matrix(y_test, preds))

		return zip(preds, probas)


	def predict_cached(self,test_data):
		infile = script_dir + os.sep + "multitrain" + os.sep + self.name + '_' + self.corpus
		if os.path.exists(infile):
			pairs = io.open(infile).read().split("\n")
		else:
			sys.stdout.write("##### No multitrain file at: " + infile + "\n")
			sys.stdout.write("##### Falling back to live prediction for DNNSentencer\n")
			return self.predict(test_data)
		preds = [(int(pr.split()[0]), float(pr.split()[1])) for pr in pairs if "\t" in pr]
		return preds


if __name__ == "__main__":

	p = argparse.ArgumentParser(description='Input parameters')
	p.add_argument("-c","--corpus",default="spa.rst.sctb",help="Corpus to train on. Not used for PDTB corpus.")
	p.add_argument("-d","--data_dir",default=os.path.normpath("../../../data"),help="Path to shared task data folder")
	# p.add_argument('-w', '--windowsize', action='store', dest='windowsize', default=5, type=int, choices=[3,5,7], help='Please specify windowsize which has to be an odd number, i.e. 1, 3, 5, 7 (defaulted to 5).')
	p.add_argument("--mode",action="store",default="train-test",choices=["train-test","test"])
	p.add_argument('--multitrain', action='store_true', help='whether to perform multitraining')

	args = p.parse_args()

	start_time = time.time()

	corpora = args.corpus

	data_dir = args.data_dir

	if corpora == "all":
		corpora = os.listdir(data_dir )
		corpora = [c for c in corpora if (os.path.isdir(os.path.join(data_dir, c)) and "pdtb" not in c)]  # No PDTB
	else:
		if "pdtb" in corpora:
			sys.exit("Try a corpora not in PDTB format.")
		corpora = [corpora]

	for corpus in corpora:
		if "." in corpus:
			lang = corpus.split(".")[0]
		else:
			lang = "eng"

		corpus_start_time = time.time()

		train_path = data_dir + os.sep + corpus + os.sep + corpus + "_train.conll"
		dev_path = data_dir + os.sep + corpus + os.sep + corpus + "_dev.conll"
		test_path = data_dir + os.sep + corpus + os.sep + corpus + "_test.conll"

		segmenter = DNNSegmenter(lang=lang,model=corpus)

		if "train" in args.mode:
			sys.stderr.write("\no Training on corpus " + corpus + "\n")
			X_train, y_train = segmenter.process_data(train_path)
			segmenter.train(X_train,y_train,multitrain=args.multitrain)

		# Now evaluate model
		predictions, probas = zip(*segmenter.predict(dev_path, as_text=False))
		elapsed = time.time() - start_time
		sys.stdout.write(str(timedelta(seconds=elapsed)) + "\n\n")
