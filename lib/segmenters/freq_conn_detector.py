import os, pickle, sys, io
from collections import Counter, defaultdict
from argparse import ArgumentParser

script_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.abspath(script_dir + os.sep + ".." + os.sep + ".." + os.sep + "models")

lib = os.path.abspath(script_dir + os.sep + "..")
sys.path.append(lib)
from conll_reader import read_conll
from seg_eval import get_scores

class FreqConnDetector:

	def __init__(self,lang="zho",model="zho.pdtb.cdtb"):
		self.name = "FreqConnDetector"
		self.lang = lang
		self.corpus = model
		self.model_path = model_dir + os.sep + self.corpus + "_conn.pkl"
		self.train_path = model_dir + os.sep + ".." + os.sep + ".." + os.sep + "data" + os.sep + self.corpus + os.sep + self.corpus + "_train.tok"
		self.dev_path = model_dir + os.sep + ".." + os.sep + ".." + os.sep + "data" + os.sep + self.corpus + os.sep + self.corpus + "_dev.tok"

	def train(self, trainfile):

		with open(trainfile, 'r', encoding='utf-8') as f:
			lines = [line.strip().split() for line in f.readlines() if "\t" in line]

		Conn = Counter()
		token_counts = Counter()
		single_token_counts = Counter()

		connectives = defaultdict(float)
		removed = set([])
		tokens = []
		labels = []
		ngram = 5

		for item in lines:
			if len(item) != 0:
				tokens.append(item[1])
				labels.append(item[-1])

		for n in list(range(ngram))[::-1]:
			for i in range(len(tokens)-n):
				if any([j in removed for j in list(range(i, i+n+1))]):
					continue
				toks = tokens[i:i+n+1]
				labs = labels[i:i+n+1]
				token_counts[tuple(toks)] += 1
				if any([l == "_" for l in labs]):
					continue
				if not labs[0] == "Seg=B-Conn":
					continue
				if len(labs) > 1:
					if any(lab != "Seg=I-Conn" for lab in labs[1:]):
						continue
				for k in range(i,i+n+1):
					removed.add(k)
				Conn[tuple(toks)] += 1

		for tokens in Conn.keys():
			ratio = Conn[tokens] / token_counts[tokens]
			connectives[tokens] = (Conn[tokens],round(ratio, 3))

		with open(self.model_path, 'wb') as fp:
			pickle.dump(connectives, fp)

		for token in tokens:
			single_token_counts[token] += 1

		return connectives

	def predict(self, lines):

		with open(self.model_path, 'rb') as fp:
			connectives = pickle.load(fp)

		tokens = []
		removed = set([])
		ngram = 5

		lines = lines.split("\n")

		for line in lines:
			if "\t" in line:
				line = line.strip().split("\t")
				if len(line) != 0:
					if "-" not in line[0]:
						tokens.append(line[1])

		preds = ["_"]*len(tokens)
		ratios = ["0.0"]*len(tokens)
		freqs = ["0.0"]*len(tokens)
		thresh = 0.5

		for n in list(range(ngram))[::-1]:
			for i in range(len(tokens)-n):
				if any([j in removed for j in list(range(i, i+n+1))]):
					continue
				toks = tuple(tokens[i:i+n+1])
				if toks in connectives:
					freq, ratio = connectives[toks]
					for j in range(i, i+n+1):
						if ratio > thresh:
							ratios[j] = str(ratio)
							freqs[j] = str(freq)
							if j == i:
								preds[j] = "Seg=B-Conn"
							else:
								preds[j] = "Seg=I-Conn"

							removed.add(j)

		# output predictions
		i = 0
		output = []
		# with open(self.path, 'r', encoding='utf-8') as f:
		# 	lines = f.readlines()
		for line in lines:
			if "\t" in line:
				fields = line.split("\t")
				if "-" in fields[0]:
					output.append(line.strip())
					continue
				fields[-1] = preds[i]
				output.append("\t".join(fields))
				i += 1
			else:
				output.append(line.strip())

		return list(zip(tokens, preds, ratios, freqs))


if __name__ == "__main__":

	p = ArgumentParser()
	p.add_argument("-c","--corpus",default="zho.pdtb.cdtb",help="Corpus to use or 'all'")
	p.add_argument("-m", "--mode", default="train", choices=["train", "predict"], help="Specify train or predict mode")
	p.add_argument("-d","--data_dir",default="../../../data",help="Path to shared task data folder")
	p.add_argument("--eval_test",action="store_true",help="Evaluate on test, not dev")
	opts = p.parse_args()

	data_dir = opts.data_dir
	corpus = opts.corpus
	lang = corpus.split('.')[0]

	train_path = model_dir + os.sep + ".." + os.sep + ".." + os.sep + "data" + os.sep + corpus + os.sep + corpus + "_train.conll"
	dev_path = model_dir + os.sep + ".." + os.sep + ".." + os.sep + "data" + os.sep + corpus + os.sep + corpus + "_dev.conll"
	test_path = model_dir + os.sep + ".." + os.sep + ".." + os.sep + "data" + os.sep + corpus + os.sep + corpus + "_test.conll"

	connective = FreqConnDetector(lang=lang,model=corpus)

	if opts.mode == "train":
		# Get Connectives: Raw frequency and ratio of the connectives found in the training data
		conn = connective.train(train_path)

	if opts.eval_test:
		dev_path = test_path

	with open(dev_path, 'r', encoding='utf-8') as f:
		lines = f.read()

	# Prediction on the devset
	pred_labels_probs = connective.predict(lines)
	resps = [tok[1] if float(tok[2])>0.5 else "_" for tok in pred_labels_probs]
	train_feats, _, _, _, _ = read_conll(dev_path,as_text=False)
	gold = io.open(dev_path,encoding="utf8").read()
	lines = gold.split("\n")
	processed = []
	i = 0
	for line in lines:
		if "\t" in line:
			fields = line.split('\t')
			if "-" in fields[0]:
				processed.append(line)
				continue
			else:
				fields[-1]=resps[i]
				processed.append("\t".join(fields))
				i+=1
		else:
			processed.append(line)
	pred = "\n".join(processed) + "\n"

	score_dict = get_scores(gold,pred,string_input=True)

	print("o Total tokens: " + str(score_dict["tok_count"]))
	print("o Gold " +score_dict["seg_type"]+": " + str(score_dict["gold_seg_count"]))
	print("o Predicted "+score_dict["seg_type"]+": " + str(score_dict["pred_seg_count"]))
	print("o Precision: " + str(score_dict["prec"]))
	print("o Recall: " + str(score_dict["rec"]))
	print("o F-Score: " + str(score_dict["f_score"]))

