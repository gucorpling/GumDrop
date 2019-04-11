from glob import glob
import os,sys,io,gc
import time
from datetime import timedelta
from argparse import ArgumentParser
from EnsembleSentencer import EnsembleSentencer
from lib.sentencers.lr_sent_wrapper import LRSentencer

start_time = time.time()
corpus_start_time = start_time
table_out = []

p = ArgumentParser()
p.add_argument("-c","--corpus",default="all",help="Corpus to start at")
p.add_argument("-d","--data_dir",default=os.path.normpath("../data"),help="Path to shared task data folder")
p.add_argument("-t","--tune_mode",default="paramwise",choices=["paramwise","full"])
p.add_argument("-r","--rare_thresh",type=int,default=100,help="Threshold rank for replacing words with POS tags")
p.add_argument("-e","--estimator",choices=["ensemble","lr"],default="ensemble",help="Estimator to score")
p.add_argument("-s","--size",type=int,default=80000,help="Maximum size of training data to use for large corpora")
p.add_argument("--mode",action="store",default="optimize-train-test",choices=["train-test","optimize-train-test","test"])
p.add_argument("--eval_test",action="store_true",help="Evaluate tests on test set, not dev")

opts = p.parse_args()

specific_corpus = opts.corpus
data_dir = opts.data_dir
mode = opts.mode
tune_mode = opts.tune_mode
estimator = opts.estimator

# For large corpora we do not perform multitraining, just use a subset
large_corpora = ["eng.pdtb.pdtb","eng.rst.rstdt","rus.rst.rrt","tur.pdtb.tdb"]

corpora = os.listdir(data_dir)
if specific_corpus == "all":
	corpora = [c for c in corpora if os.path.isdir(os.path.join(data_dir, c))]
else:
	corpora = [c for c in corpora if os.path.isdir(os.path.join(data_dir, c)) and c== specific_corpus]

log = io.open("results_sent.log",'w',encoding="utf8")

proceed = False

corpora = [c for c in corpora if os.path.isdir(os.path.join(data_dir, c))]

for corpus in corpora:

	log.write("="*10 + "\n")
	sys.stderr.write("\no Scoring corpus "+corpus+"\n")
	train = glob(data_dir + os.sep + corpus + os.sep + "*_train.conll")[0]
	dev = glob(data_dir + os.sep + corpus + os.sep + "*_dev.conll")[0]
	test = glob(data_dir + os.sep + corpus + os.sep + "*_test.conll")[0]
	model_path = "models" + os.sep + corpus + "_ensemble_sent.pkl"

	if "." in corpus:
		lang = corpus.split(".")[0]
	else:
		lang = "eng"

	# Predict sentence splits
	if opts.estimator == "ensemble":
		e = EnsembleSentencer(lang=lang,model=corpus)
	elif opts.estimator == "lr":
		e = LRSentencer(lang=lang,model=corpus)

	# Special genre patterns and feature settings
	if "gum" in corpus:
		e.genre_pat = "GUM_(.+)_.*"

	if "optimize" in mode and opts.estimator == "ensemble":  # Only supported for ensemble
		sys.stderr.write("\no Optimizing based on subset of data\n")
		vars, best_params = e.optimize(train,size=10000,tune_mode=tune_mode)
		# Now train on whole training set with those variables
		sys.stderr.write("\no Training best configuration\n")
		e.train(train,model_path=model_path,chosen_feats=vars,rare_thresh=100,clf_params=best_params,size=opts.size)
	elif "train" in mode:
		if corpus not in large_corpora:
			e.train(train,model_path=model_path,rare_thresh=100,size=None,multitrain=True)
		else:
			e.train(train,model_path=model_path,rare_thresh=100,size=opts.size,multitrain=False)


	if "test" in mode:
		if opts.eval_test:
			dev = test

		# Tag dev data and evaluate

		if opts.estimator == "ensemble":
			conf_mat, prec, rec, f1 = e.predict(dev,eval_gold=True,model_path=model_path,as_text=False)
		elif opts.estimator == "lr":
			conllu_in = io.open(dev, 'r', encoding='utf8').read()
			labels = []
			for line in conllu_in.split("\n"):
				if "\t" in line:
					fields = line.split("\t")
					if "-" in fields[0]:
						continue
					if fields[0] == "1":
						labels.append(1)
					else:
						labels.append(0)

			preds, probas = zip(*e.predict(dev,as_text=False,do_tag=True))
			from sklearn.metrics import classification_report, confusion_matrix
			conf_mat = confusion_matrix(labels, preds)
			true_positive = conf_mat[1][1]
			false_positive = conf_mat[0][1]
			false_negative = conf_mat[1][0]
			prec = true_positive / (true_positive + false_positive)
			rec = true_positive / (true_positive + false_negative)
			f1 = 2*prec*rec/(prec+rec)


		gc.collect()

		log.write(corpus + "\n")
		log.write("="*10 + "\n")
		log.write(str(conf_mat) + "\n")
		log.write("P: " + str(prec) + "\n")
		log.write("R: " + str(rec) + "\n")
		log.write("F1: " + str(f1) + "\n\n")

		sys.stderr.write("\no Total time elapsed: ")
		elapsed = time.time() - start_time
		sys.stderr.write(str(timedelta(seconds=elapsed)) + "\n\n")

		sys.stderr.write("\no Time analyzing this corpus: ")
		elapsed = time.time() - corpus_start_time
		sys.stderr.write(str(timedelta(seconds=elapsed)) + "\n\n")

		corpus_start_time = time.time()
		table_out.append("\t".join([corpus,str(prec),str(rec),str(f1)]))

	log.write("\n"+"\n".join(table_out)+"\n")
