from glob import glob
from lib.segmenters.subtree_segmenter import SubtreeSegmenter
from lib.segmenters.rnn_segmenter import RNNSegmenter
from EnsembleSegmenter import EnsembleSegmenter
from lib.conll_reader import get_seg_labs
import os,sys,io
import time
from datetime import timedelta
from argparse import ArgumentParser
from lib.tune import get_best_params
from sklearn.metrics import confusion_matrix

script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "..")
models = os.path.abspath(script_dir + os.sep + "models")

start_time = time.time()
corpus_start_time = start_time
table_out = []

p = ArgumentParser()
p.add_argument("-c","--corpus",default="deu.rst.pcc",help="Corpus to evaluate or 'all'")
p.add_argument("-d","--data_dir",default=os.path.normpath("../data"),help="Path to shared task data folder")
p.add_argument("-t","--tune_mode",default="paramwise",choices=["paramwise","full"])
p.add_argument("-r","--rare_thresh",type=int,default=200,help="Threshold rank for replacing words with POS tags")
p.add_argument("-e","--estimator",choices=["subtree","rnn","ensemble"],default="ensemble",help="Estimator to train/score")
p.add_argument("-b","--best_params",action="store_true",help="Load best parameters from file")
p.add_argument("--mode",action="store",default="optimize-train-test",choices=["train-test","optimize-train-test","test"])
p.add_argument("--eval_test",action="store_true",help="Evaluate tests on test set, not dev")
p.add_argument("-o","--outfile",action="store_true",help="Print output file CORPUS.pred.out.conll for EnsembleSegmenter")
p.add_argument("--auto",action="store_true",help="Evaluate on automatic parse")

opts = p.parse_args()

corpus = opts.corpus
data_dir = opts.data_dir
mode = opts.mode
tune_mode = opts.tune_mode
estimator = opts.estimator  # Only subtree supported initially, this is a place holder to test different modules

if opts.auto:
	data_dir = data_dir + "_parsed"
	sys.stderr.write("o Evaluating on automatically parsed data\n")


corpora = os.listdir(data_dir)
if corpus == "all":
	corpora = [c for c in corpora if os.path.isdir(os.path.join(data_dir, c))]
else:
	corpora = [c for c in corpora if os.path.isdir(os.path.join(data_dir, c)) and c == corpus]

log = io.open("segmenter_results.log",'w',encoding="utf8")

auto = "_auto" if opts.auto else ""

for corpus in corpora:
	if "pdtb" in corpus or "stan" in corpus:  # Not scoring PDTB connective detection with segmenter module
		continue
	log.write("="*10 + "\n")
	sys.stderr.write("\no Scoring corpus "+corpus+"\n")
	train = glob(data_dir + os.sep + corpus + os.sep + "*_train.conll")[0]
	dev = glob(data_dir + os.sep + corpus + os.sep + "*_dev.conll")[0]
	test = glob(data_dir + os.sep + corpus + os.sep + "*_test.conll")[0]

	# Set estimator module being tested
	if estimator == "subtree":
		est = SubtreeSegmenter(model=corpus,lang=corpus.split(".")[0],auto=auto)
	elif estimator == "rnn":
		est = RNNSegmenter(model=corpus,auto=auto,load_params=opts.best_params)
		if "optimize" in mode:  # No optimization in global scorer for RNN
			mode = "train-test"
	else:  # Default estimator
		est = EnsembleSegmenter(lang=corpus.split(".")[0],model=corpus,multitrain=True)

	# Special genre patterns and feature settings
	if "gum" in corpus:
		est.genre_pat = "GUM_(.+)_.*"

	if "optimize" in mode:
		clf, vars, best_params = est.optimize(train,size=5000,tune_mode=tune_mode,cached_params=opts.best_params)
		if "best_score" in best_params:
			best_params.pop("best_score")
		# Now train on whole training set with those variables
		sys.stderr.write("\no Training best configuration\n")
		if len(vars) > 0:
			est.train(train,chosen_feats=vars,rare_thresh=200,clf_params=best_params,as_text=False,multitrain=True)
		else:
			est.train(train,rare_thresh=200,clf_params=best_params,as_text=False,chosen_clf=clf)
	elif "train" in mode:
		if opts.best_params and est.name in ["SubtreeSegmenter","EnsembleSegmenter"]:
			best_clf, params, feats = get_best_params(corpus, est.name)
		else:
			best_clf = None
			params = None
			feats = None
		if est.name == "SubtreeSegmenter":
			est.train(train,rare_thresh=200,as_text=False,multitrain=True,chosen_clf=best_clf,clf_params=params,chosen_feats=feats)
		elif est.name =="RNNSegmenter":
			est.train(train,as_text=False,multifolds=5)
		elif est.name =="EnsembleSegmenter":
			est.train(train,as_text=False,chosen_clf=best_clf,clf_params=params)

	if "test" in mode:
		# Get prediction performance on dev or test

		if opts.eval_test:
			dev = test

		if est.name == "SubtreeSegmenter":
			conf_mat, prec, rec, f1 = est.predict(dev,eval_gold=True,as_text=False)
		elif est.name == "EnsembleSegmenter":
			conf_mat, prec, rec, f1 = est.predict(dev,eval_gold=True,as_text=False,serialize=opts.outfile)
		elif est.name == "RNNSegmenter":
			pred_labs, _ = zip(*est.predict(dev,as_text=False))
			gold_labs = get_seg_labs(dev,as_text=False)
			conf_mat = confusion_matrix(gold_labs,pred_labs)
			true_positive = conf_mat[1][1]
			false_positive = conf_mat[0][1]
			false_negative = conf_mat[1][0]
			prec = true_positive / (true_positive + false_positive)
			rec = true_positive / (true_positive + false_negative)
			f1 = 2*prec*rec/(prec+rec)
			sys.stderr.write(corpus + "\n")
			sys.stderr.write("="*10 + "\n")
			sys.stderr.write(str(conf_mat) + "\n")
			sys.stderr.write("P: " + str(prec) + "\n")
			sys.stderr.write("R: " + str(rec) + "\n")
			sys.stderr.write("F1: " + str(f1) + "\n\n")

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

if "test" in mode:
	log.write("\n"+"\n".join(table_out)+"\n")
