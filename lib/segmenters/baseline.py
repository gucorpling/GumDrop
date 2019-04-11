import io, os, sys

script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "..")
sys.path.append(lib)

from glob import glob
import numpy as np
from conll_reader import read_conll
from sklearn.metrics import confusion_matrix

# data_dir = os.path.normpath("../../../data")

data_dir = os.path.abspath(lib + os.sep + ".." + os.sep + ".." + os.sep + "data_parsed")

corpora = os.listdir(data_dir)
corpora = [c for c in corpora if os.path.isdir(os.path.join(c, data_dir))]

log = io.open("baseline_parsed.log",'w',encoding="utf8")
table_out = []

all_f =[]

for corpus in corpora:
	if "pdtb" in corpus:
		continue
	dev = glob(data_dir + os.sep + corpus + os.sep + "*_test.conll")[0]
	train_feats, vocab, toks, firsts, lasts = read_conll(dev)
	labels = [int(t["label"]!="_") for t in train_feats]
	baseline_preds = [int(t["wid"]==1) for t in train_feats]
	conf_mat = confusion_matrix(labels, baseline_preds)
	true_positive = conf_mat[1][1]
	false_positive = conf_mat[0][1]
	false_negative = conf_mat[1][0]
	prec = true_positive / (true_positive + false_positive)
	rec = true_positive / (true_positive + false_negative)
	f1 = 2*prec*rec/(prec+rec)
	log.write("corpus: " + corpus + "\n")
	log.write("="*10 + "\n")
	log.write(str(confusion_matrix(labels, baseline_preds)) + "\n")
	log.write("P: " + str(prec) + "\n")
	log.write("R: " + str(rec) + "\n")
	log.write("F1: " + str(f1) + "\n\n")
	all_f.append(f1)
	table_out.append("\t".join([corpus,str(prec),str(rec),str(f1)]))
sys.stderr.write("Mean F1: " + str(sum(all_f)/len(all_f))+"\n")
log.write("\n"+"\n".join(table_out)+"\n")