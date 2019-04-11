#!/usr/bin/python
# -*- coding: utf-8 -*-

import io, os
from collections import defaultdict, Counter
from argparse import ArgumentParser

thresh = 220

def read_bow(infile, as_text=False, test_vocab=None):
	output = []
	vocab = defaultdict(int)
	s_len = 0
	sent = defaultdict(int)
	sent["__LABEL__"] = 0
	all_sents = []
	seg_count = 0
	label = 0

	if as_text:
		lines = infile.split("\n")
	else:
		lines = io.open(infile,encoding="utf8").readlines()

	for line in lines:
		if "\t" in line:
			fields = line.split("\t")
			if "-" in fields[0]:
				continue
			word, lemma, pos, cpos, feats, head, deprel = fields[1:-2]
			label = fields[-1]
			vocab[word] += 1
			vocab[pos] += 1
			s_len +=1
			sent[word] +=1
			sent[pos] += 1
			if "BeginSeg" in label:
				seg_count += 1
				#if seg_count > 1:
				#	sent["__LABEL__"] = 1
				sent["__LABEL__"] = seg_count
		elif len(line.strip())==0:
			sent["__SLEN__"] = s_len
			all_sents.append(sent)
			sent = defaultdict(int)
			sent["__LABEL__"] = 0
			s_len = 0
			seg_count = 0

	if s_len > 0 :
		sent["__SLEN__"] = s_len
		all_sents.append(sent)

	if test_vocab is not None:
		all_rows = []
		for s in all_sents:
			row = []
			for key in test_vocab:
				if key in s:
					row.append(s[key])
				else:
					row.append(0)
			row.append(s["__SLEN__"])
			row.append(s["__LABEL__"])
			all_rows.append(row)
		return all_rows
	else:
		vocab = Counter(vocab)
		top_n = vocab.most_common(thresh)
		top_n = [t[0] for t in top_n]
		for s in all_sents:
			s_copy = {}
			s_copy.update(s)
			for f in s:
				if f not in top_n and f not in ["__LABEL__","__SLEN__"]:
					s_copy.pop(f)
			output.append(s_copy)

		outrows = []
		all_keys = sorted(list(top_n),reverse=True)
		for s in output:
			row = []
			for key in all_keys:
				row.append(str(s[key])) if key in s else row.append("0")
			row.append(str(s["__SLEN__"]))
			row.append(str(s["__LABEL__"]))
			outrows.append(row)
		return outrows, all_keys

if __name__ == "__main__":

	p = ArgumentParser()
	p.add_argument("-c","--corpus",default="deu.rst.pcc")
	p.add_argument("-d","--data_dir",default=os.path.normpath("../../data"),help="Path to shared task data folder")
	opts = p.parse_args()

	corpus = opts.corpus
	data_dir = opts.data_dir

	tab,_ = read_bow(data_dir + corpus +os.sep + corpus+ "_dev.conll")
	for row in tab:
		print("\t".join(row))