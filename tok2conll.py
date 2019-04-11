#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Script to serialize automatic parses for .tok files
"""

import io, os, sys, re
from glob import glob
from argparse import ArgumentParser

from EnsembleSentencer import EnsembleSentencer
from lib.exec import exec_via_temp

script_dir = os.path.dirname(os.path.realpath(__file__))
lib = os.path.abspath(script_dir + os.sep + "lib")


class Tok2Conll:

	def __init__(self,lang="eng",model="eng.rst.gum"):
		self.lang = lang
		lang_map = {"deu":"german","eng":"english","spa":"spanish","fra":"french","nld":"dutch","rus":"russian","eus":"basque","por":"portuguese","zho":"chinese", "tur":"turkish"}
		self.long_lang = lang_map[lang] if lang in lang_map else lang
		self.genre_pat = "^(..)"
		try:
			self.udpipe_model = glob(os.path.abspath(os.path.join(lib,"udpipe",self.long_lang+"*.udpipe")))[0]
		except:
			sys.stderr.write("! Model not found for language " + self.long_lang + "*.udpipe in " + os.path.abspath(os.path.join([lib,"udpipe",self.long_lang+"*.udpipe"]))+"\n")
			sys.exit(0)
		self.udpipe_path = os.path.abspath(os.path.join(lib,"udpipe")) + os.sep

	def run_udpipe(self,text):

		cmd = [self.udpipe_path + "udpipe","--tag", "--parse", self.udpipe_model,"tempfilename"]
		parsed = exec_via_temp(text, cmd, workdir=self.udpipe_path)
		return parsed


if __name__ == "__main__":

	p = ArgumentParser()
	p.add_argument("-c","--corpus",default="spa.rst.sctb",help="Corpus name to parse or 'all' for all corpora")
	p.add_argument("-d","--data_dir",default=os.path.normpath("../data"),help="Path to shared task data folder")
	opts = p.parse_args()

	specific_corpus = opts.corpus
	data_dir = opts.data_dir

	corpora = os.listdir(data_dir)
	if specific_corpus == "all":
		corpora = [c for c in corpora if os.path.isdir(os.path.join(data_dir, c))]
	else:
		corpora = [c for c in corpora if os.path.isdir(os.path.join(data_dir, c)) and c== specific_corpus]

	for corpus in corpora:

		lang = corpus.split(".")[0]

		# Make Ensemble Sentencer
		e = EnsembleSentencer(model=corpus,lang=lang)

		# Special genre patterns
		if "gum" in corpus:
			e.genre_pat = "GUM_(.+)_.*"

		# loop through all *.tok files in the directory
		for f in glob(os.path.join(data_dir,corpus, corpus + "*.tok")):

			# predict sentence boundary
			predicted = e.predict(f, eval_gold=False, as_text=False).tolist()

			# read *.tok files and separate tokens by predicted sentence breaks (re-indexing from 1 for each sentence)
			conllu = io.open(f, encoding="utf8").readlines()
			i_pred = 0
			splitted = ''
			for l in conllu:
				if '\t' in l:
					if predicted[i_pred] == 1:
						tok_id = 1
						new_l = '\t'.join([str(tok_id)] + [l.split('\t')[1]] + ['_'] * 7 + [l.split('\t')[9]])
						splitted += '\n' + new_l
					else:
						new_l = '\t'.join([str(tok_id)] + [l.split('\t')[1]] + ['_'] * 7 + [l.split('\t')[9]])
						splitted += new_l
					tok_id += 1
					i_pred += 1
				else:
					splitted += l
					tok_id = 1

			# fix extra line breaks
			splitted = re.sub(r'(#[^\n\t]+\n)\n', r'\1', splitted)

			# run udpipe tagger & parser on the sentence-splitted conllu text
			t2c = Tok2Conll(lang=lang,model=corpus)
			output = t2c.run_udpipe(splitted)

			# assert that the parsed *.tok file has the same amount tokens as the original
			assert len([x for x in conllu if '\t' in x]) == len([x for x in output.split('\n') if '\t' in x])


			# save parsed *.tok files to corresponding subdirectories in data_parsed/
			data_parsed_dir = script_dir + os.sep + ".." + os.sep + "data_parsed" + os.sep + corpus
			if not os.path.exists(data_parsed_dir):
				os.makedirs(data_parsed_dir)

			with io.open(data_parsed_dir + os.sep + os.path.basename(f).replace(".tok",".conll"), 'w', encoding='utf8') as fout:
				fout.write(output)

			sys.stderr.write("o Saved parsed " + os.path.basename(f).replace(".tok",".conll") + " to data_parsed/ directory \n")





