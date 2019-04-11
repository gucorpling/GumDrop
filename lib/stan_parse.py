import stanfordnlp
from stanfordnlp.models.common.conll import CoNLLFile
import io, sys, os, re
from collections import defaultdict
from argparse import ArgumentParser
from opencc import OpenCC
s2t = OpenCC('s2t')
t2s = OpenCC('t2s')


data_dir = os.path.normpath("../../data")
data_parsed_dir = os.path.normpath( "../../data_stanparse")




def fetch_line_per_sent(lines):

	sents = []
	tokens = []
	segs = []
	comments = defaultdict(list)
	i = -1
	for line in lines:
		if "\t" in line:
			fields = line.split("\t")
			if "-" in fields[0]:
				continue
			segs.append(fields[-1].strip())
			tokens.append(fields[1])
			i+=1
		elif len(line.strip()) == 0:
			if len(tokens)>0:
				sents.append(" ".join(tokens))
		elif line.startswith("#"):
			comments[i].append(line.strip())

	sents = "\n".join(sents)
	if opts.corpus[:3] == "zho":
		sents = s2t.convert(sents)
	return "\n".join(sents), segs, comments


def outputstanparse():
	output = []
	if -1 in comments:
		output = comments[-1] + output
	i = 0
	for s in annotated.conll_file._sents:
		for word in s:
			if i - 1 in comments and i != 0:
				output += comments[i - 1]
			fields = word[:]
			if i == len(segs):
				a = 5
			fields[-1] = segs[i]
			output.append("\t".join(fields))
			i += 1
		output.append("")
	return output


if __name__ == "__main__":

	p = ArgumentParser()
	p.add_argument("-c","--corpus",default="spa.rst.sctb",help="corpus to use or 'all'")
	p.add_argument("-d","--data_dir",default=os.path.normpath("../../data"),help="Path to shared task data folder")
	p.add_argument("-f","--filetoparse",action="store", default="test", choices=["train", "dev", "test"], help="Choose the conll file to parse: train, dev or test")
	opts = p.parse_args()

	lang_map = {"deu": "de", "eng": "en", "spa": "es", "fra": "fr", "nld": "nl", "rus": "ru",
				"eus": "eu", "por": "pt", "zho": "zh", "tur": "tr"}

	filepath = data_dir + os.sep + opts.corpus + os.sep + opts.corpus + "_" + opts.filetoparse + ".conll"

	config = {
			'processors': 'pos,lemma,depparse',
			'tokenize_pretokenized': True,
			'lang': lang_map[opts.corpus[:3]],
			# 'treebank': 'es_ancora',
			'use_gpu':False
			 }

	nlp = stanfordnlp.Pipeline(**config)

	with io.open(filepath,encoding="utf8") as f:
		lines = f.readlines()
	split_indexes = [i for i,x in enumerate(lines) if x.startswith('# newdoc')]

	# select every 5th index
	split_indexes = [x for i,x in enumerate(split_indexes) if i%5==0]


	all_outputs = []
	for i, split_i in enumerate(split_indexes):
		print("%d/%d" %(i, len(split_indexes)))
		if i == len(split_indexes)-1:
			new_input = lines[split_i:]
		else:
			new_input = lines[split_i:split_indexes[i+1]]

		sents, segs, comments = fetch_line_per_sent(new_input)
		doc = stanfordnlp.Document('')

		doc.conll_file = CoNLLFile(input_str="".join(new_input))
		annotated = nlp(doc)
		# all_outputs += doc.conll_file.conll_as_string() + "\n"

		new_output = outputstanparse()
		all_outputs += new_output
		del doc
		del annotated





	# doc.conll_file = CoNLLFile(input_str=alldoc)

	# annotated = nlp(doc)

	#annotated = nlp(sents)

	# output = outputstanparse()

	if opts.corpus[:3] == "zho":
		all_outputs = [t2s.convert(x) for x in all_outputs]
		# all_outputs = t2s.convert(all_outputs)

	#annotated.write_conll_to_file("out_stan.conll")

	if not os.path.exists(data_parsed_dir + os.sep + opts.corpus):
		os.makedirs(data_parsed_dir + os.sep + opts.corpus)

	filestanparsedpath = data_parsed_dir + os.sep + opts.corpus + os.sep + opts.corpus + "_" + opts.filetoparse + ".conll"

	with io.open(filestanparsedpath,'w',encoding="utf8",newline="\n") as f:
		f.write("\n".join(all_outputs).strip() + "\n\n")
		# f.write(all_outputs)