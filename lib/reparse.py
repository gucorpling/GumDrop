import stanfordnlp, os, io, re
script_dir = os.path.dirname(os.path.realpath(__file__))
from opencc import OpenCC
s2t = OpenCC('s2t')
t2s = OpenCC('t2s')

data_dir = os.path.normpath("../../data")
corpus = "zho.rst.sctb"



def usestanfordparse(lines, language="zh", conversion=True):
	nlp = stanfordnlp.Pipeline(lang=language, tokenize_pretokenized=True, use_gpu=False)

	toks = []
	labels = []
	for line in lines:
		if '\t' in line:
			toks.append(line.split('\t')[1])
			labels.append(line.split('\t')[9])
		else:
			toks.append('_OOV_')
			labels.append('_OOV_')

	OOVindexes = [i for i, x in enumerate(toks) if x=="_OOV_"]
	nonOOVlabels = [x for x in labels if x!="_OOV_"]

	toparse = ""
	for i in range(len(OOVindexes)-1):
		toparse += " ".join(toks[OOVindexes[i]+1:OOVindexes[i+1]])+"\n"

	# Simplified to traditional
	if language=="zh" and conversion==True:
		toparse = s2t.convert(toparse)

	doc = nlp(toparse)
	parsed = doc.conll_file.conll_as_string()

	l_parsed = parsed.split('\n')
	assert len([x for x in l_parsed if "\t" in x]) == len(nonOOVlabels)
	id_nonOOVlabel = 0
	for id_parsed in range(len(l_parsed)):
		line = l_parsed[id_parsed]
		if "\t" in line:
			l_parsed[id_parsed] = re.sub(r'\t[^\n\t]+$', '\t'+nonOOVlabels[id_nonOOVlabel], line)
			id_nonOOVlabel += 1


	# add back meta data
	for id_line, line in enumerate(lines):
		if l_parsed[id_line]!=line and "\t" not in line:
			l_parsed.insert(id_line, line)

			# conll_parsed = conll_parsed[:id_line+1] + line + conll_parsed[id_line+1]


	# traditional to simplified
	if language=="zh" and conversion==True:
		l_parsed = [t2s.convert(x) for x in l_parsed]

	return l_parsed



if __name__ == "__main__":
	train = os.path.normpath(script_dir + ".." + os.sep + ".." + os.sep + 'data/zho.rst.sctb/zho.rst.sctb_train.conll')
	dev = os.path.normpath(script_dir + ".." + os.sep + ".." + os.sep + 'data/zho.rst.sctb/zho.rst.sctb_dev.conll')
	test = os.path.normpath(script_dir + ".." + os.sep + ".." + os.sep + 'data/zho.rst.sctb/zho.rst.test_test.conll')

	with io.open(train, 'r', encoding='utf8') as f:
		lines = f.read().split('\n')

	lines = usestanfordparse(lines)

	with io.open("stanfordparsed_" + os.path.basename(train), 'w', encoding='utf8') as f:
		f.write("\n".join(lines))

