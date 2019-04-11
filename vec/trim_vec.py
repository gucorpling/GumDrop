import sys, io, os

lim = 50000
dim = 300

with io.open(sys.argv[1],encoding="utf8") as fin:
	with io.open(os.path.basename(sys.argv[1])+"_trim.vec",'w',encoding="utf8",newline="\n") as fout:
		for i, line in enumerate(fin.readlines()):
			if line.strip().count(" ") != dim or " " in line:  # Prevent no-break space in line or line does not have dim dimensions
				continue
			if line.startswith("rik"):
				a=3
			if i < lim:
				fout.write(line.strip() + "\n")
			else:
				sys.exit()