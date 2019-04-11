#!/usr/bin/bash

for j in  tur.pdtb.tdb
	do
       echo -e "\n### $j ###"
       python3 tok2conll.py -c $j
done;

# deu.rst.pcc  eus.rst.ert  spa.rst.sctb eng.pdtb.pdtb  fra.sdrt.annodis tur.pdtb.tdb eng.rst.gum  nld.rst.nldt  zho.pdtb.cdtb eng.rst.rstdt  por.rst.cstn  zho.rst.sctb eng.sdrt.stac  rus.rst.rrt  spa.rst.rststb