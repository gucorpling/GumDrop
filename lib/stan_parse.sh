
for j in  spa.rst.rststb spa.rst.sctb zho.rst.sctb rus.rst.rrt 
	do
		for i in train dev test
			do
		       echo -e "\n### Parsing Stanford *.conll on \n ### corpus $j ### file $i ###"
		       python3 stan_parse.py -c $j -f $i
    done;
done;
