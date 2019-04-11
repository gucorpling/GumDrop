# GumDrop

Ensemble EDU segmenter, sentencer and connective detector

## Requirements

Pre-trained models assume Python 3.6 and use the following libraries:

  * nltk
  * scikit-learn
  * pandas
  * numpy
  * scipy
  * xgboost
  * tensorflow-gpu
  * keras
  * pytorch
  * opencc-python-reimplemented 
  * hyperopt
  
Note also that GPU support is required for the neural network libraries (i.e. CUDA+cuDNN 9 or 10). Additionally some external tools are used for parsing and tagging and not included, though pre-trained models are available. Please make sure to put:

  * A tree-tagger binary for your OS into the folder lib/treetagger/ (from http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)
  * A udpipe binary for your OS into the folder lib/udpipe/ (from https://ufal.mff.cuni.cz/udpipe)

## External resources

After installing the required libraries, do the following:

  * Get word FastText word embeddings and put them in gumdrop/vec/; files are named like `wiki.spa.vec_trim.vec` (note that paper results are based on trimmed embeddings containing the top 50,000 items for each language)
  * Get UDPipe models and put them in gumdrop/lib/udpipe/; files are named like `spanish-gsd-ud-2.3-181115.udpipe` (must begin with the language 'long name' and end in .udpipe)
  * Get TreeTagger models and put them in gumdrop/lib/treetagger/; files are named like `spa.par`
  * Put the shared task folder in data/, a sister folder of the gumdrop/ repo folder
  * If you want to auto-parse raw .tok data, create another folder data_parsed/ and run `python tok2conll.py -c all` in the main system folder to reproduce auto sentence splitting and parsing with UDPipe.

## Reproducing paper results

Results from the DISRPT2019 paper can be reproduced by running the following in Python 3.6, assuming the gumdrop/ folder is a sister of the shared task data/ folder.

### Discourse unit segmentation 

Run the following command (note this requires GPU support for the neural network):

```
python get_segmenter_scores.py -c all --mode test --eval_test -o
```

This will print P/R/F scores to segmenter_results.log, as well as producing predicted output conll files for each test set (if the option `-o` is used). You can also get results for specific corpora using `-c CORPUSNAME`

To reproduce results on automatic parses, run:

```
python get_segmenter_scores.py -c all --mode test --eval_test -o --auto
```

Note that this assumes that the automatically sentence splitted and UDPipe parsed files are already in a sister folder of gumdrop/, called data_parsed/. To reproduce the parse data from shared task input folders, see below.

### Connective detection

This runs similarly to the discourse unit segmentation, for example:

```
py -3.6 EnsembleConnective.py -c zho.pdtb.cdtb --mode train-test --eval_test
```

## Reproducing sentence splitting and automatic parses

To reproduce the included automatically parsed files, run:

```
python tok2conll.py -c all
```

This assumes that data/ is a sister folder of gumdrop/ with complete shared task data, and that the parsed output should be placed in data_parsed/

## Retraining the system

To retrain all models, run the respective modules as shown below. The option -b always loads stored best hyperparameters from the appropriate params/ subfolder. Retraining can take a long time, especially for RNNs. To train on auto-parse data, make sure the data is in data_parsed/ and add the --auto flag. If you are retraining the Ensemble, you will need to re-use the existing files in multitrain/, or else rerun multitraining using the --multitrain option for each trainable module. Note that a few rule based modules are not trainable (e.g. nltk_sent_wrapper.py).

```
python subtree_segmenter.py --mode train-test -b -c all
python rnn_segmenter.py --mode train-test -b -c all
python bow_seg_counter.py --mode train-test -b -c all
```

## Re-optimizing hyperparameters

Hyperparameter optimization was carried using `hyperopt` on a GPU cluster, and can be expected to take **long times**. To re-optimize, add 'optimize-' to the mode switch, and set tune mode to `-t hyperopt` for some modules, as follows, for example:

```
python subtree_segmenter.py --mode optimize-train-test -t hyperopt -c spa.rst.sctb
python rnn_segmenter.py --mode tune -c spa.rst.sctb
```

Separate optimization should be carried out for automatic parses.
