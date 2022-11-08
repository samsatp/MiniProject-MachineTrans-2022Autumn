# Implementation
- IBM model1
- Alignment
- Translation
- Evaluation


# How to use
- To check the probability of the ranslation `t(e|f)` of the Book example at each epoch, set param: `--toy=True --writeP=True`
    Then this program will write files: data/prob_{i}.csv which contains t(e|f) at the end of i-th epoch.
    Note that i starts from 1, i.e i=1,2,3,...

- data/results.align is a resulting file from write_alignments() function

- data/translations.txt is a resulting file from write_translations() function
    
```bash
usage: main.py [-h] [--toy [TOY]] [--limit [LIMIT]] [--epochs [EPOCHS]] [--writeP [WRITEP]]

options:
  -h, --help         show this help message and exit
  --toy [TOY]        set to True to use book example
  --limit [LIMIT]    limit the number of sentences used to train model
  --epochs [EPOCHS]  specify how many epochs
  --writeP [WRITEP]  set to True to write t(e|f) at each epoch
```