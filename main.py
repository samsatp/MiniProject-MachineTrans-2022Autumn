from typing import Dict, Tuple, List, Set, NewType
import os, json

t_dtype = NewType('t_dtype', Dict[Tuple[str,str] , float])
language_corpus = NewType('language_corpus', List[List[str]])
alignments_dtype = NewType('alignments_dtype', List[List[Tuple[int, int]]])

def read_data(
    training:       bool = True,
    sentence_limit: int  = None,
    toy:            bool = False
    ) -> Tuple[language_corpus, language_corpus]:

    """
        This function read two files: target and source
        Then tokenize each sentence into list of words
    """

    if toy:
        eng = [['the', 'house'], ['the', 'book'], ['a', 'book']]
        foreign = [['das', 'Haus'], ['das', 'Buch'], ['ein', 'Buch']]
        return eng, foreign

    en_data_path = os.path.join("data","training.en" if training else "test.en")
    es_data_path = os.path.join("data","training.es" if training else "test.es")

    with open(en_data_path, 'r') as f:
        eng = f.readlines()[:sentence_limit]
        eng = [e.strip().split() for e in eng]
    with open(es_data_path, 'r') as f:
        foreign = f.readlines()[:sentence_limit]
        foreign = [e.strip().split() for e in foreign]
        
    if sentence_limit:
        return eng[:sentence_limit], foreign[:sentence_limit]
    return eng, foreign


def get_vocab(sentences: language_corpus) -> Set[str]:
    """
        This function returns a set containing all uniques word in a given corpus
    """
    vocab = set()
    for sentence in sentences:
        for word in sentence:
            vocab.add(word)
    return vocab


def train_iter(
    E: language_corpus, F: language_corpus, 
    vocab_e: Set[str], vocab_f: Set[str], 
    t: t_dtype
    ) -> t_dtype:
    """
        This function train one-epoch of EM

        ## parameters
        - `E`, `F`, `vocab_e`, `vocab_f` : The same as the train function

        - `t` : Dict[Tuple[str,str] , float] : 
            Our final Foreign-English dictionary
    """
    # initialize count and total
    count = dict()
    for e_sentence, f_sentence in zip(E,F):
        count.update({
            (e_word, f_word): 0.0 for e_word in e_sentence 
            for f_word in f_sentence
        })

    total = {f_word: 0.0 for f_word in vocab_f}

    
    for e_sentence, f_sentence in zip(E,F):
        # normalization term
        s_total = dict()
        for e_word in e_sentence:
            s_total[e_word] = sum([t[(e_word, f_word)] for f_word in f_sentence])

        # E step
        for e_word in e_sentence:
            for f_word in f_sentence:
                count[(e_word, f_word)] += t[(e_word, f_word)]  / s_total[e_word]
                total[f_word] += t[(e_word, f_word)]  / s_total[e_word]
    
    # M step
    for e_word in vocab_e:
        for f_word in vocab_f:
            try:
                t[(e_word, f_word)] = count[(e_word, f_word)] / total[f_word]
            except KeyError as e:
                assert (e_word, f_word) not in count
                pass
            except Exception as e:
                raise(e)

    return t

def write_prob(filepath, prob):
    """
        This function writes probability of translations into a file,
        for checking how probability converges at each iteration
    """
    output = ''
    for (ew, fw), p in prob.items():
        output+=f'{ew},{fw},{p}\n'
            
    with open(filepath, 'w') as f:
        f.write(output)


def train(
    E: language_corpus, F: language_corpus, 
    vocab_e: Set[str], vocab_f: Set[str], 
    iters: int
    ) -> t_dtype:
    """
        This function do the whole training process
        First, it initialize all the alignment weights with uniform weights
        Then, it train on the given data for a number of iterations

        ## Parameters
        - `E`, `F`: List[List[str]] : 
            For E, a list of all English sentences, each member is each sentence,
            each member of each sentence is each token in that sentence. 
            For F, the same of foreign sentences

        - `vocab_e`, `vocab_f`: set[str] :
            For vocab_e, a set of all unique vocab in English.
            For vocab_f, the same of foreign language
        
        - `iters`: int :
            The number of iterations to train EM
    """

    # initialize
    t = dict()
    for sentence_e, sentence_f in zip(E,F):
        t.update({
            (e_word, f_word): 1/len(vocab_e) for e_word in sentence_e 
            for f_word in sentence_f
        })

    for i in range(iters):
        print(f'iter: {i+1}')
        t = train_iter(E, F, vocab_e, vocab_f, t)
        write_prob(filepath=os.path.join('data',f'prob_{i}.csv'), prob=t)
        

    return t

def align(
    prob: t_dtype, 
    f_sentences: language_corpus, 
    e_sentences: language_corpus
    ) -> alignments_dtype:

    output = []

    for i, e_sentence in enumerate(e_sentences):
        sentence_align = []
        for ew_index, e_word in enumerate(e_sentence):
            max_prob = 0
            max_prob_fw_index = None
            for j, f_word in enumerate(f_sentences[i]):
                    try:
                        if prob[(e_word, f_word)] > max_prob:
                            max_prob = prob[(e_word, f_word)]
                            max_prob_fw_index = j
                    except KeyError as e:
                        continue
                    except Exception as e:
                        raise(e)
            sentence_align.append((max_prob_fw_index if max_prob_fw_index is not None else 999, ew_index))
        output.append(sentence_align)
            
    return output


def write_alignments(alignments: alignments_dtype) -> str:
    stdout = ''
    for alignment in alignments:
        stdout += " ".join([f"{e[0]}-{e[1]}" for e in alignment])
        stdout += "\n"

    output_file = os.path.join("data","res.txt")
    with open(output_file, 'w') as f:
        f.write(stdout)

    return output_file

def evaluate(output_file:str):
    with open(output_file, 'r') as f:
        pred = f.readlines()
        pred = [sentence.split() for sentence in pred]

    with open(os.path.join("data","test.align"), 'r') as f:
        gold = f.readlines()
        gold = [sentence.split() for sentence in gold]

    n_predict = 0
    n_gold = 0
    n_recall = 0
    n_precision = 0

    for pred_alings, gold_aligns in zip(pred, gold):
        n_gold += len(gold_aligns)
        for gold_align in gold_aligns:
            if gold_align in pred_alings:
                n_recall += 1
        n_predict += len(pred_alings)
        for pred_align in pred_alings:
            if pred_align in gold_aligns:
                n_precision += 1

    print(f'RECALL: {round(n_recall/n_gold, 3)}')
    print(f'PRECISION: {round(n_precision/n_predict, 3)}')

def write_translations():
    pass


if __name__ == "__main__":

    # Train
    print("TRAINING...")
    E, F = read_data(training=True, sentence_limit=10000)
    print(f'Data size: Eng: {len(E)}, Es: {len(F)}')

    vocab_e = get_vocab(E)
    vocab_f = get_vocab(F)
    print(f'Vocab size: Eng: {len(vocab_e)}, Es: {len(vocab_f)}')

    prob = train(E=E, F=F, vocab_e=vocab_e, vocab_f=vocab_f, iters=4)

    # Test
    print("\n\n\nTESTING...")
    E_test, F_test = read_data(training=False)
    alignments = align(prob=prob, f_sentences=F_test, e_sentences=E_test)
    output_file = write_alignments(alignments=alignments)

    evaluate(output_file)

