from typing import Dict, Tuple, List, Set, NewType
import os

t_dtype = NewType('t_dtype', Dict[Tuple[str,str] , float])
language_corpus = NewType('language_corpus', List[List[str]])
alignments_dtype = NewType('alignments_dtype', List[List[Tuple[int, int]]])

def read_data(
    training: bool = True,
    sentence_limit: int = None,
    toy: bool = False
    ) -> Tuple[List[List[str]]]:

    """
        This function read two files: target and source
        and tokenize each sentence into list of words
    """

    if toy:
        eng = [['the', 'house'], ['the', 'book'], ['a', 'book']]
        foreign = [['das', 'Haus'], ['das', 'Buch'], ['ein', 'Buch']]
        return eng, foreign

    en_data_path = os.path.join("data","training.en" if training else "test.en")
    es_data_path = os.path.join("data","training.es" if training else "test.en")

    with open(en_data_path, 'r') as f:
        eng = f.readlines()[:sentence_limit]
        eng = [e.strip().split() for e in eng]
    with open(es_data_path, 'r') as f:
        foreign = f.readlines()[:sentence_limit]
        foreign = [e.strip().split() for e in foreign]
        
    if sentence_limit:
        return eng[:sentence_limit], foreign[:sentence_limit]
    return eng, foreign


def get_vocab(sentences: List[List[str]]) -> Set[str]:
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
        - `E`, `F`: List[List[str]] :
            For E, a list of all English sentences, each member is each sentence,
            each member of each sentence is each token in that sentence. 
            For F, the same of foreign sentences

        - `vocab_e`, `vocab_f`: set[str] :
            For vocab_e, a set of all unique vocab in English.
            For vocab_f, the same of foreign language

        - `t` : Dict[Tuple[str,str] , float]
            Our final Foreign-English dictionary
    """
    # initialize count and total
    count = dict()
    for sentence_e, sentence_f in zip(E,F):
        count.update({
            (e_word, f_word): 0.0 for e_word in sentence_e 
            for f_word in sentence_f
        })

    total = {f_word: 0.0 for f_word in vocab_f}

    
    for sentence_e, sentence_f in zip(E,F):

        # normalization term
        s_total = dict()
        for e_word in sentence_e:
            s_total[e_word] = sum([t[(e_word, f_word)] for f_word in sentence_f])

        # E step
        for e_word in sentence_e:
            for f_word in sentence_f:
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

def train(
    E: language_corpus, F: language_corpus, 
    vocab_e: Set[str], vocab_f: Set[str], 
    iters: int
    ) -> t_dtype:

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

    return t

def align(
    prob: t_dtype, 
    source_sentences: language_corpus, 
    target_sentences: language_corpus
    ) -> alignments_dtype:

    output = []

    for i, target_sentence in enumerate(target_sentences):
        sentence_align = []
        for k, ew in enumerate(target_sentence):
            max_prob = 0
            max_prob_fw = None
            for j, fw in enumerate(source_sentences[i]):
                if prob[(ew, fw)] > max_prob:
                    max_prob = prob[(ew, fw)]
                    max_prob_fw = j
            sentence_align.append((max_prob_fw,k))
        output.append(sentence_align)
            
    return output


def write_alignments(alignments: alignments_dtype):
    stdout = ''
    for alignment in alignments:
        stdout += " ".join([f"{e[0]}-{e[1]}" for e in alignment])
        stdout += "\n"

    with open("data/res.txt", 'w') as f:
        f.write(stdout)

    pass

def evaluate():
    pass

def write_translations():
    pass


if __name__ == "__main__":
    E, F = read_data(toy=True)
    print(f'Data size: Eng: {len(E)}, Es: {len(F)}')

    vocab_e = get_vocab(E)
    vocab_f = get_vocab(F)
    print(f'Vocab size: Eng: {len(vocab_e)}, Es: {len(vocab_f)}')

    prob = train(E=E, F=F, vocab_e=vocab_e, vocab_f=vocab_f, iters=3)
    print(prob)
    alignments = align(prob=prob, source_sentences=F, target_sentences=E)
    write_alignments(alignments=alignments)

