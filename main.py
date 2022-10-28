from typing import Dict, Tuple, List, Set
import os


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


def train_iter(E, F, vocab_e, vocab_f, t):
    count = dict()
    total = dict()

    #--- initialize 0 for all e, f
    for fw in vocab_f:
        total[fw] = 0.0
        for ew in vocab_e:
            count[(ew, fw)] = 0.0


    #--- Expectation
    s_total = dict()
    for e, f in zip(E, F):
        # Compute Normalization
        for ew in e:
            s_total[ew] = 0.0
            for fw in f:
                s_total[ew] += t[(ew, fw)]

        # Count
        for ew in e:
            for fw in f:
                count[(ew, fw)] += t[(ew, fw)]/s_total[ew]
                total[fw] += t[(ew, fw)] / s_total[ew]

    #--- Max
    for ew in vocab_e:
        for fw in vocab_f:
            t[(ew, fw)] = count[(ew, fw)] / total[fw]

    return t

def train(E, F, vocab_e, vocab_f, iters: int) -> Dict[Tuple, float]:
    # initialize
    t = dict()
    for ew in vocab_e:
        for fw in vocab_f:
            t[(ew, fw)] = 1/len(vocab_e)

    for i in range(iters):
        print(f'iter: {i+1}')
        t = train_iter(E, F, vocab_e, vocab_f, t)

    return t

def align(
    prob:Dict[Tuple, float], 
    source_sentences: List[List[str]], 
    target_sentences: List[List[str]]
    ) -> List[List[Tuple[int, int]]]:

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


def write_alignments(alignments: List[List[Tuple[int, int]]]):
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

    prob = train(E=E, F=F, vocab_e=vocab_e, vocab_f=vocab_f, iters=1)
    print(prob)
    alignments = align(prob=prob, source_sentences=F, target_sentences=E)
    write_alignments(alignments=alignments)

