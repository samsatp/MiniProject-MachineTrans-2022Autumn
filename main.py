from typing import Dict, Tuple, List, Set, NewType
import os, json, argparse

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

        ## Parameters
        1. `training`: bool, default = True : 
            If True, read training datasets. If False, read testing datasets.

        2. `sentence_limit`: int, default = None : 
            How many sentences to include in training process.
            If None, use all sentences.

        3. `toy`: bool, default = False : 
            If True use toy dataset(book example)
    """

    if toy:
        eng = [['the', 'house'], ['the', 'book'], ['a', 'book']]
        foreign = [['das', 'Haus'], ['das', 'Buch'], ['ein', 'Buch']]
        return eng, foreign

    en_data_path = os.path.join("data","training.en" if training else "test.en")
    es_data_path = os.path.join("data","training.es" if training else "test.es")

    with open(en_data_path, 'r') as f:
        eng = f.readlines()
        eng = [e.strip().split() for e in eng]
    with open(es_data_path, 'r') as f:
        foreign = f.readlines()
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
    print('\t\tinitializing...')
    count = dict()
    for e_sentence, f_sentence in zip(E,F):
        count.update({
            (e_word, f_word): 0.0 for e_word in e_sentence 
            for f_word in f_sentence
        })
    total = {f_word: 0.0 for f_word in vocab_f}

    print('\t\tDoing Expectation step...')
    for i, (e_sentence, f_sentence) in enumerate(zip(E,F)):
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
    print('\t\tDoing Maximizing step...')
    i = 0
    for e_word in vocab_e:
        for f_word in vocab_f:
            i += 1
            if (e_word, f_word) in count.keys():
                t[(e_word, f_word)] = count[(e_word, f_word)] / total[f_word]
            
    print()
    return t


def train(
    E: language_corpus, F: language_corpus, 
    vocab_e: Set[str], vocab_f: Set[str], 
    iters: int,
    write_prob_at_each_iter: bool = False
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

        - `write_prob_at_each_iter`: bool, default = False :
            If true, at the end of each epoch, write a file 
            containing the probabilities t(e|f). The purpose is 
            to see behaviour of t(e|f) for debugging.
    """

    # initialize
    t = dict()
    for sentence_e, sentence_f in zip(E,F):
        t.update({
            (e_word, f_word): 1/len(vocab_e) for e_word in sentence_e 
            for f_word in sentence_f
        })

    for i in range(iters):
        print(f'\n\t-epoch: {i+1}')
        t = train_iter(E, F, vocab_e, vocab_f, t)

        if write_prob_at_each_iter:
            write_prob(filepath=os.path.join('data',f'prob_{i+1}.csv'), prob=t)

    return t


def write_prob(filepath, prob):
    """
        This function writes probabilities t(e|f) into a file,
        for checking how probabilities converges at each iteration
    """
    print('writing prob...')
    output = ''
    for (ew, fw), p in prob.items():
        output+=f'{ew},{fw},{p}\n'
            
    with open(filepath, 'w') as f:
        f.write(output)


def align(
    prob: t_dtype, 
    f_sentences: language_corpus, 
    e_sentences: language_corpus
    ) -> alignments_dtype:
    """
        This function creates alignments. Each target word needs to
        align to some word in a source sentence. Alignment is done by
        argmax decoding.
    """

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
    """
        This funtion writes alignment results to a file
    """
    stdout = ''
    for alignment in alignments:
        stdout += " ".join([f"{e[0]}-{e[1]}" for e in alignment])
        stdout += "\n"

    output_file = os.path.join("data","results.align")
    with open(output_file, 'w') as f:
        f.write(stdout)

    return output_file


def evaluate(output_file:str):
    """
        This function tests the model with testing dataset
        by calculating precision, recall, f1-score
    """

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

    precision = round(n_precision/n_predict, 4)
    recall = round(n_recall/n_gold, 4)
    f_1 = round(2*((precision*recall)/(precision+recall)), 4)

    print(f'\t\tRECALL: {recall*100}%')
    print(f'\t\tPRECISION: {precision*100}%')
    print(f'\t\tF1 score: {f_1*100}%')


def get_dictionary(prob: t_dtype):
    """
        This function summarizes t(e|f) into a source-target dictionary
    """
    dictionary = dict()
    scores = dict()
    
    for (e_word, f_word), p in prob.items():
        if f_word not in dictionary:
            dictionary[f_word] = e_word
            scores[f_word] = p
        else:
            if scores[f_word] < p:
                dictionary[f_word] = e_word
                scores[f_word] = p
    
    with open(os.path.join("data", "dictionary.json"), 'w') as f:
        json.dump(dictionary, f)
        print("\tWrite dictionary...")
    return dictionary, scores


def write_translations(
    dictionary: Dict[str, str], 
    source_sentences: language_corpus,
    prob: t_dtype):
    """
        This function use source-target dictionary to
        translate source sentences into target sentences
        and write the results to a file
    """

    output = ""
    for sentence in source_sentences:
        output += " ".join(sentence) + " "
        translation_prob = 1
        for f_word in sentence:
            if f_word in dictionary:
                e_word = dictionary[f_word]
                translation_prob *= (prob[(e_word, f_word)] / sum([prob[(e_word, fw)]  for fw in sentence if (e_word, fw) in prob]))
            else:
                # This is when the foreign word 
                # doesn't appear in our training data
                e_word = '<UNK>'
            output += e_word
            output += " "
        output += str(translation_prob) + "\n"

    with open(os.path.join("data", "translations.txt"), "w") as f:
        f.write(output)    
        print("\nDone!")


if __name__ == "__main__":
    print("="*30)
    print("""
    README
    - To check t(e|f) of the Book example at each epoch, set param: `--toy=True --writeP=True`
        Then this program will write files: data/prob_{i}.csv which contains t(e|f) at the end of i-th epoch.
        Note that i starts from 1, i.e i=1,2,3,...

    - data/results.align is a resulting file from write_alignments() function

    - data/translations.txt is a resulting file from write_translations() function
    """)
    print("="*30)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', type=bool, nargs='?', const=1, default=False, help="set to True to use book example")
    parser.add_argument('--limit', type=int, nargs='?', const=1, default=None, help="limit the number of sentences used to train model")
    parser.add_argument('--epochs', type=int, nargs='?', const=1, default=3, help="specify how many epochs")
    parser.add_argument('--writeP', type=bool, nargs='?', const=1, default=False, help="set to True to write t(e|f) at each epoch")
    args = parser.parse_args()

    # Train
    print("TRAINING...")
    E, F = read_data(training=True, sentence_limit=args.limit, toy=args.toy)
    print(f'Data size: Eng: {len(E)}, Es: {len(F)}')

    vocab_e = get_vocab(E)
    vocab_f = get_vocab(F)
    print(f'Vocab size: Eng: {len(vocab_e)}, Es: {len(vocab_f)}')

    print(f"running {args.epochs} epochs...")
    prob = train(E=E, F=F, vocab_e=vocab_e, vocab_f=vocab_f, iters=args.epochs, write_prob_at_each_iter=args.writeP)
    dictionary, scores = get_dictionary(prob=prob)

    # Test
    if not args.toy:
        print("\n\nTESTING...")
        E_test, F_test = read_data(training=False)

        print("\tDoing alignments...")
        alignments = align(prob=prob, f_sentences=F_test, e_sentences=E_test)
        output_file = write_alignments(alignments=alignments)

        print("\tDoing evaluation...")
        evaluate(output_file)

        print("\tDoing translation...")
        write_translations(dictionary=dictionary, source_sentences=F_test, prob=prob)
