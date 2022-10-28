from typing import Dict, Tuple, List
import os


def read_data(training: bool = True):
    en_data_path = os.path.join("data","training.en" if training else "test.en")
    es_data_path = os.path.join("data","training.es" if training else "test.en")
    
    with open(en_data_path, 'r') as f:
        eng = f.readlines()
        eng = [e.strip().split() for e in eng]
    with open(es_data_path, 'r') as f:
        foreign = f.readlines()
        foreign = [e.strip().split() for e in foreign]
        
    return eng, foreign



def train() -> Dict[Tuple, float]:
    pass

def align(
    prob:Dict[Tuple, float], 
    source_sentences: List[List[str]], 
    target_sentences: List[List[str]]
    ) -> List[List[Tuple[int, int]]]:

    pass

def write_alignments(alignments: List[List[Tuple[int, int]]]):
    pass

def evaluate():
    pass

def write_translations():
    pass


if __name__ == "__main__":
    E, F = read_data(training=True)

    prob = train()
    alignments = align(prob=prob)

    pass