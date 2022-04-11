import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import pandas as pd
import spacy
from enum import Enum
from typing import List, Tuple, Optional, Set, Iterable

nltk.download("wordnet")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class wordnet:
    def __init__(self):
        self.output_path = '../Data/'

    def wn_synsets_from_word_pos(self, word: str, pos: str) -> List[str]:
        wn_offset = []
        for synset in wn.synsets(word, pos):
            wn_offset.append(self.wn_offset_from_synset(synset))
        return wn_offset

    def wn_offset_from_synset(self, synset: Synset) -> str:
        return 'wn:' + str(synset.offset()).zfill(8) + synset.pos()

    def wn_offset_from_synset_no_wn(self, synset: Synset) -> str:
        return str(synset.offset()).zfill(8) + synset.pos()

    def wn_offset_from_sense_key(self, sense_key: str) -> str:
        synset = wn.lemma_from_key(sense_key).synset()
        return self.wn_offset_from_synset(synset)

    def synset_from_offset(self, synset_offset: str) -> Synset:
        return wn.of2ss(synset_offset)

    def gloss_from_offset(self, synset_offset: str) -> str:
        return wn.of2ss(synset_offset).definition()

    def gloss_from_sense_key(self, sense_key: str) -> str:
        return wn.lemma_from_key(sense_key).synset().definition()

    def gloss_from_synset_name(self, synset_name: str) -> str:
        return wn.synset(synset_name).definition()

    def synset_from_sense_key(self, sense_key: str) -> Synset:
        return wn.lemma_from_key(sense_key).synset()

    def synsets_from_lemmapos(self, lemma: str, pos: str) -> List[Synset]:
        return wn.synsets(lemma, pos)

    def synset_from_name(self, name: str) -> Synset:
        return wn.synset(name)

    def wn_offsets_from_lemmapos(self, lemma: str, pos: str) -> List[str]:
        return [self.wn_offset_from_synset(syns) for syns in self.synsets_from_lemmapos(lemma, pos)]

    def lemmasname_from_synset(self, synset: Synset) -> List[str]:
        lemma_name = synset.lemma_names()
        lemma_name_lst = ' '.join(lemma_name).split(' ')
        return lemma_name_lst

    def name_from_synset(self, synset: Synset) -> str:
        return synset.name()

    def gloss_from_synset(self, synset: Synset) -> str:
        return synset.definition()

    def sensekey_from_synset_lemmaname(self, synset: Synset, lemmaname: str) -> str:
        for l in synset.lemmas():
            key = l.key()
            if lemmaname == key.split('%')[0].lower():
                return key

    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wn.ADJ,
                    "N": wn.NOUN,
                    "V": wn.VERB,
                    "R": wn.ADV}

        return tag_dict.get(tag, wn.NOUN)


if __name__ == "__main__":
    None
