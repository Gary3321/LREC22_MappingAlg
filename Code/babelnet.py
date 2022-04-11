import pandas as pd
from typing import List, Tuple, Optional, Set, Iterable
import warnings
import pickle
from tqdm import tqdm

warnings.filterwarnings("ignore", 'This pattern has match groups')


class Babelnet:
    def __init__(self, language, file_path, file_name):
        self.language = language
        self.path = '../Data/'
        self.out_file_path = '../Data/'
        self.name = 'bn_data.csv'
        self.lemma_lan_file = 'wn_lemmas_lan_dict'
        self.lemma_lan_file_entire_synsets = 'wn_entire_syn_lemmas_lan_dict'
        self.language_file = 'Clics_BN_lemmas_overlap_oriLan.csv'

    def read_bn_whole(self):
        df_bn = pd.read_csv(self.path + self.name)
        df_bn = df_bn[df_bn['BNLemma'].notna()]
        df_bn = df_bn[df_bn['BNLanguage'].notna()]
        print('total synsets:', len(set(df_bn['WNSynsetID'].to_list())))
        return df_bn

    def read_lemmas_pickle_entire_synsets(self):
        '''
        store all lemmas to a big dictionary, like {wn_offset:{lan1:[lemma1, lemma2, ...], lan2:[lemmas]..}
        :return:
        '''
        # languages shared with CLICS
        df_lan_overlap = pd.read_csv(self.out_file_path + self.language_file)
        wn_lans = df_lan_overlap['Lan'].to_list()

        df_bn = self.read_bn_whole()
        df_bn = df_bn[df_bn['BNLanguage'].isin(wn_lans)]

        wn_dict = {}
        wn_ids = list(set(df_bn['WNSynsetID'].to_list()))
        for wn_id in tqdm(wn_ids):
            df_bn_sub = df_bn[df_bn['WNSynsetID'] == wn_id]
            wn_id_lans = df_bn_sub['BNLanguage'].to_list()
            # wn_lans = list(set(df_bn_sub['BNLanguage'].to_list()))
            lan_dict = {}
            for lan in wn_id_lans:
                lemmas = df_bn_sub[df_bn_sub['BNLanguage'] == lan]['BNLemma'].values[0]
                lan_dict[lan] = lemmas.strip(';').split(';')
            lan_dict['Languages'] = wn_id_lans
            wn_dict[wn_id] = lan_dict

        # store the dictionary to a pickle file
        with open(self.out_file_path + self.lemma_lan_file_entire_synsets, 'wb') as pickled_file:
            pickle.dump(wn_dict, pickled_file)

    def read_lemmas_pickle(self):
        '''
        store all lemmas to a big dictionary, like {wn_offset:{lan1:[lemma1, lemma2, ...], lan2:[lemmas]..}
        :return:
        '''
        # languages shared with CLICS
        df_lan_overlap = pd.read_csv(self.out_file_path + self.language_file)
        wn_lans = df_lan_overlap['Lan'].to_list()

        df_bn = self.read_bn_whole()
        df_bn = df_bn[df_bn['BNLanguage'].isin(wn_lans)]

        wn_dict = {}
        wn_ids = list(set(df_bn['WNSynsetID'].to_list()))
        for wn_id in tqdm(wn_ids):
            df_bn_sub = df_bn[df_bn['WNSynsetID'] == wn_id]
            wn_id_lans = df_bn_sub['BNLanguage'].to_list()
            lan_dict = {}
            for lan in wn_id_lans:
                lemmas = df_bn_sub[df_bn_sub['BNLanguage'] == lan]['BNLemma'].values[0]
                lan_dict[lan] = lemmas.strip(';').split(';')
            lan_dict['Languages'] = wn_id_lans
            wn_dict[wn_id] = lan_dict

        # store the dictionary to a pickle file
        with open(self.out_file_path + self.lemma_lan_file, 'wb') as pickled_file:
            pickle.dump(wn_dict, pickled_file)

    def read_bn_by_lanlst(self, lan_lst):
        df_bn = self.read_bn_whole()
        df_bn_tr = df_bn[df_bn['BNLanguage'].isin(lan_lst)]
        df_bn_tr = df_bn_tr[df_bn_tr['BNLemma'].notna()]
        return df_bn_tr

    def read_bn_by_lanlst_quick(self, lan_lst, df_bn):
        df_bn_tr = df_bn[df_bn['BNLanguage'].isin(lan_lst)]
        df_bn_tr = df_bn_tr[df_bn_tr['BNLemma'].notna()]
        return df_bn_tr

    def read_bn_by_single_lan(self, language, df_bn):
        df_bn_tr = df_bn[df_bn['BNLanguage'] == language]
        df_bn_tr = df_bn_tr[df_bn_tr['BNLemma'].notna()]
        if language == 'EN':
            df_bn_tr['BNLemma'] = df_bn_tr['BNLemma'].str.replace('_', ' ')
        return df_bn_tr

    def extract_synsets(self, keyword, pos, df_bn_tran):
        '''
        extracting synsets from BN by giving keyword, pos
        Return:
        a list of WN synset id
        '''
        try:
            reg_exp = '(^|;)' + keyword.lower() + ';'  # remove lower
            synsets = df_bn_tran[df_bn_tran['BNLemma'].str.lower().str.contains(reg_exp)][
                'WNSynsetID'].to_list()

        except Exception as e:
            print('keyword', keyword)
            print('reg_exp:', reg_exp)
            print('error:', e)
        syn_pos = []
        for s in synsets:
            if s[-1] == pos:
                syn_pos.append(s)
        return syn_pos

    def extract_synsets_withoutpos(self, keyword, df_bn_tran):
        '''
        extracting synsets from BN by giving keyword, pos
        Return:
        a list of WN synset id
        '''
        try:
            keyword = keyword.replace('(', '\(').replace(')', '\)')
            reg_exp = '(^|;)' + keyword.lower() + ';'
            synsets = df_bn_tran[df_bn_tran['BNLemma'].str.lower().str.contains(reg_exp)][
                'WNSynsetID'].to_list()

        except Exception as e:
            print('keyword', keyword)
            print('reg_exp:', reg_exp)
            print('error:', e)
            # synsets = []

        return synsets

    def extract_bn_lemmas(self, df_bn_tran, wn_synset_id: List) -> List[str]:
        '''
        extract BN lemmas by given a list wn synset ids
        :param wn_synset_id:  a list
        :param df_bn_tran: BN dataframe
        :return:
        '''
        bn_lemmas_lst = []
        for wn_id in wn_synset_id:
            bn_lemma = df_bn_tran[df_bn_tran['WNSynsetID'] == wn_id]['BNLemma'].to_list()
            if len(bn_lemma) == 1:
                bn_lemmas = bn_lemma[0].strip(';').split(';')
            elif len(bn_lemma) == 0:
                bn_lemmas = []
            else:
                print('Error: more than one row related the same WN offset:', wn_id)
                bn_lemmas = []
            bn_lemmas_lst = bn_lemmas_lst + bn_lemmas
        return bn_lemmas_lst

    def extract_bn_lemmas_wnid(self, df_bn_tran, wn_synset_id: str) -> List[str]:
        '''
        extract BN lemmas by given a wn synset offset, from one language
        :param wn_synset_id:  a string
        :param df_bn_tran: BN dataframe
        :return:
        '''
        bn_lemma = df_bn_tran[df_bn_tran['WNSynsetID'] == wn_synset_id]['BNLemma'].to_list()
        if len(bn_lemma) == 1:
            bn_lemmas = bn_lemma[0].lower().strip(';').split(';')
        elif len(bn_lemma) == 0:
            bn_lemmas = []
        else:
            print('Error: more than one row related the same WN offset:', wn_synset_id)
            bn_lemmas = []
        return bn_lemmas

    def extract_bn_lemmas_wnid_multi_lans(self, df_bn_tran, wn_synset_id: str) -> List[str]:
        '''
        extract BN lemmas by given a wn synset offset, from more than one language
        :param wn_synset_id:  a string
        :param df_bn_tran: BN dataframe
        :return:
        '''
        bn_lemma = df_bn_tran[df_bn_tran['WNSynsetID'] == wn_synset_id]['BNLemma'].to_list()
        bn_lemmas = []
        for blemma in bn_lemma:
            bn_lemmas = bn_lemmas + blemma.strip(';').split(';')
        return bn_lemmas

    def extract_bn_lemmas_by_lan(self, df_bn, language):
        '''
        obtain lemmas by given language
        :param df_bn: the whole BN data
        :param language:
        :return:
        '''
        df_bn_tran = df_bn[df_bn['BNLanguage'] == language]
        df_bn_tran = df_bn_tran[df_bn_tran['BNLemma'].notna()]
        bn_lemma = df_bn_tran['BNLemma'].to_list()

        bn_lemma_whole = ''.join(bn_lemma)
        bn_lemmas = bn_lemma_whole.strip(';').split(';')
        return set(bn_lemmas)

    def bn_lans_dict(self):
        '''
        :return: dictionary, language name: abb
        '''
        # read BN languages name and abb.
        with open('../Data/' + 'BN_language.txt', 'r') as bf:
            bn_lan_name_abb = bf.readlines()
        bn_lan_name = []
        for i in range(len(bn_lan_name_abb)):
            if i % 2 != 0:
                bn_lan_name.append(bn_lan_name_abb[i].lower().strip('\n').strip())

        bn_lan_abb = []
        for i in range(len(bn_lan_name_abb)):
            if i % 2 == 0:
                bn_lan_abb.append(bn_lan_name_abb[i].strip('\n'))

        bn_lan_dict = {}  # name - abb
        for i in range(len(bn_lan_name)):
            bn_lan_dict[bn_lan_name[i]] = bn_lan_abb[i]
        return bn_lan_dict


if __name__ == '__main__':
    None
