import pandas as pd
from wordnet import wordnet
from babelnet import Babelnet
from tqdm import tqdm
import ast
import pickle


class OmegaWiki:
    def __init__(self):
        self.file_path = "../Data/"
        self.bn_file = 'bn_data.csv'
        self.gold_file = "WN_OW_alignment_gold.csv"
        self.out_path = "../Data/"
        self.lemma_lan_file = 'ow_lemmas_lan_dict'

    def generate_wn_om_data(self):
        '''
        statistics of the whole gold data: WordNet 215; OmegaWiki 4968
        only consider the identical glosses between two versions:
        Gold_sub: 686, WN: 148, OM: 3248
        cosider identical + containing: Gold_sub: 790, WN: 167, OM: 3839
        :return:
        '''
        wn_obj = wordnet()
        df_gold = pd.read_csv(self.file_path + self.gold_file)

        # identical sense ids
        ow_gold = self.comapre_sense(df_gold=df_gold)
        df_gold_sub = df_gold[df_gold['OW_ID'].isin(ow_gold)]
        df_om_data = self.generate_om_data(om_gold_id=ow_gold, df_ow_gold=df_gold)

        wn_gold = list(set(df_gold[df_gold['ALIGN'] == 1]['WN_OFFSET'].to_list()))
        # wn_gold = list(set(df_gold['WN_OFFSET'].to_list()))
        df_wn_data = self.generate_wn_data(wn_gold_offset=wn_gold, wn_obj=wn_obj,
                                           df_ow_gold=df_gold, om_gold_id=ow_gold)

        df_gold_sub.to_csv(self.file_path + 'WN_OW_alignment_modified_sub_contain.csv', index=False)
        df_wn_data.to_csv(self.file_path + 'wn_data_from_gold_contain.csv', index=False)
        df_om_data.to_csv(self.file_path + 'om_data_from_gold_contain.csv', index=False)
        print(f'Gold_sub: {len(df_gold_sub)}, WN: {len(df_wn_data)}, OM: {len(df_om_data)}')

    def generate_all_wn_data(self):
        df_wn = pd.DataFrame(columns=['wn_offset', 'pos', 'synset', 'gloss', 'lemmas', 'ow_id', 'ow_id_ties'])
        wn_obj = wordnet()
        df_gold = pd.read_csv(self.file_path + self.gold_file)
        # identical sense ids
        ow_gold = self.comapre_sense(df_gold=df_gold)
        df_gold_sub = df_gold[df_gold['OW_ID'].isin(ow_gold)]
        wn_gold_offset = list(set(df_gold_sub['WN_OFFSET'].to_list()))

        for offset in wn_gold_offset:
            wn_offset = 'wn:' + offset.replace('-', '')
            pos = offset.split('-')[1]
            synset = wn_obj.synset_from_offset(offset.replace('-', ''))
            synset_name = wn_obj.name_from_synset(synset)
            gloss = wn_obj.gloss_from_synset_name(synset_name)
            lemma_name_lst = wn_obj.lemmasname_from_synset(synset)

            # get OW id; pick the first one for 1-many mapping
            ow_id_lst = []
            ow_id = df_gold_sub[(df_gold_sub['WN_OFFSET'] == offset)]['OW_ID'].to_list()
            for owid in ow_id:
                if owid in ow_gold:
                    ow_id_lst.append(owid)
            if len(ow_id_lst) > 0:
                # print(wn_offset, synset, gloss, lemma_name_lst)
                df_wn = df_wn.append({'wn_offset': wn_offset, 'pos': pos, 'synset': synset_name, 'gloss': gloss,
                                      'lemmas': str(lemma_name_lst), 'ow_id': ow_id_lst[0],
                                      'ow_id_ties': str(ow_id_lst)}, ignore_index=True)

        df_wn.to_csv(self.file_path + 'wn_data_from_gold_all.csv', index=False)

    def generate_wn_data(self, wn_gold_offset, wn_obj, df_ow_gold, om_gold_id):
        df_wn = pd.DataFrame(columns=['wn_offset', 'pos', 'synset', 'gloss', 'lemmas', 'ow_id', 'ow_id_ties'])
        for offset in wn_gold_offset:
            wn_offset = 'wn:' + offset.replace('-', '')
            pos = offset.split('-')[1]
            synset = wn_obj.synset_from_offset(offset.replace('-', ''))
            synset_name = wn_obj.name_from_synset(synset)
            gloss = wn_obj.gloss_from_synset_name(synset_name)
            lemma_name_lst = wn_obj.lemmasname_from_synset(synset)

            # get OW id; pick the first one for 1-many mapping
            ow_id_lst = []
            ow_id = df_ow_gold[(df_ow_gold['WN_OFFSET'] == offset) & (df_ow_gold['ALIGN'] == 1)]['OW_ID'].to_list()
            for owid in ow_id:
                if owid in om_gold_id:
                    ow_id_lst.append(owid)
            if len(ow_id_lst) > 0:
                # print(wn_offset, synset, gloss, lemma_name_lst)
                df_wn = df_wn.append({'wn_offset': wn_offset, 'pos': pos, 'synset': synset_name, 'gloss': gloss,
                                      'lemmas': str(lemma_name_lst), 'ow_id': ow_id_lst[0],
                                      'ow_id_ties': str(ow_id_lst)}, ignore_index=True)
        return df_wn

    def generate_om_data(self, om_gold_id, df_ow_gold):
        pos_file = 'om_pos.tab'
        language_file = 'om_languages.tab'
        en_file_2 = 'om_term_senses_en2.tab'
        translation_file_2 = 'om_term_senses_translations2.tab'

        df_pos = pd.read_csv(self.file_path + pos_file, sep='\t')
        df_lan = pd.read_csv(self.file_path + language_file, sep='\t')
        df_en2 = pd.read_csv(self.file_path + en_file_2, sep='\t')  # 64759
        df_trans2 = pd.read_csv(self.file_path + translation_file_2, sep='\t')

        df_en2 = df_en2[df_en2['ow_gloss'].notna()]
        df_en2['ow_id'] = pd.to_numeric(df_en2['ow_id'])
        df_ow = df_en2[df_en2['ow_id'].isin(om_gold_id)]
        # get gloss
        df_ow_gloss = df_ow.drop_duplicates(['ow_id'])

        df_ow_data = pd.DataFrame(
            columns=['ow_id', 'ow_term', 'ow_pos', 'ow_gloss', 'ow_gloss_id', 'ow_trans', 'ow_lan_id',
                     'ow_lan_name'])
        df_data = df_ow_gloss.copy()
        df_data.reset_index(inplace=True)

        for i in range(len(df_data)):
            ow_id = df_data['ow_id'][i]
            ow_term = df_data['ow_term'][i]
            ow_gloss = df_data['ow_gloss'][i]
            ow_gloss_id = df_data['ow_gloss_id'][i]

            # get pos; have tested that ow_id is associated with only one pos
            om_term_pos = df_ow_gold[df_ow_gold['OW_ID'] == ow_id]['OW_TERMID'].to_list()
            pos_lst = []
            for termpos in om_term_pos:
                pos_tmp = termpos.split('#')[2]
                if pos_tmp in ['a', 'n', 'v', 'r']:
                    pos_lst.append(pos_tmp)

            try:
                ow_pos = pos_lst[0]
            except:
                ow_pos = pos_tmp

            lan_ids = set(df_trans2[df_trans2['ow_id'] == ow_id]['language_id'].to_list())
            for lanid in lan_ids:
                lemma_lst = df_trans2[(df_trans2['ow_id'] == ow_id) &
                                      (df_trans2['language_id'] == lanid)]['ow_term'].to_list()
                lemmas = ';'.join(lemma_lst)
                lan_name = df_lan[df_lan['language_id'] == lanid]['language_name'].values[0]
                df_ow_data = df_ow_data.append({'ow_id': ow_id, 'ow_term': ow_term, 'ow_pos': ow_pos,
                                                'ow_gloss': ow_gloss,
                                                'ow_gloss_id': ow_gloss_id, 'ow_trans': lemmas,
                                                'ow_lan_id': lanid, 'ow_lan_name': lan_name}, ignore_index=True)

        return df_ow_data

    def comapre_sense(self, df_gold):
        '''
        return the identical sense ids with the current OM version
        :param df_gold:
        :return:
        '''
        ok = 0
        wrong = 0
        count = 0
        mayok = 0
        missed_id = []
        wrong_ids = []
        identical_ids = []
        ow_ids = []
        en_file_2 = 'om_term_senses_en2.tab'
        df_en2 = pd.read_csv(self.file_path + en_file_2, sep='\t')

        for i in range(len(df_gold)):
            ow_id = df_gold['OW_ID'][i]
            if ow_id in ow_ids:
                continue
            else:
                ow_ids.append(ow_id)

                count = count + 1
                ow_term = df_gold['#TERM'][i]
                ow_gloss_gold = df_gold['OW_GLOSS'][i]
                ow_gloss_gold = ow_gloss_gold.split('#')[0]
                ow_gloss_gold = ow_gloss_gold.strip('"').strip('“')
                try:
                    ow_gloss_current = df_en2[df_en2['ow_id'] == str(ow_id)]['ow_gloss'].values[0]
                except:
                    missed_id.append(ow_id)

                if ow_gloss_gold == ow_gloss_current:
                    ok = ok + 1
                    identical_ids.append(ow_id)

        return identical_ids

    def cal_overlap_bn_om(self):

        # get bn data

        bn_obj = Babelnet(language='', file_path=self.bn_file, file_name=self.bn_file)

        df_bn = bn_obj.read_bn_whole()

        bn_lans_dict = bn_obj.bn_lans_dict()
        bn_lans_dict['mandarin (simplified)'] = 'ZH'  # manually add Chinese

        om_lans, df_om_data = self.get_languages()
        # languages are shared by OM and bn
        om_bn_lans = set(om_lans) & set(bn_lans_dict.keys())

        df_om_bn_lan_overlap = pd.DataFrame(columns=['Languages', 'Lan', 'Shared_lemmas_num'])

        for obl in tqdm(om_bn_lans):
            om_lemmas = self.get_lemmas_by_lan(df_om_data, obl)
            bn_lemmas = bn_obj.extract_bn_lemmas_by_lan(df_bn, bn_lans_dict[obl])
            overlap_size = len(om_lemmas & bn_lemmas)
            df_om_bn_lan_overlap = df_om_bn_lan_overlap.append({'Languages': obl, 'Lan': bn_lans_dict[obl],
                                                                'Shared_lemmas_num': overlap_size},
                                                               ignore_index=True)

        df_om_bn_lan_overlap.to_csv(self.file_path + 'OM_BN_languages_overlap.csv', index=False)

    def get_languages(self):
        df_om_data = pd.read_csv(self.file_path + 'om_data_from_gold.csv')
        om_languages = set(df_om_data['ow_lan_name'].to_list())
        om_lans_lst = [lan.lower().strip() for lan in om_languages]
        return om_lans_lst, df_om_data

    def get_lemmas_by_lan(self, df_om_data, language):
        df_om_lan = df_om_data[df_om_data['ow_lan_name'].str.lower() == language.lower()]
        df_om_lan = df_om_lan[df_om_lan['ow_trans'].notna()]
        om_lemma = df_om_lan['ow_trans'].to_list()
        om_lemmas = ';'.join(om_lemma).strip(';').split(';')
        return set(om_lemmas)

    def get_ids_by_targetlist(self, target_list, df_om_lans):
        '''
        giving a target lemmas list, extract OM ids whose lemmas  have overlap with the list
        :param target_list:  WN lemmas
        :param df_om_lans: omegawiki data in given languages
        '''
        om_ids = []
        for target in target_list:
            target = target.replace('(', '\(').replace(')', '\)')

            reg_exp1 = '(^|;)' + target + ';?'
            om_ids = om_ids + df_om_lans[df_om_lans['ow_trans'].str.contains(reg_exp1)]['ow_id'].to_list()

        om_target_ids = list(set(om_ids))

        return om_target_ids

    def get_lemmas_by_id(self, df_om_trans, sense_id):
        '''
        get lemmas in df_om_trans
        :param sense_ids:
        :param df_om_trans:
        :return:
        '''
        om_trans = df_om_trans[df_om_trans['ow_id'] == sense_id]['ow_trans'].to_list()
        lemmas = []
        for trans in om_trans:
            lemmas = lemmas + trans.split(';')
        om_lemmas = list(set(lemmas))
        return om_lemmas

    def read_om_by_lans_quick(self, language_lst, df_om):
        df_om_lans = df_om[df_om['ow_lan_name'].str.lower().isin(language_lst)]
        return df_om_lans

    def generate_all_ow_data(self):
        '''
        pos: noun, adverb, adjective, verb
        :return:
        '''
        pos_file = 'om_pos.tab'
        language_file = 'om_languages.tab'
        en_file_2 = 'om_term_senses_en2.tab'
        translation_file_2 = 'om_term_senses_translations2.tab'

        df_pos = pd.read_csv(self.file_path + pos_file, sep='\t')
        df_lan = pd.read_csv(self.file_path + language_file, sep='\t')
        df_en2 = pd.read_csv(self.file_path + en_file_2, sep='\t')  # 64759
        df_trans2 = pd.read_csv(self.file_path + translation_file_2, sep='\t')

        df_trans2 = df_trans2[df_trans2['ow_term'].notna()]

        df_en2 = df_en2[df_en2['ow_gloss'].notna()]
        df_en2['ow_id'] = pd.to_numeric(df_en2['ow_id'])
        # get gloss
        df_ow_gloss = df_en2.drop_duplicates(['ow_id'])

        df_ow_data = pd.DataFrame(
            columns=['ow_id', 'ow_term', 'ow_pos', 'ow_gloss', 'ow_gloss_id', 'ow_trans', 'ow_lan_id',
                     'ow_lan_name'])
        df_data = df_ow_gloss.copy()
        df_data.reset_index(inplace=True)

        for i in tqdm(range(len(df_data))):
            ow_id = df_data['ow_id'][i]
            ow_term = df_data['ow_term'][i]
            ow_gloss = df_data['ow_gloss'][i]
            ow_gloss_id = df_data['ow_gloss_id'][i]

            # get pos; have tested that ow_id is associated with only one pos
            om_term_pos = df_pos[df_pos['ow_id'] == ow_id]['pos'].to_list()
            ow_pos = list(set(om_term_pos))

            lan_ids = set(df_trans2[df_trans2['ow_id'] == ow_id]['language_id'].to_list())
            for lanid in lan_ids:
                lemma_lst = df_trans2[(df_trans2['ow_id'] == ow_id) &
                                      (df_trans2['language_id'] == lanid)]['ow_term'].to_list()
                lemmas = ';'.join(lemma_lst)
                lan_name = df_lan[df_lan['language_id'] == lanid]['language_name'].values[0]
                df_ow_data = df_ow_data.append({'ow_id': ow_id, 'ow_term': ow_term, 'ow_pos': ow_pos,
                                                'ow_gloss': ow_gloss,
                                                'ow_gloss_id': ow_gloss_id, 'ow_trans': lemmas,
                                                'ow_lan_id': lanid, 'ow_lan_name': lan_name}, ignore_index=True)

        df_ow_data.to_csv(self.out_path + 'OW_data_all.csv')

    def general_picle_lemmas(self):

        df_ow_data = pd.read_csv(self.out_path + 'OW_data_all.csv')
        ow_dict = {}
        ow_ids_all = set(df_ow_data['ow_id'].to_list())
        for owid in tqdm(ow_ids_all):
            df_ow_sub = df_ow_data[df_ow_data['ow_id'] == owid]
            ow_id_lans = df_ow_sub['ow_lan_name'].to_list()
            ow_id_pos = df_ow_sub['ow_pos'].values[0]
            ow_id_pos = ast.literal_eval((ow_id_pos))  # list
            lan_dict = {}
            for lan in ow_id_lans:
                lemmas = df_ow_sub[df_ow_sub['ow_lan_name'] == lan]['ow_trans'].values[0]
                lan_dict[lan] = lemmas.strip(';').split(';')
            lan_dict['Languages'] = ow_id_lans
            lan_dict['pos'] = ow_id_pos
            ow_dict[owid] = lan_dict

        # store the dictionary to a pickle file
        with open(self.out_path + self.lemma_lan_file, 'wb') as pickled_file:
            pickle.dump(ow_dict, pickled_file)


if __name__ == "__main__":
    None
