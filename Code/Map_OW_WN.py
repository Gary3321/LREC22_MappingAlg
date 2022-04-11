from MappingMethods import Mapping
from wordnet import wordnet
import pandas as pd
from Map_Clics_Wn import MapClicsWn
import pickle
import sys
from tqdm import tqdm
import random
import nltk
from evaluation import Evaluation
from babelnet import Babelnet
from omegawiki import OmegaWiki


class MapOWWn:
    def __init__(self):
        self.input_path = '../Data/'
        self.gold_sub_file = 'WN_OW_alignment_modified_sub.csv'
        self.wn_file = 'wn_data_from_gold_all.csv'
        self.om_file = 'om_data_from_gold.csv'
        self.om_all_file = 'OW_data_all.csv'
        self.ow_dict_file = 'ow_lemmas_lan_dict'

        self.ovalsim = 'WordVote'
        self.mvotesim = 'LangVote'
        self.pickle_file = '_ow_wn_allpairs_ori_old7lan_dict'

        self.bn_language = ['EN', 'NL', 'RO', 'DE', 'IT', 'ID', 'GA']
        self.ow_lan = ['English', 'Dutch', 'Romanian', 'German', 'Italian', 'Indonesian', 'Irish']

        self.mapping_obj = Mapping()
        self.mapping_all_file_name = '_all_mapping_old7lan_results.csv'
        self.wn_obj = wordnet()
        self.map_obj = MapClicsWn()
        self.wn_dict = self.map_obj.load_bn_dict(file_path=self.map_obj.input_path, file_name=self.map_obj.wn_dict_file)
        self.ow_dict = self.load_embd_file(self.input_path + self.ow_dict_file)
        self.tb_random = 'random'
        self.tb_norma = 'normA'
        self.tb_normb = 'normB'
        self.tb_name_sim = 'name_sim'
        self.tb_synset_size = 'synsetSize'
        self.tb_namesim_synsize = 'namesim_synsize'
        self.mapping_onetoone_file_name = '_all_mapping_old7lan_onetoone.csv'
        self.mapping_cmp_gold_file_name = '_mapping_gold_old7lan_cmp.csv'

    def map_ow_wn(self, mapping_obj, source_dict, target_dict, source_lan_list, target_lan_list,
                  method_name, bn_obj, df_bn, om_obj, map_obj, wn_obj, lower_flag=False):
        '''
        map from Clics to WN
        :param mapping_obj:
        :param target_dict: wn_dict
        :param source_dict: clics_dict
        :return:
        '''

        df_om_data = pd.read_csv(self.input_path + self.om_file)
        df_bn_trans = bn_obj.read_bn_by_lanlst_quick(lan_lst=target_lan_list, df_bn=df_bn)

        target_cpt_ids = list(target_dict.keys())
        source_cpt_ids = list(source_dict.keys())
        # read gold data
        df_gold = self.read_gold_data()
        gold_cpt_ids = df_gold['wn_offset'].to_list()
        print('missing concepts:', set(gold_cpt_ids) - set(target_cpt_ids))
        gold_cpt_ids = set(gold_cpt_ids) & set(target_cpt_ids)
        print('total gold data:', len(gold_cpt_ids))

        ow_gold_ids = list(set(df_gold['OW_ID'].to_list()))
        df_ow_all = pd.read_csv(self.input_path + self.om_all_file)

        cnt = 0
        pairs_dict = {}

        for source_id in tqdm(source_cpt_ids):
            cnt = cnt + 1

            ow_term = df_ow_all[df_ow_all['ow_id'] == source_id]['ow_term'].values[0]
            ow_pos_tmp = df_ow_all[df_ow_all['ow_id'] == source_id]['ow_pos'].values[0]
            if 'noun' in ow_pos_tmp:
                ow_pos = 'n'
            elif 'adjective' in ow_pos_tmp:
                ow_pos = 'a'
            elif 'verb' in ow_pos_tmp:
                ow_pos = 'v'
            elif 'adverb' in ow_pos_tmp:
                ow_pos = 'r'
            else:
                continue

            wn_candidas = wn_obj.wn_synsets_from_word_pos(word=ow_term, pos=ow_pos)

            ow_lemmas = om_obj.get_lemmas_by_id(df_om_trans=df_om_data, sense_id=source_id)
            for olemma in ow_lemmas:
                wn_synsets = bn_obj.extract_synsets_withoutpos(keyword=olemma,
                                                               df_bn_tran=df_bn_trans)  # a list of WN ids
                wn_candidas = wn_candidas + wn_synsets
                wn_candidas = wn_candidas + wn_obj.wn_synsets_from_word_pos(word=olemma, pos=ow_pos)

            wn_candidas = list(set(wn_candidas) & set(target_cpt_ids))

            target_candidates = list(set(wn_candidas))  # remove duplicated candidates

            random.Random(2021).shuffle(target_candidates)

            for target_id in target_candidates:
                pair_key = str(source_id) + '-' + target_id
                if method_name == self.ovalsim:
                    # start = time.time()
                    score, _ = mapping_obj.OvalSimilarity(cpt_source=source_id,
                                                          cpt_dict_source=source_dict,
                                                          cpt_target=target_id,
                                                          cpt_dict_target=target_dict,
                                                          lan_list_source=source_lan_list,
                                                          lan_list_target=target_lan_list,
                                                          lower_flag=lower_flag)

                    if score > 0:
                        pairs_dict[pair_key] = score


                elif method_name == self.mvotesim:
                    score, _ = mapping_obj.MvoteSimilarity(cpt_source=source_id,
                                                           cpt_dict_source=source_dict,
                                                           cpt_target=target_id,
                                                           cpt_dict_target=target_dict,
                                                           lan_list_source=source_lan_list,
                                                           lan_list_target=target_lan_list,
                                                           lower_flag=lower_flag)
                    if score > 0:
                        pairs_dict[pair_key] = score

                else:
                    score = 0
                    pairs_dict[pair_key] = score

                    print('Method name is invalid')
                    sys.exit(1)

        pair_file_name = method_name + self.pickle_file
        # store sorted_pair_dict to pickle file

        with open(self.input_path + pair_file_name, 'wb') as pickled_file:
            pickle.dump(pairs_dict, pickled_file)

    def read_gold_data(self):
        df_gold = pd.read_csv(self.input_path + self.gold_sub_file)
        df_gold['wn_offset'] = 'wn:' + df_gold['WN_OFFSET'].str.replace('-', '')

        return df_gold

    def test_tie_breaking_method(self, method_name, wn_obj):
        # method_name = self.ovalsim
        df_concept = self.read_gold_data()
        pair_dict = self.read_pair_dict(method_name=method_name)
        pair_dict = self.add_names_dict(pair_dict=pair_dict, wn_obj=wn_obj,
                                        df_concept=df_concept)
        # sort first by similarity score in descending order, then by concept pair name in ascending
        sorted_pair = sorted(pair_dict.items(), key=lambda kv: (-kv[1], kv[0]))

        df_mapping_result = self.generate_mapping_result(pairs_list=sorted_pair)

        df_mapping_result.to_csv(self.input_path + method_name + 'ow_wn_mapping_result.csv', index=False)
        return df_mapping_result

    def read_pair_dict(self, method_name):
        with open(self.input_path + method_name + self.pickle_file, 'rb') as pickled_file:
            pairs_lst = pickle.load(pickled_file)
        return pairs_lst

    def add_names_dict(self, pair_dict, wn_obj, df_concept):
        cpt_name_pair_dict = {}
        for key, value in tqdm(pair_dict.items()):
            cpt_name_pair_dict[self.convert_id_name(concept_pair=key, wn_obj=wn_obj,
                                                    df_concept=df_concept)] = value
        return cpt_name_pair_dict

    def generate_mapping_result(self, pairs_list):
        df_mapping_result = pd.DataFrame(columns=['wn_offset', 'ow_id', 'similarity'])
        used_ows = set()
        used_wn = set()
        for cpt_pair in pairs_list:
            pair = cpt_pair[0].split(';;')
            ow_wn = pair[1].split('-')
            ow_id = int(ow_wn[0])
            wn_id = ow_wn[1]
            if wn_id not in used_wn and ow_id not in used_ows:
                # if clics_id not in used_clics:
                df_mapping_result = df_mapping_result.append({'wn_offset': wn_id,
                                                              'ow_id': ow_id,
                                                              'similarity': cpt_pair[1]}, ignore_index=True)
                used_ows.add(ow_id)
                used_wn.add(wn_id)

        return df_mapping_result

    def convert_id_name(self, concept_pair, wn_obj, df_concept):
        cpt_pairs = concept_pair.split('-')
        cpt_id = int(cpt_pairs[0])
        wn_offset = cpt_pairs[1]
        wn_synset = wn_obj.name_from_synset(wn_obj.synset_from_offset(wn_offset.replace('wn:', '')))
        wn_items = wn_synset.split('.')

        wn_sense_num = wn_items[2]
        wn_name = wn_items[0]

        ow_name = df_concept[df_concept['OW_ID'] == cpt_id]['OW_TERMID'].values[0]

        sorting = self.sort_ties_concept_name(ow_name=ow_name, wn_name=wn_name)
        distance = sorting

        # using sense number
        sorting = self.sort_ties_sense_num(wn_sense_num=wn_sense_num)
        # combine edit distance and sense number
        sorting = str(distance) + str(sorting)

        converted_cpt_pair = ow_name + str(sorting) + wn_name + ';;' + concept_pair
        return converted_cpt_pair

    def sort_ties_concept_name(self, ow_name, wn_name):
        if nltk.edit_distance(ow_name, wn_name) == 0:
            return 0
        else:
            return 9

    def sort_ties_sense_num(self, wn_sense_num):
        return wn_sense_num

    def load_embd_file(self, dict_file):
        '''
        load the dict files
        :param dict_file:
        :return:
        '''
        with open(dict_file, 'rb') as pickled_file:
            dict_obj = pickle.load(pickled_file)
        return dict_obj

    def read_allpairs_dict_pickle_csv(self, method_name):
        with open(self.input_path + method_name + self.pickle_file, 'rb') as pickled_file:
            pairs_dict = pickle.load(pickled_file)

        sorted_pair = sorted(pairs_dict.items(), key=lambda kv: (-kv[1], kv[0]))
        df_mapping_result_all = pd.DataFrame(columns=['ow_id', 'wn_offset', 'similarity', 'random_number',
                                                      'editing_dis', 'synset_size', 'ow_cpt_size',
                                                      'editing_synsize'])
        # get a random shuffled list for breaking ties with random number
        random_lst = [i for i in range(len(sorted_pair))]
        random.Random(2021).shuffle(random_lst)
        i = 0
        df_ow_all = pd.read_csv(self.input_path + self.om_all_file)
        for item in tqdm(sorted_pair):
            ow_id = int(item[0].split('-')[0])
            wn_offset = item[0].split('-')[1]
            similary_score = item[1]

            ow_term = df_ow_all[df_ow_all['ow_id'] == ow_id]['ow_term'].values[0]

            wn_synset = self.wn_obj.name_from_synset(self.wn_obj.synset_from_offset(wn_offset.replace('wn:', '')))
            wn_items = wn_synset.split('.')
            wn_name = wn_items[0]
            name_edit = nltk.edit_distance(ow_term, wn_name)
            synset_size = self.get_wn_synset_size(wn_offset)
            ow_cpt_size = self.get_ow_cpt_size(ow_id)
            editing_synsize = str(name_edit).zfill(3) + str(synset_size)
            random_num = random_lst[i]
            i = i + 1

            df_mapping_result_all = df_mapping_result_all.append({'ow_id': ow_id, 'wn_offset': wn_offset,
                                                                  'similarity': similary_score,
                                                                  'random_number': random_num,
                                                                  'editing_dis': name_edit,
                                                                  'synset_size': synset_size,
                                                                  'ow_cpt_size': ow_cpt_size,
                                                                  'editing_synsize': editing_synsize},
                                                                 ignore_index=True)

        df_mapping_result_all.to_csv(self.input_path + method_name + self.mapping_all_file_name,
                                     index=False)

        print('total mapping pairs:', len(sorted_pair))

    def get_wn_synset_size(self, wn_tie):
        size = 0
        for lan in self.bn_language:
            size = size + len(self.mapping_obj.get_lex(cpt_name=wn_tie, cpt_dict=self.wn_dict,
                                                       language=lan))
        return size

    def get_ow_cpt_size(self, ow_tie):
        size = 0
        for lan in self.ow_lan:
            size = size + len(self.mapping_obj.get_lex(cpt_name=ow_tie, cpt_dict=self.ow_dict,
                                                       language=lan))
        return size

    def map_wn_ow_full(self, method_name, sort_method):

        eval_obj = Evaluation(file_path='')

        df_mapping_result_all = pd.read_csv(self.input_path + method_name + self.mapping_all_file_name)

        print('all mapping result:', len(df_mapping_result_all))
        # sorting
        df_mapping_result_all['editing_synsize1'] = 'a' + df_mapping_result_all['editing_dis'].astype(str).str.zfill(
            3) + df_mapping_result_all['synset_size'].astype(str)

        df_mapping_result_all['editing_synsize2'] = 'a' + df_mapping_result_all['editing_dis'].astype(str).str.zfill(
            3) + df_mapping_result_all['ow_cpt_size'].astype(str)

        sort_column = ''
        if sort_method == self.tb_random:
            sort_column = 'random_number'
        elif sort_method == self.tb_namesim_synsize:
            sort_column = 'editing_synsize2'
        elif sort_method == self.tb_synset_size:
            sort_column = 'ow_cpt_size'
        print('sort_column:', sort_column)
        df_mapping_result_all = df_mapping_result_all.sort_values(['similarity', 'wn_offset', sort_column],
                                                                  ascending=[False, True, True])

        df_mapping_result_all.reset_index(inplace=True)
        # df_mapping_result_all.to_csv(self.output_path + method_name + sort_method + '_sorted_mapping.csv')

        # get one-to-one mapping
        df_onetoone_mapping = pd.DataFrame(columns=['ow_id', 'wn_offset', 'similarity'])
        used_ow = set()
        used_wn = set()

        for i in tqdm(range(len(df_mapping_result_all))):
            ow_id = df_mapping_result_all['ow_id'][i]
            wn_offset = df_mapping_result_all['wn_offset'][i]
            sim_score = df_mapping_result_all['similarity'][i]
            if ow_id not in used_ow and wn_offset not in used_wn:
                df_onetoone_mapping = df_onetoone_mapping.append({'ow_id': ow_id,
                                                                  'wn_offset': wn_offset,
                                                                  'similarity': sim_score}, ignore_index=True)
                used_ow.add(ow_id)
                used_wn.add(wn_offset)

        df_onetoone_mapping.to_csv(self.input_path + method_name + sort_method + self.mapping_onetoone_file_name,
                                   index=False)

        print('all one-to-one mapping:', len(df_onetoone_mapping))
        # evaluate with gold data
        df_sub_gold = self.read_gold_data()
        print('gold pairs:', len(df_sub_gold))

        df_final_result = df_onetoone_mapping.merge(df_sub_gold, how='inner', left_on='wn_offset', right_on='wn_offset')

        df_final_result.to_csv(self.input_path + method_name + sort_method + self.mapping_cmp_gold_file_name,
                               index=False)

        _, _ = eval_obj.analysis_ow(df_result=df_onetoone_mapping,
                                    target_lan_lst=['EN'],
                                    gold_col_name="ow_id", mapped_col_name='ow_id',
                                    mapped_unique_col_rand='ow_id',
                                    mapped_unique_col_sim='ow_id',
                                    df_gold_data=df_sub_gold, threshold='')

    def run_map_ow_wn(self):
        bn_file_path = '../Data/'
        bn_file_name = 'bn_data.csv'
        bn_obj = Babelnet(language='', file_path=bn_file_path, file_name=bn_file_name)
        df_bn = bn_obj.read_bn_whole()

        owwn_obj = MapOWWn()
        wn_obj = wordnet()
        om_obj = OmegaWiki()

        mapping_obj = Mapping()
        map_obj = MapClicsWn()

        method_names = [owwn_obj.ovalsim, owwn_obj.mvotesim]

        target_dict = map_obj.load_bn_dict(file_path=map_obj.input_path, file_name=map_obj.wn_dict_file)
        source_dict = owwn_obj.load_embd_file(owwn_obj.input_path + owwn_obj.ow_dict_file)

        target_lan_list = ['EN', 'NL', 'RO', 'DE', 'IT', 'ID', 'GA']
        source_lan_list = ['English', 'Dutch', 'Romanian', 'German', 'Italian', 'Indonesian', 'Irish']

        for method_name in method_names:
            owwn_obj.map_ow_wn(mapping_obj=mapping_obj, source_dict=source_dict,
                               target_dict=target_dict,
                               source_lan_list=source_lan_list,
                               target_lan_list=target_lan_list,
                               method_name=method_name,
                               bn_obj=bn_obj, df_bn=df_bn,
                               om_obj=om_obj,
                               map_obj=map_obj,
                               wn_obj=wn_obj)


if __name__ == '__main__':

    owwn_obj = MapOWWn()
    eval_obj = Evaluation(file_path='')

    # generate concept pairs first
    owwn_obj.run_map_ow_wn()

    df_sub_gold = pd.read_csv(owwn_obj.input_path + owwn_obj.gold_sub_file)

    method_names = [owwn_obj.ovalsim, owwn_obj.mvotesim]

    for method_name in method_names:
        print('method_name:', method_name)
        owwn_obj.read_allpairs_dict_pickle_csv(method_name=method_name)
        for sort_method in [owwn_obj.tb_namesim_synsize]:
            owwn_obj.map_wn_ow_full(method_name=method_name, sort_method=sort_method)
