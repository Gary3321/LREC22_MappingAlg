from MappingMethods import Mapping
from babelnet import Babelnet
from wordnet import wordnet
from clics import Clics
from evaluation import Evaluation
from utility import get_mapped_cpt, get_wn_candidates_by_clics
import pandas as pd
import pickle
import sys
from tqdm import tqdm
import random
import nltk


class MapClicsWn:
    def __init__(self):
        self.input_path = '../Data/'
        self.output_path = '../Data/'
        self.wn_dict_file = 'wn_entire_syn_lemmas_lan_dict'
        self.clics_dict_file = 'clics_lemmas_lan_dict_en'
        self.ovalsim = 'WordVote'
        self.dev_flag = 'devset'
        self.test_flag = 'testset'
        self.full_flag = 'fullset'
        self.mvotesim = 'LangVote'
        self.gold_data = 'concepticon-wordnet-mapping.csv'
        self.pickle_file = '_clics_wn_allpairs_dict.pickle'
        self.dev_pickle_file = '_clics_wn_dev_dict.pickle'
        self.tb_random = 'random'
        self.tb_synset_size = 'synsetSize'
        self.tb_namesim_synsize = 'namesim_synsize'
        self.clics_file_path = "../Data/"
        self.clics_file = "clics_data.csv"
        self.clics_obj = Clics(file_path=self.clics_file_path, file_name=self.clics_file)

        self.df_concept = self.clics_obj.read_whole_concepts()
        self.wn_obj = wordnet()
        self.bn_language = ['EN', 'ID', 'NL', 'DE', 'RO', 'IT', 'GA']
        self.clics_lan = ['English', 'Indonesian', 'Dutch', 'German', 'Romanian', 'Italian', 'Irish']
        self.mapping_obj = Mapping()
        self.mapping_dev_file_name = '_mapping_results.csv'

    def map_clics_wn_dev(self, mapping_obj, source_dict, target_dict, source_lan_list, target_lan_list,
                         method_name, clics_obj, bn_obj, df_bn, df_clics, lower_flag=False):
        '''
        map from Clics to WN
        :param mapping_obj:
        :param target_dict: wn_dict
        :param source_dict: clics_dict
        :return:
        '''

        # store clics concepts into a list
        source_cpt_ids = list(source_dict.keys())

        # read BabelNet data
        print('reading BabelNet ...')
        df_bn_lans = bn_obj.read_bn_by_lanlst_quick(lan_lst=target_lan_list, df_bn=df_bn)
        # read CLCIS data
        print('reading CLICS ...')
        df_clics_lans = clics_obj.read_clics_by_lans_quick(language_lst=source_lan_list, df_clics=df_clics)
        df_concept = clics_obj.read_whole_concepts()

        cnt = 0
        pairs_dict = {}

        for source_id in tqdm(source_cpt_ids):
            cnt = cnt + 1

            target_candidates = []
            for l in range(len(source_lan_list)):
                source_lemmas = source_dict[source_id].get(source_lan_list[l], [])
                if source_lan_list[l] == 'English':
                    source_lemmas = [l.replace('_', ' ').strip(' ') for l in source_lemmas]

                df_bn_trans = bn_obj.read_bn_by_single_lan(language=target_lan_list[l], df_bn=df_bn_lans)
                df_clics_trans = clics_obj.read_clics_by_lans_quick(language_lst=[source_lan_list[l]],
                                                                    df_clics=df_clics_lans)
                target_candidates_tmp = get_wn_candidates_by_clics(clics_obj=clics_obj, bn_obj=bn_obj,
                                                                   concept_id=source_id,
                                                                   clics_lan_lst=source_lan_list,
                                                                   df_concept=df_concept, df_bn_trans=df_bn_trans,
                                                                   df_clics_trans=df_clics_trans,
                                                                   concept_lemmas=source_lemmas)
                target_candidates = target_candidates + target_candidates_tmp

            target_candidates = list(set(target_candidates))

            for target_id in target_candidates:
                pair_key = str(source_id) + '-' + target_id
                if method_name == self.ovalsim:
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

        pair_file_name = method_name + str(len(target_lan_list)) + self.dev_pickle_file
        print('pair_file_name:', pair_file_name)

        # store sorted_pair_dict to pickle file
        with open(self.output_path + pair_file_name, 'wb') as pickled_file:
            pickle.dump(pairs_dict, pickled_file)

    def load_embd_file(self, dict_file):
        '''
        load the dict files
        :param dict_file:
        :return:
        '''
        with open(dict_file, 'rb') as pickled_file:
            dict_obj = pickle.load(pickled_file)

        dict_obj[330] = {'English': ['headband'], 'Languages': ['English']}
        dict_obj[283] = {'English': ['paddy'], 'Languages': ['English']}
        dict_obj[358] = {'English': ['tinplate'], 'Languages': ['English']}
        dict_obj[2138] = {'English': ['legendary creature'], 'Languages': ['English']}

        dict_obj[238]['English'] = dict_obj[238].get('English', []) + ['prawn']
        dict_obj[1758]['English'] = dict_obj[1758].get('English', []) + ['big sister']

        return dict_obj

    def load_bn_dict(self, file_path, file_name):
        with open(file_path + file_name, 'rb') as pickled_file:
            wn_dict = pickle.load(pickled_file)
        print('total synsets in pickled file:', len(list(wn_dict.keys())))

        return wn_dict

    def read_gold_mapping(self, subset_flag):
        df_gold = pd.read_csv(self.input_path + self.gold_data)
        df_gold['wn_offset'] = 'wn:' + df_gold['WORDNET_SYNSET'].str[1:] + df_gold['WORDNET_SYNSET'].str[0]

        # adjusted manually in order to make sure one-to-one mapping
        df_gold.loc[df_gold['CONCEPTICON_ID'] == 426, 'WORDNET_SYNSET'] = 'n10295819'
        df_gold.loc[df_gold['CONCEPTICON_ID'] == 426, 'wn_offset'] = 'wn:10295819n'
        df_gold.loc[df_gold['CONCEPTICON_ID'] == 289, 'WORDNET_SYNSET'] = 'v01659248'
        df_gold.loc[df_gold['CONCEPTICON_ID'] == 289, 'wn_offset'] = 'wn:01659248v'

        df_gold['clics_id'] = df_gold['CONCEPTICON_ID']

        id_removed = [424, 889, 1446, 1752]

        df_gold = df_gold[~(df_gold['CONCEPTICON_ID'].isin(id_removed))]

        if subset_flag == self.dev_flag:
            df_dev = pd.read_csv(self.input_path + self.dev_file_old)
            dev_wn_ids = df_dev['wn_offset'].to_list()
            df_gold = df_gold[df_gold['wn_offset'].isin(dev_wn_ids)]
        if subset_flag == self.test_flag:
            df_dev = pd.read_csv(self.input_path + self.dev_file_old)
            dev_wn_ids = df_dev['wn_offset'].to_list()
            df_gold = df_gold[~(df_gold['wn_offset'].isin(dev_wn_ids))]
        df_gold.reset_index(inplace=True)

        return df_gold

    def run_generate_dev_pairs(self):
        clics_file_path = "../Data/"
        clics_file = "clics_data.csv"

        bn_file_path = "../Data/"
        bn_file_name = 'bn_data.csv'

        clics_obj = Clics(file_path=clics_file_path, file_name=clics_file)
        df_clics = clics_obj.read_clics_whole()

        bn_obj = Babelnet(language='', file_path=bn_file_path, file_name=bn_file_name)
        df_bn = bn_obj.read_bn_whole()

        mapping_obj = Mapping()

        source_dict = self.load_embd_file(dict_file=self.input_path + self.clics_dict_file)
        target_dict = self.load_bn_dict(file_path=self.input_path, file_name=self.wn_dict_file)
        method_names = [self.mvotesim, self.ovalsim]

        # languages for OvalSim
        target_lan_list_oval = [['EN', 'NL', 'IT'],
                                ['EN', 'NL', 'IT', 'ES'],
                                ['EN', 'NL', 'IT', 'ES', 'GA'],
                                ['EN', 'NL', 'IT', 'ES', 'GA', 'ID'],
                                ['EN', 'NL', 'IT', 'ES', 'GA', 'ID', 'DE'],
                                ['EN', 'NL', 'IT', 'ES', 'GA', 'ID', 'DE', 'RU'],
                                ['EN', 'NL', 'IT', 'ES', 'GA', 'ID', 'DE', 'RU', 'PT'],
                                ['EN', 'NL', 'IT', 'ES', 'GA', 'ID', 'DE', 'RU', 'PT', 'RO'],
                                ['EN', 'NL', 'IT', 'ES', 'GA', 'ID', 'DE', 'RU', 'PT', 'RO', 'ZH'],
                                ['EN', 'NL', 'IT', 'ES', 'GA', 'ID', 'DE', 'RU', 'PT', 'RO', 'ZH', 'FR']]

        # languages for Mvote
        target_lan_list_mvote = [['EN', 'IT', 'NL'],
                                 ['EN', 'IT', 'NL', 'DE'],
                                 ['EN', 'IT', 'NL', 'DE', 'ID'],
                                 ['EN', 'IT', 'NL', 'DE', 'ID', 'ES'],
                                 ['EN', 'IT', 'NL', 'DE', 'ID', 'ES', 'GA'],
                                 ['EN', 'IT', 'NL', 'DE', 'ID', 'ES', 'GA', 'RU'],
                                 ['EN', 'IT', 'NL', 'DE', 'ID', 'ES', 'GA', 'RU', 'PT'],
                                 ['EN', 'IT', 'NL', 'DE', 'ID', 'ES', 'GA', 'RU', 'PT', 'ZH'],
                                 ['EN', 'IT', 'NL', 'DE', 'ID', 'ES', 'GA', 'RU', 'PT', 'ZH', 'RO'],
                                 ['EN', 'IT', 'NL', 'DE', 'ID', 'ES', 'GA', 'RU', 'PT', 'ZH', 'RO', 'FR']]

        for method_name in method_names:
            print('method_name:', method_name)
            if method_name == self.ovalsim:
                target_lan_list = target_lan_list_oval
            else:
                target_lan_list = target_lan_list_mvote

            for target_lans in target_lan_list:
                source_lan_list = []
                for target_lan in target_lans:
                    source_lan = \
                        mapping_obj.df_language_common[mapping_obj.df_language_common['Lan'] == target_lan][
                            'Clics_Ori_lan'].values[0]
                    source_lan_list.append(source_lan)
                print('source_lan_set', source_lan_list)

                self.map_clics_wn_dev(mapping_obj=mapping_obj, source_dict=source_dict, target_dict=target_dict,
                                      source_lan_list=source_lan_list,
                                      target_lan_list=target_lans,
                                      method_name=method_name,
                                      clics_obj=clics_obj, bn_obj=bn_obj, df_bn=df_bn, df_clics=df_clics)

    def run_generate_test_pairs(self):
        clics_file_path = "../Data/"
        clics_file = "clics_data.csv"

        bn_file_path = "../Data/"
        bn_file_name = 'bn_data.csv'

        clics_obj = Clics(file_path=clics_file_path, file_name=clics_file)
        df_clics = clics_obj.read_clics_whole()

        bn_obj = Babelnet(language='', file_path=bn_file_path, file_name=bn_file_name)
        df_bn = bn_obj.read_bn_whole()

        mapping_obj = Mapping()

        source_dict = self.load_embd_file(dict_file=self.input_path + self.clics_dict_file)
        target_dict = self.load_bn_dict(file_path=self.input_path, file_name=self.wn_dict_file)
        method_names = [self.mvotesim, self.ovalsim]

        target_lan_list = [['EN', 'ID', 'NL', 'DE', 'RO', 'IT', 'GA']]

        for method_name in method_names:

            for target_lans in target_lan_list:
                source_lan_list = []
                for target_lan in target_lans:
                    source_lan = \
                        mapping_obj.df_language_common[mapping_obj.df_language_common['Lan'] == target_lan][
                            'Clics_Ori_lan'].values[0]
                    source_lan_list.append(source_lan)
                print('source_lan_set', source_lan_list)

                self.map_clics_wn_dev(mapping_obj=mapping_obj, source_dict=source_dict, target_dict=target_dict,
                                      source_lan_list=source_lan_list,
                                      target_lan_list=target_lans,
                                      method_name=method_name,
                                      clics_obj=clics_obj, bn_obj=bn_obj, df_bn=df_bn, df_clics=df_clics)

    def read_devpairs_dict_pickle_csv(self, method_name, lan_list, df_concept, wn_obj, mapping_obj, wn_dict):
        '''
        1) read all pairs pickle file to csv, for mapping all CLICS concepts to all WN concepts;
        adding two additional columns to the csv files for breaking/sorting ties:
        column1 -- a number from a shuffled list which its length equals the number of pairs;
        column2 -- editing distance between two concept names
        column3 -- synset size
        2) generate another csv file containing ties for all CLICS concepts
        :param method_name: OvalSimilarity or MvoteSimilarity
        :param file_name: pickle file name
        :return: two csv files
        '''

        pair_file_name = method_name + str(len(lan_list)) + self.dev_pickle_file
        print('pair_file_name:', pair_file_name)

        with open(self.output_path + pair_file_name, 'rb') as pickled_file:
            pairs_dict = pickle.load(pickled_file)

        # sort first by similarity score in descending order, then by concept pair name in ascending
        sorted_pair = sorted(pairs_dict.items(), key=lambda kv: (-kv[1], kv[0]))
        df_mapping_result_all = pd.DataFrame(columns=['clics_id', 'wn_offset', 'similarity', 'random_number',
                                                      'editing_dis', 'synset_size', 'editing_synsize'])
        # get a random shuffled list for breaking ties with random number
        random_lst = [i for i in range(len(sorted_pair))]
        random.Random(2021).shuffle(random_lst)
        i = 0
        for item in tqdm(sorted_pair):
            clics_id = int(item[0].split('-')[0])
            wn_offset = item[0].split('-')[1]
            similary_score = item[1]

            clics_name = df_concept[df_concept['ID'] == clics_id]['GLOSS'].values[0].lower()

            wn_synset = wn_obj.name_from_synset(wn_obj.synset_from_offset(wn_offset.replace('wn:', '')))
            wn_items = wn_synset.split('.')
            wn_name = wn_items[0]
            name_edit = nltk.edit_distance(clics_name, wn_name)
            synset_size = self.get_wn_synset_size(wn_tie=wn_offset, lan_list=lan_list,
                                                  mapping_obj=mapping_obj, wn_dict=wn_dict)
            editing_synsize = str(name_edit).zfill(3) + str(synset_size)
            random_num = random_lst[i]
            i = i + 1

            df_mapping_result_all = df_mapping_result_all.append({'clics_id': clics_id, 'wn_offset': wn_offset,
                                                                  'similarity': similary_score,
                                                                  'random_number': random_num,
                                                                  'editing_dis': name_edit, 'synset_size': synset_size,
                                                                  'editing_synsize': editing_synsize},
                                                                 ignore_index=True)

        df_mapping_result_all.to_csv(self.output_path + method_name + str(len(lan_list)) + self.mapping_dev_file_name,
                                     index=False)

        print('total mapping result:', len(df_mapping_result_all))

    def get_wn_synset_size(self, wn_tie, lan_list, mapping_obj, wn_dict):
        size = 0
        for lan in lan_list:
            size = size + len(mapping_obj.get_lex(cpt_name=wn_tie, cpt_dict=wn_dict,
                                                  language=lan))
        return size

    def map_wn_clics_dev(self, method_name, sort_method, lan_list):

        eval_obj = Evaluation(file_path='')

        df_mapping_result_all = pd.read_csv(
            self.output_path + method_name + str(len(lan_list)) + self.mapping_dev_file_name)

        # sorting
        df_mapping_result_all['editing_synsize1'] = 'a' + df_mapping_result_all['editing_dis'].astype(str).str.zfill(
            3) + df_mapping_result_all['synset_size'].astype(str)

        sort_column = ''
        if sort_method == self.tb_random:
            sort_column = 'random_number'
        elif sort_method == self.tb_namesim_synsize:
            sort_column = 'editing_synsize1'
        elif sort_method == self.tb_synset_size:
            sort_column = 'synset_size'
        print('sort_column:', sort_column)
        df_mapping_result_all = df_mapping_result_all.sort_values(['similarity', 'clics_id', sort_column],
                                                                  ascending=[False, True, True])

        df_mapping_result_all.reset_index(inplace=True)

        # get one-to-one mapping
        df_onetoone_mapping = pd.DataFrame(columns=['clics_id', 'wn_offset_ties', 'wn_offset', 'similarity'])
        used_clics = set()
        used_wn = set()

        for i in tqdm(range(len(df_mapping_result_all))):
            clics_id = df_mapping_result_all['clics_id'][i]
            wn_offset = df_mapping_result_all['wn_offset'][i]
            sim_score = df_mapping_result_all['similarity'][i]
            if clics_id not in used_clics and wn_offset not in used_wn:
                df_onetoone_mapping = df_onetoone_mapping.append({'clics_id': clics_id,
                                                                  'wn_offset_ties': str([wn_offset]),
                                                                  'wn_offset': wn_offset,
                                                                  'similarity': sim_score}, ignore_index=True)
                used_clics.add(clics_id)
                used_wn.add(wn_offset)

        df_result = df_onetoone_mapping[['clics_id', 'wn_offset_ties', 'wn_offset']]
        df_result.columns = ['clics_id', 'wn_offset_ties', 'wn_offset_mapped']

        # evaluate with test data
        df_gold = self.read_gold_mapping(subset_flag=self.test_flag)

        print('gold pairs:', len(df_gold))
        print('gold - onetoone:', set(df_gold['clics_id'].to_list()) - set(df_onetoone_mapping['clics_id'].to_list()))

        df_final_result = df_result.merge(df_gold, how='inner', left_on='clics_id', right_on='clics_id')

        print('mapping pairs number after combining with gold:', len(df_final_result))

        df_analysis, acc = eval_obj.analysis_v3(mapping_direction='clics-wn', method=method_name,
                                                tie_breaking=sort_method,
                                                df_result=df_final_result, gold_col_name='wn_offset',
                                                mapped_col_name='wn_offset_ties',
                                                mapped_unique_col='wn_offset_mapped', col_name_bk_ties=[])
        return acc

    def run_test_mrr(self):
        clics_file_path = "../Data/"
        clics_file = "clics_data.csv"

        clics_obj = Clics(file_path=clics_file_path, file_name=clics_file)
        df_concept = clics_obj.read_whole_concepts()

        wn_obj = wordnet()

        mapping_obj = Mapping()

        wn_dict = self.load_bn_dict(file_path=self.input_path, file_name=self.wn_dict_file)

        target_lan_list = [['EN', 'ID', 'NL', 'DE', 'RO', 'IT', 'GA']]

        method_names = [self.ovalsim, self.mvotesim]
        for method in method_names:
            print('method:', method)
            for tie_break in [self.tb_namesim_synsize]:
                for lan in target_lan_list:
                    self.read_devpairs_dict_pickle_csv(method_name=method, lan_list=lan, df_concept=df_concept,
                                                       wn_obj=wn_obj, mapping_obj=mapping_obj, wn_dict=wn_dict)
                    acc = self.map_wn_clics_dev(method_name=method, sort_method=tie_break, lan_list=lan)
                    print('acc....', acc)

    def cal_mrr_test_parepare(self, method_name, sort_method, lan_list):

        eval_obj = Evaluation(file_path='')

        df_mapping_result_all = pd.read_csv(
            self.output_path + method_name + str(len(lan_list)) + self.mapping_dev_file_name)

        print('all mapping result:', len(df_mapping_result_all))
        # sorting
        df_mapping_result_all['editing_synsize1'] = 'a' + df_mapping_result_all['editing_dis'].astype(
            str).str.zfill(3) + df_mapping_result_all['synset_size'].astype(str)

        sort_column = ''
        if sort_method == self.tb_random:
            sort_column = 'random_number'
        elif sort_method == self.tb_namesim_synsize:
            sort_column = 'editing_synsize1'
        elif sort_method == self.tb_synset_size:
            sort_column = 'synset_size'
        print('sort_column:', sort_column)
        df_mapping_result_all = df_mapping_result_all.sort_values(['similarity', 'clics_id', sort_column],
                                                                  ascending=[False, True, True])

        df_mapping_result_all.reset_index(inplace=True)

        # evaluate with test data
        df_gold = self.read_gold_mapping(subset_flag=self.test_flag)

        rank_num = set()
        df_mrr_rank = pd.DataFrame(columns=['clics_id', 'wn_offset_lst', 'wn_offset_gold'])
        for i in tqdm(range(len(df_gold))):
            gold_clics_id = df_gold['clics_id'][i]
            gold_wn_offset = df_gold['wn_offset'][i]
            rank_lst = []
            for j in range(len(df_mapping_result_all)):
                clics_id = df_mapping_result_all['clics_id'][j]
                if clics_id == gold_clics_id:
                    wn_offset = df_mapping_result_all['wn_offset'][j]
                    rank_lst.append(wn_offset)

            rank_num.add(len(rank_lst))
            df_mrr_rank = df_mrr_rank.append({'clics_id': gold_clics_id,
                                              'wn_offset_lst': str(rank_lst),
                                              'wn_offset_gold': gold_wn_offset}, ignore_index=True)

        mrr = eval_obj.cal_mrr(df_result=df_mrr_rank, gold_col_name='wn_offset_gold',
                               mapped_col_name='wn_offset_lst')

        # print('rank number:', rank_num)
        print('langauges:', lan_list, 'mrr:', mrr)
        return mrr

    def run_cal_test_mrr(self):

        target_lan_list = [['EN', 'ID', 'NL', 'DE', 'RO', 'IT', 'GA']]

        method_names = [self.ovalsim, self.mvotesim]
        for method in method_names:

            for tie_break in [self.tb_namesim_synsize]:
                for lan in target_lan_list:
                    mrr = self.cal_mrr_test_parepare(method_name=method, sort_method=tie_break, lan_list=lan)
                    print('method:', method, 'mrr:', mrr)


if __name__ == '__main__':
    map_obj = MapClicsWn()

    map_obj.run_test_mrr()
    map_obj.run_cal_test_mrr()
