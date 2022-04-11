import pandas as pd
import re
import pickle
from tqdm import tqdm


class Clics:
    def __init__(self, file_path, file_name):
        self.file_path = file_path
        self.file_name = file_name
        self.out_file_path = '../Data/'
        self.lemma_lan_file = 'clics_lemmas_lan_dict'
        self.lemma_lan_file_en = 'clics_lemmas_lan_dict_en'
        self.language_file = 'Clics_BN_lemmas_overlap_oriLan.csv'
        self.concept_file = 'concepticon.csv'
        self.clics_pos = {'Action/Process': ['v'], 'Person/Thing': ['n'], 'Property': ['a', 'r']}

    def get_languages(self, df_clics):
        '''
        :return:a list of CLICS language names
        '''
        clics_lans_set = set(df_clics['language'].to_list())
        clics_lans_lst = [lan.lower().strip() for lan in clics_lans_set]
        return clics_lans_lst

    def read_clics_whole(self):
        df_clics = pd.read_csv(self.file_path + self.file_name)
        df_clics = df_clics[df_clics['Value'].notna()]
        df_clics = df_clics[df_clics['language'].notna()]
        return df_clics

    def read_lemmas_pickle_all(self):
        '''
        store all lemmas to a big dictionary, like {wn_offset:{lan1:[lemma1, lemma2, ...], lan2:[lemmas]..}
        the last item is a language list contained by the concept
        :return:
        '''

        df_clics = self.read_clics_whole()

        # only consider the languages shared with BN

        clics_dict = {}
        clics_ids = list(
            set(df_clics['ConceptionID'].to_list() + [330, 283, 2138, 358]))  # clics does not contain those 4 concepts

        for clics_id in tqdm(clics_ids):

            df_clics_sub = df_clics[df_clics['ConceptionID'] == clics_id]
            clics_id_lans = set(df_clics_sub['language'].to_list())
            clics_id_lans.add('English')
            lan_dict = {}
            for lan in clics_id_lans:
                df_clics_lan = df_clics_sub[df_clics_sub['language'] == lan]
                lemmas = self.get_lemmas_by_id(df_lans=df_clics_lan, language_lst=[lan], concept_id=clics_id)
                lan_dict[lan] = lemmas
            lan_dict['Languages'] = clics_id_lans

            clics_dict[clics_id] = lan_dict

        # store the dictionary to a pickle file
        with open(self.out_file_path + self.lemma_lan_file_en, 'wb') as pickled_file:
            pickle.dump(clics_dict, pickled_file)

    def read_lemmas_pickle(self):
        '''
        store all lemmas to a big dictionary, like {wn_offset:{lan1:[lemma1, lemma2, ...], lan2:[lemmas]..}
        the last item is a language list contained by the concept
        :return:
        '''
        df_lan_overlap = pd.read_csv(self.out_file_path + self.language_file)
        clics_lan = df_lan_overlap['Clics_Ori_lan'].to_list()
        df_clics = self.read_clics_by_lans(clics_lan)
        df_clics = df_clics[df_clics['Value'].notna()]
        df_clics = df_clics[df_clics['language'].notna()]
        # only consider the languages shared with BN

        clics_dict = {}
        clics_ids = list(set(df_clics['ConceptionID'].to_list()))

        for clics_id in tqdm(clics_ids):

            df_clics_sub = df_clics[df_clics['ConceptionID'] == clics_id]
            clics_id_lans = df_clics_sub['language'].to_list()

            lan_dict = {}
            for lan in clics_id_lans:
                df_clics_lan = df_clics_sub[df_clics_sub['language'] == lan]
                lemmas = self.get_lemmas_by_id(df_lans=df_clics_lan, language_lst=[lan], concept_id=clics_id)
                lan_dict[lan] = lemmas
            lan_dict['Languages'] = clics_id_lans

            clics_dict[clics_id] = lan_dict

        # store the dictionary to a pickle file
        with open(self.out_file_path + self.lemma_lan_file, 'wb') as pickled_file:
            pickle.dump(clics_dict, pickled_file)

    def read_clics_lemma_dict(self):
        with open(self.out_file_path + self.lemma_lan_file_en, 'rb') as pickled_file:
            clics_dict = pickle.load(pickled_file)

        print(len(clics_dict.keys()))

        print(clics_dict[417])

    def read_whole_concepts(self):
        df_concept = pd.read_csv(self.file_path + self.concept_file)
        return df_concept

    def read_clics_by_lans(self, language_lst):
        '''
        :param language_lst: like ['Mandarin Chinese', 'Italian', 'English', 'Russian']
        :return:
        '''
        df_clics = pd.read_csv(self.file_path + self.file_name)
        df_clics_lan = df_clics[df_clics['language'].isin(language_lst)]
        return df_clics_lan

    def read_clics_by_lans_quick(self, language_lst, df_clics):
        '''
        :param language_lst: like ['Mandarin Chinese', 'Italian', 'English', 'Russian']
        :return:
        '''
        df_clics_lan = df_clics[df_clics['language'].isin(language_lst)]
        return df_clics_lan

    def preprocess_concept_names(self):
        '''
        remove brackets and OR from concept names
        :return:
        '''
        df_concept = pd.read_csv(self.file_path + "concepticon.csv")
        p1 = re.compile(r"[(](.*?)[)]", re.S)  # 最小匹配
        for i in range(len(df_concept)):
            gloss = df_concept['GLOSS'][i]
            concept_name = gloss.replace(' OR ', ';')
            cpt_gloss = re.findall(p1, concept_name)
            cpt_name = re.sub(r"[(](.*?)[)]", "", concept_name)

            df_concept.loc[i, 'ConceptName'] = cpt_name
            df_concept.loc[i, 'ConceptGloss'] = str(cpt_gloss)

        df_concept.to_csv(self.file_path + 'concepticon_processed.csv', index=False)

    def get_lemmas_by_id(self, df_lans, language_lst, concept_id):
        '''
        extract CLICS lemmas, for English, I add CLICS concept name to the lemma list
        :param df_lans: CLICS language data
        :param language_lst: like ['Mandarin Chinese', 'Italian', 'English', 'Russian']
        :param concept_id:
        :return: a list of clics lemmas
        '''
        # df_lans = self.read_clics_by_lans(language_lst)
        clics_lemma = df_lans[df_lans['ConceptionID'] == concept_id]['Value'].to_list()
        clics_lemma = list(set(clics_lemma))
        clics_lemmas = []
        for lemma in clics_lemma:
            if '/' in lemma:
                clics_lemmas = clics_lemmas + lemma.split('/')
            else:
                clics_lemmas = clics_lemmas + lemma.split(',')
                clics_lemmas = [lemma.strip() for lemma in clics_lemmas]
        clics_lemmas = list(set([lemma.strip() for lemma in clics_lemmas]))

        # add concept name to the lemma list
        cpt_names = []
        if "English" in language_lst:
            df_cpts = pd.read_csv(self.file_path + 'concepticon_processed.csv')
            cpt_name = df_cpts[df_cpts['ID'] == concept_id]['ConceptName'].values[0]
            cpt_names = cpt_name.lower().replace('-', ' ').split(';')
        return list(set(clics_lemmas + cpt_names))

    def get_lans_by_id(self, df_clics, cpt_id):
        '''
        return all languages that a concept contains
        :param df_clics:
        :param cpt_id:
        :return:
        '''
        lans = df_clics[df_clics['ConceptionID'] == cpt_id]['language'].to_list()
        return list(set(lans))

    def get_lemmas_by_lan(self, df_clics, language):
        '''
        extract CLICS lemmas, for English, I add CLICS concept name to the lemma list
        :param df_clics: the whole CLICS data
        :param language: like 'Mandarin Chinese', 'Italian', 'English', 'Russian'
        :return: a list of clics lemmas
        '''
        clics_lemma = df_clics[df_clics['language'].str.lower() == language.lower()]['Value'].to_list()
        clics_lemma = list(set(clics_lemma))
        clics_lemmas = []
        for lemma in clics_lemma:
            if '/' in lemma:
                clics_lemmas = clics_lemmas + lemma.split('/')

            else:
                clics_lemmas = clics_lemmas + lemma.split(',')
        clics_lemmas = set([lemma.strip() for lemma in clics_lemmas])

        return clics_lemmas

    def get_ids_by_targetlist(self, target_list, df_clics_lans):
        '''
        giving a target list and pos, extract clics ids whose lemmas  have overlap with the list
        :param target_list:  WN lemmas
        :param df_clics_lans: clics data in given languages
        :return: a dataframe, a list of concept ids
        '''
        df_clics_lemmas = df_clics_lans[df_clics_lans['Value'].isin(target_list)]
        df_clics_sp = df_clics_lans[(df_clics_lans['Value'].str.contains(',')) |
                                    (df_clics_lans['Value'].str.contains('/'))]
        df_cpt_ids = pd.concat([df_clics_lemmas, df_clics_sp])

        cpt_list = list(set(df_cpt_ids['ConceptionID'].to_list()))  # get candidate list
        return df_cpt_ids, cpt_list

    def get_clics_pos_by_id(self, concept_id, df_concept):
        '''
        return a list of POS by given a concept id and concept dataframe
        '''
        cpt_cat = df_concept[df_concept['ID'] == concept_id]['ONTOLOGICAL_CATEGORY'].values[0]
        return self.clics_pos.get(cpt_cat, [])

    def get_lemmas_by_pos(self, df_subset_clics, cpt_id, target_pos, df_concept, lan_list):
        '''

        :param df_subset_clics: returned from "get_ids_by_targetlist"
        :param cpt_id: concept id
        :param target_pos: WN pos
        :param df_concept: concept data; returned from "read_whole_concepts"
        :param lan_list: CLICS lemmas
        :return: a list of CLCIS lemmas
        '''

        cpt_pos_lst = self.get_clics_pos_by_id(concept_id=cpt_id, df_concept=df_concept)
        if len(cpt_pos_lst) > 0 and target_pos in cpt_pos_lst:
            cpt_lemmas = self.get_lemmas_by_id(df_lans=df_subset_clics, language_lst=lan_list,
                                               concept_id=cpt_id)
        elif len(cpt_pos_lst) == 0:
            cpt_lemmas = self.get_lemmas_by_id(df_lans=df_subset_clics, language_lst=lan_list,
                                               concept_id=cpt_id)
        else:
            cpt_lemmas = []
        return cpt_lemmas

    def get_concept_name(self, df_concept, concept_id):
        concept_name = df_concept[df_concept['ID'] == concept_id]['GLOSS'].values[0]
        return concept_name.lower()

    def get_gloss(self, df_concept, concept_id):
        concept_gloss = df_concept[df_concept['ID'] == concept_id]['DEFINITION'].values[0]
        return concept_gloss.lower()

    def get_semantic_field(self, df_concept, concept_id):
        df_concept = df_concept[df_concept['SEMANTICFIELD'].notna()]
        concept_semantic = df_concept[df_concept['ID'] == concept_id]['SEMANTICFIELD'].values[0]
        return concept_semantic.lower()

    def get_concepts_def(self, output_path):
        '''
        get all clics concepts and their definition, definition is comprised of concept name, semanticfield,
        and gloss
        :return:
        '''
        df_concept = self.read_whole_concepts()
        df_concept['SEMANTICFIELD'] = df_concept['SEMANTICFIELD'].fillna('')
        df_concept['Gloss-Def'] = df_concept['GLOSS'] + ';' + df_concept['SEMANTICFIELD'] + ';' + df_concept[
            'DEFINITION']
        df_clics_def = df_concept[['ID', 'GLOSS', 'Gloss-Def']]
        df_clics_def.columns = ['ID', 'Concept', 'Def']
        df_clics_def.to_csv(output_path + 'clics_all_concepts_defs.csv', index=False)

        print(len(df_clics_def))
        print(df_clics_def[:3])

    def get_concept_size_per_language(self, language_lst):
        '''
        get number of lemmas in each concept per language
        :param language_lst:
        :return:
        '''
        df_clics = self.read_clics_whole()
        for language in language_lst:
            df_lan = df_clics[df_clics['language'].str.lower() == language.lower()]
            total_cpt = len(set(df_lan['ConceptionID'].to_list()))
            totla_lemmas = len(self.get_lemmas_by_lan(df_clics=df_clics, language=language))
            average = round(totla_lemmas / total_cpt, 2)
            print(language, average)


if __name__ == '__main__':
    None
