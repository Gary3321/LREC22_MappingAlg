import pandas as pd
import ast


class Mapping:
    def __init__(self):
        self.out_file_path = '../Data/'
        self.language_file = 'Clics_BN_lemmas_overlap_oriLan.csv'
        self.df_language_common = pd.read_csv(self.out_file_path + self.language_file)
        self.OW_WN_lan_dict = {'English': 'EN', 'Dutch': 'NL', 'Italian': 'IT', 'Romanian': 'RO', 'Spanish': 'ES',
                               'Irish': 'GA', 'German': 'DE',
                               'Russian': 'RU', 'Portuguese': 'PT', 'Romanian': 'RO'}

        self.WN_OW_lan_dict = {'EN': 'English', 'NL': 'Dutch', 'IT': 'Italian', 'RO': 'Romanian', 'ES': 'Spanish',
                               'GA': 'Irish', 'DE': 'German',
                               'RU': 'Russian', 'PT': 'Portuguese', 'RO': 'Romanian'}

    def get_lex(self, cpt_name, cpt_dict, language):
        '''
        return the set of lexicalizations of a given concept in a given
        language
        :param cpt_name: wn offset or CLICS id
        :param cpt_dict: a dictionary contains concepts name, language,
                and corresponding lemmas
        :param language: a given language
        :return:
        '''
        lex = cpt_dict[cpt_name].get(language, [])
        if language in ['EN', 'English']:
            lex = [l.replace('_', ' ').strip(' ') for l in lex]
        return set(lex)

    def get_lan(self, cpt_name, cpt_dict, lan_set):
        '''
        return a subset of languages in a given language set that lexicalize
        the given concept
        :param cpt_name: wn offset or CLCIS ids
        :param cpt_dict: a dictionary contains concepts name, language,
                and corresponding lemmas
        :param lan_set: a given language set
        :return: a set of BN languages format
        '''
        lan_lst = list(set(cpt_dict[cpt_name]['Languages']) & lan_set)
        # convert CLICS language to BN language format
        lan_lst_convert = self.df_language_common[self.df_language_common['Clics_Ori_lan'].isin(lan_lst)][
            'Lan'].to_list()
        if len(lan_lst_convert) > 0:
            return set(lan_lst_convert)
        else:
            return set(lan_lst)

    def get_lan_ow(self, cpt_name, cpt_dict, lan_set):
        '''
        return a subset of languages in a given language set that lexicalize
        the given concept
        :param cpt_name: wn offset or CLCIS ids
        :param cpt_dict: a dictionary contains concepts name, language,
                and corresponding lemmas
        :param lan_set: a given language set
        :return: a set of BN languages format
        '''
        lan_lst = list(set(cpt_dict[cpt_name]['Languages']) & lan_set)
        # convert OW language to BN language format
        lan_lst_convert = []
        if len(set(list(self.OW_WN_lan_dict.keys())) & lan_set) > 0:
            for l in lan_lst:
                lan_lst_convert.append(self.OW_WN_lan_dict[l])
            return set(lan_lst_convert)

        else:
            return set(lan_lst)

    def OvalSimilarity(self, cpt_source, cpt_dict_source, cpt_target, cpt_dict_target,
                       lan_list_source, lan_list_target, lower_flag=False):
        '''
        return word overlap between two concepts in a given language set
        :param cpt_source: concept  from the source resource
        :param cpt_dict_source: concept dictionary from the source resource
        :param cpt_target:
        :param cpt_dict_target:
        :param lan_list_source: a given language list from source resource
        :param lan_list_target:
        :return: overlap size
        '''
        overlap_lst = []
        overlap_lan_dict = {}  # for log, store language and overlap
        for i in range(len(lan_list_source)):
            source_lex = self.get_lex(cpt_name=cpt_source, cpt_dict=cpt_dict_source, language=lan_list_source[i])
            target_lex = self.get_lex(cpt_name=cpt_target, cpt_dict=cpt_dict_target, language=lan_list_target[i])
            if lower_flag:
                source_lex = set([i.lower() for i in source_lex])
                target_lex = set([i.lower() for i in target_lex])

            overlap_set = source_lex & target_lex
            if len(overlap_set) > 0:
                overlap_lst = overlap_lst + list(overlap_set)
                overlap_lan_dict[lan_list_target[i]] = overlap_set

        return len(overlap_lst), overlap_lan_dict

    def MvoteSimilarity(self, cpt_source, cpt_dict_source, cpt_target, cpt_dict_target,
                        lan_list_source, lan_list_target, lower_flag=False):
        '''
        return language overlap between two concepts in a given language set
        :param cpt_source: concept  from the source resource
        :param cpt_dict_source: concept dictionary from the source resource
        :param cpt_target:
        :param cpt_dict_target:
        :param lan_set_source: a given language list from source resource
        :return: langauge overlap size
        '''

        overlap_lans = []
        source_lan = self.get_lan(cpt_name=cpt_source, cpt_dict=cpt_dict_source, lan_set=set(lan_list_source))
        target_lan = self.get_lan(cpt_name=cpt_target, cpt_dict=cpt_dict_target, lan_set=set(lan_list_target))
        # get languages shared by the two concepts
        overlap_set = source_lan & target_lan

        target_overlap_lst = list(overlap_set)  # BN language
        source_overlap_lst = []
        # get clics languages
        for tl in target_overlap_lst:
            source_overlap_lst.append(
                self.df_language_common[self.df_language_common['Lan'] == tl]['Clics_Ori_lan'].values[0])

        # filter out the language that two concepts have no common words
        for t in range(len(target_overlap_lst)):
            if lower_flag:
                tl = set([i.lower() for i in cpt_dict_target[cpt_target][target_overlap_lst[t]]])
                sl = set([i.lower() for i in cpt_dict_source[cpt_source][source_overlap_lst[t]]])
                if len(tl & sl) > 0:
                    overlap_lans.append(target_overlap_lst[t])
            else:
                source_lex = self.get_lex(cpt_name=cpt_source, cpt_dict=cpt_dict_source, language=source_overlap_lst[t])
                target_lex = self.get_lex(cpt_name=cpt_target, cpt_dict=cpt_dict_target, language=target_overlap_lst[t])

                if len(source_lex & target_lex) > 0:
                    overlap_lans.append(target_overlap_lst[t])

        # return overlap size, shared languages
        return len(overlap_lans), overlap_lans

    def MvoteSimilarity_ow(self, cpt_source, cpt_dict_source, cpt_target, cpt_dict_target,
                           lan_list_source, lan_list_target, lower_flag=False):
        '''
        return language overlap between two concepts in a given language set
        :param cpt_source: concept  from the source resource
        :param cpt_dict_source: concept dictionary from the source resource
        :param cpt_target:
        :param cpt_dict_target:
        :param lan_set_source: a given language list from source resource
        :return: langauge overlap size
        '''

        overlap_lans = []
        source_lan = self.get_lan_ow(cpt_name=cpt_source, cpt_dict=cpt_dict_source, lan_set=set(lan_list_source))
        target_lan = self.get_lan_ow(cpt_name=cpt_target, cpt_dict=cpt_dict_target, lan_set=set(lan_list_target))
        # get languages shared by the two concepts
        overlap_set = source_lan & target_lan

        source_overlap_lst = list(overlap_set)  # BN language
        target_overlap_lst = []
        # get ow languages
        for sl in source_overlap_lst:
            target_overlap_lst.append(self.WN_OW_lan_dict[sl])

        # filter out the language that two concepts have no common words
        for t in range(len(target_overlap_lst)):
            source_lex = self.get_lex(cpt_name=cpt_source, cpt_dict=cpt_dict_source, language=source_overlap_lst[t])
            target_lex = self.get_lex(cpt_name=cpt_target, cpt_dict=cpt_dict_target, language=target_overlap_lst[t])

            if len(source_lex & target_lex) > 0:
                overlap_lans.append(target_overlap_lst[t])

        # return overlap size, shared languages
        return len(overlap_lans), overlap_lans


if __name__ == '__main__':
    None
