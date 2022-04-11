import pandas as pd
import ast


class Evaluation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df_analysis_ow = pd.DataFrame(columns=['Threshold', 'Language', 'MRR', 'Ties', 'GoldSynsets', 'NONEs',
                                                    'Accuracy-random', 'Accuracy-similarity', 'P', 'R', 'A', 'F1',
                                                    'Dict'])

        self.df_analysis_threshold = pd.DataFrame(
            columns=['Threshold', 'Language', 'MRR', 'Ties', 'GoldSynsets', 'NONEs',
                     'Accuracy-random', 'Accuracy-similarity', 'Accuracy-norm', 'ties_norm'])

        self.df_analysis = pd.DataFrame(columns=['Language', 'MRR', 'Ties', 'GoldSynsets', 'NONEs',
                                                 'Accuracy-random', 'Accuracy-similarity'])

        self.df_analysis_v3 = pd.DataFrame(columns=['mapping_direction', 'Method', 'tie_breaking', 'MRR', 'Ties',
                                                    'Accuracy', 'ties_after_breaking'])

    def cal_mrr(self, df_result, gold_col_name, mapped_col_name):
        '''
        calculate Mean reciprocal rank (MRR) based on
        https://en.wikipedia.org/wiki/Mean_reciprocal_rank
        :param df_result: final mapping result file
        :return:
        '''
        gold_col = gold_col_name
        mapped_col = mapped_col_name

        df_data = df_result.copy()
        mrr = 0
        for i in range(len(df_data)):
            gold = df_data[gold_col][i]
            pred = df_data[mapped_col][i]
            pred = ast.literal_eval(pred)
            if gold in pred:
                mrr_each = 1 / (pred.index(gold) + 1)
                # print(f'mrr_each: {1}/{pred_senses.index(gold_sense) + 1}')
                mrr += mrr_each
        mrr_avg = mrr / len(df_data)
        return round(mrr_avg, 3)

    def cal_acc(self, df_result, gold_col_name, mapped_col_name):
        gold_col = gold_col_name
        mapped_col = mapped_col_name
        ok = len(df_result[df_result[gold_col_name] == df_result[mapped_col_name]])
        total = len(df_result)
        acc = round(ok / total, 3)
        return acc

    def get_ties_num(self, df_result, mapped_col_name):
        '''
        calculate the # instances with ties
        :param df_mapping_result:
        :return: a number
        '''
        mapped_col = mapped_col_name
        df_data = df_result.copy()
        ties_num = len(df_data[df_data[mapped_col].str.contains(',')])
        return ties_num

    def get_none_mapping(self, df_result, mapped_col_name):
        '''
        calculate the # instances that are mapped to nothing (empty synsets)
        :param df_result:
        :param target_lan: target language
        :return:
        '''
        mapped_col = mapped_col_name
        df_data = df_result.copy()
        inst_num = len(df_data[df_data[mapped_col].str.len() < 3])
        return inst_num

    def get_gold_instance(self, df_result, gold_col_name, mapped_col_name):
        '''
        calculate the # instances containing the gold synset
        :param df_result:
        :param target_lan:
        :return: a num
        '''
        gold_col = gold_col_name
        mapped_col = mapped_col_name
        df_data = df_result.copy()
        cnt = 0
        for i in range(len(df_data)):
            gold = df_data[gold_col][i]
            mapped = df_data[mapped_col][i]
            mapped = ast.literal_eval(mapped)
            if gold in mapped:
                cnt += 1
        return cnt

    def analysis(self, df_result, target_lan_lst, gold_col_name, mapped_col_name,
                 mapped_unique_col_rand, mapped_unique_col_sim):
        '''

        :param df_result:
        :param target_lan_lst:
        :return:
        '''

        df_data = df_result.copy()
        acc_rand = self.cal_acc(df_data, gold_col_name, mapped_unique_col_rand)
        acc_sim = self.cal_acc(df_data, gold_col_name, mapped_unique_col_sim)
        mrr_t = self.cal_mrr(df_data, gold_col_name, mapped_col_name)
        ties_t = self.get_ties_num(df_data, mapped_col_name)
        gold_t = self.get_gold_instance(df_data, gold_col_name, mapped_col_name)
        none_t = self.get_none_mapping(df_data, mapped_col_name)

        self.df_analysis = self.df_analysis.append({'Language': '-'.join(target_lan_lst), 'MRR': mrr_t, 'Ties': ties_t,
                                                    'GoldSynsets': gold_t, 'NONEs': none_t, 'Accuracy-random': acc_rand,
                                                    'Accuracy-similarity': acc_sim}, ignore_index=True)

        print('\n MRR:', mrr_t, 'ties number:', ties_t, 'random:', acc_rand, 'sim:', acc_sim)
        return self.df_analysis

    def analysis_ow(self, threshold, df_result, target_lan_lst, gold_col_name, mapped_col_name,
                    mapped_unique_col_rand, mapped_unique_col_sim, df_gold_data):
        '''

        :param df_result:
        :param target_lan_lst:
        :return:
        '''

        df_data = df_result.copy()

        df_gold_compare, p, r, a, f1, dict = self.cal_p_r_f_a(df_mapping_result=df_result, df_gold_data=df_gold_data,
                                                              mapped_col_name=mapped_unique_col_sim)
        self.df_analysis_ow = self.df_analysis_ow.append(
            {'Threshold': str(threshold), 'Language': '-'.join(target_lan_lst),
             'MRR': '', 'Ties': '',
             'GoldSynsets': '', 'NONEs': '', 'Accuracy-random': '',
             'Accuracy-similarity': '', 'P': p,
             'R': r, 'A': a, 'F1': f1, 'Dict': dict}, ignore_index=True)

        print('\n', 'p:', p, 'r:', r, 'A:', a, 'f1:', f1)
        return self.df_analysis_ow, df_gold_compare

    def analysis_ow_wn(self, threshold, df_result, target_lan_lst, mapped_col_name,
                       mapped_unique_col_sim, df_gold_data):
        '''

        :param df_result:
        :param target_lan_lst:
        :return:
        '''

        df_data = df_result.copy()

        ties_t = self.get_ties_num(df_data, mapped_col_name)
        none_t = self.get_none_mapping(df_data, mapped_col_name)

        df_gold_compare, p, r, a, f1, dict = self.cal_p_r_f_a_ow_wn(df_mapping_result=df_result,
                                                                    df_gold_data=df_gold_data,
                                                                    mapped_col_name=mapped_unique_col_sim)
        self.df_analysis_ow = self.df_analysis_ow.append(
            {'Threshold': str(threshold), 'Language': '-'.join(target_lan_lst),
             'MRR': '', 'Ties': '',
             'GoldSynsets': '', 'NONEs': none_t, 'Accuracy-random': '',
             'Accuracy-similarity': '', 'P': p,
             'R': r, 'A': a, 'F1': f1, 'Dict': dict}, ignore_index=True)

        print('\n ties number:', ties_t, 'p:', p, 'r:', r, 'A:', a, 'f1:', f1)
        return self.df_analysis_ow, df_gold_compare

    def cal_p_r_f_a(self, df_mapping_result, df_gold_data, mapped_col_name):
        '''
        cal precision, recall, f1, and accuracy
        :param df_mapping_result:
        :param df_gold_data:
        :return:
        '''
        df_data = df_gold_data.copy()
        df_data['WN_OFFSET'] = 'wn:' + df_data['WN_OFFSET'].str.replace('-', '')

        mapping_ids = df_mapping_result['wn_offset'].to_list()
        df_data = df_data[df_data['WN_OFFSET'].isin(mapping_ids)]
        df_data.reset_index(inplace=True)
        for i in range(len(df_data)):
            wn_id = df_data['WN_OFFSET'][i]
            ow_id = df_data['OW_ID'][i]
            align = df_data['ALIGN'][i]
            mapped_len = len(df_mapping_result[(df_mapping_result['wn_offset'] == wn_id) &
                                               (df_mapping_result[mapped_col_name] == ow_id)])
            if mapped_len > 0:
                df_data.loc[i, 'mapped'] = 1
            else:
                df_data.loc[i, 'mapped'] = 0

        tp = len(df_data[(df_data['mapped'] == 1) & (df_data['ALIGN'] == 1)])
        fp = len(df_data[(df_data['mapped'] == 1) & (df_data['ALIGN'] == 0)])
        tn = len(df_data[(df_data['mapped'] == 0) & (df_data['ALIGN'] == 0)])
        fn = len(df_data[(df_data['mapped'] == 0) & (df_data['ALIGN'] == 1)])

        p = tp / (tp + fp)
        r = tp / (tp + fn)
        a = (tp + tn) / (tp + fp + tn + fn)
        f1 = (2 * p * r) / (p + r)
        dict = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

        return df_data, round(p, 3), round(r, 3), round(a, 3), round(f1, 3), dict

    def cal_p_r_f_a_ow_wn(self, df_mapping_result, df_gold_data, mapped_col_name):
        '''
        cal precision, recall, f1, and accuracy
        :param df_mapping_result:
        :param df_gold_data:
        :return:
        '''
        df_data = df_gold_data.copy()
        df_data['WN_OFFSET'] = 'wn:' + df_data['WN_OFFSET'].str.replace('-', '')

        for i in range(len(df_data)):
            wn_id = df_data['WN_OFFSET'][i]
            ow_id = df_data['OW_ID'][i]
            align = df_data['ALIGN'][i]
            mapped_len = len(df_mapping_result[(df_mapping_result[mapped_col_name] == wn_id) &
                                               (df_mapping_result['OW_ID'] == ow_id)])
            if mapped_len > 0:
                df_data.loc[i, 'mapped'] = 1
            else:
                df_data.loc[i, 'mapped'] = 0

        tp = len(df_data[(df_data['mapped'] == 1) & (df_data['ALIGN'] == 1)])
        fp = len(df_data[(df_data['mapped'] == 1) & (df_data['ALIGN'] == 0)])
        tn = len(df_data[(df_data['mapped'] == 0) & (df_data['ALIGN'] == 0)])
        fn = len(df_data[(df_data['mapped'] == 0) & (df_data['ALIGN'] == 1)])

        p = tp / (tp + fp)
        r = tp / (tp + fn)
        a = (tp + tn) / (tp + fp + tn + fn)
        f1 = (2 * p * r) / (p + r)
        dict = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

        return df_data, round(p, 3), round(r, 3), round(a, 3), round(f1, 3), dict

    def analysis_v3(self, mapping_direction, method, tie_breaking, df_result, gold_col_name, mapped_col_name,
                    mapped_unique_col, col_name_bk_ties):
        '''

        :param df_result:
        :param target_lan_lst:
        :return:
        '''

        df_data = df_result.copy()

        acc = self.cal_acc(df_data, gold_col_name, mapped_unique_col)

        if len(mapped_col_name) > 1:
            mrr_t = self.cal_mrr(df_data, gold_col_name, mapped_col_name)
            ties_t = self.get_ties_num(df_data, mapped_col_name)
        else:
            mrr_t = 0
            ties_t = 0
        if len(col_name_bk_ties) > 0:
            ties_after = self.get_ties_num(df_data, col_name_bk_ties)
        else:
            ties_after = 0

        self.df_analysis_v3 = self.df_analysis_v3.append(
            {'mapping_direction': mapping_direction, 'Method': method, 'tie_breaking': tie_breaking,
             'MRR': mrr_t, 'Ties': ties_t, 'Accuracy': acc,
             'ties_after_breaking': ties_after}, ignore_index=True)

        print('\n Method:', method, 'mapping_direction:', mapping_direction, 'tie_breaking:', tie_breaking, 'MRR:',
              mrr_t, 'ties number:', ties_t,
              'acc:', acc, 'ties_after_breaking:', ties_after)

        return self.df_analysis_v3, acc


if __name__ == '__main__':
    None
