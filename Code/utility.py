def get_mapped_cpt(mapped_dict):
    result_dict = {}
    for source_id in list(mapped_dict.keys()):
        # sorted dictionary items by values
        targets_sorted = sorted(mapped_dict[source_id].items(), key=lambda x: x[1], reverse=True)
        result_dict[source_id] = targets_sorted

    return result_dict


def get_wn_candidates_by_clics(clics_obj, bn_obj, concept_id, clics_lan_lst,
                               df_concept, df_bn_trans, df_clics_trans, concept_lemmas):
    '''
    given a Clics concept id, return WN candidates that shares at least one lemmas with the
    clics concept
    :param clics_obj:
    :param bn_obj:
    :param concept_id:
    :param clics_lan_lst:
    :param df_concept:
    :param df_bn_trans: BN data in a given language set
    :param df_clics_trans: CLICS data in a given language set
    :return:
    '''

    concept_pos = clics_obj.get_clics_pos_by_id(concept_id=concept_id, df_concept=df_concept)
    # get wn candidates
    wn_candidas_tmp = []
    wn_candidas = []
    for flemma in concept_lemmas:
        wn_synsets = bn_obj.extract_synsets_withoutpos(keyword=flemma, df_bn_tran=df_bn_trans)  # a list of WN ids
        wn_candidas_tmp = wn_candidas_tmp + wn_synsets

    if len(concept_pos) > 0:
        for fpos in concept_pos:
            for synset in wn_candidas_tmp:
                if fpos == synset[-1]:
                    wn_candidas.append(synset)
    else:
        wn_candidas = wn_candidas_tmp
    return wn_candidas
