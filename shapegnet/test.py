s = 'results/prediction/grid_min_GraphGruRnn__layers_4_hidden_64_predictions_epoch_18_sample_3.dat'
parsed = s.split('_')


def extract_value_from_file(f, key):
    """

    @param f:
    @param key:
    @return:
    """
    parsed = s.split('_')
    for i, k in enumerate(parsed):
        if key in k:
            return parsed[i + 1]

extract_value_from_file(s, "epoch")
