s = 'results/prediction/grid_min_GraphGruRnn__layers_4_hidden_64_predictions_epoch_18_sample_3.dat'
parsed = s.split('_')


def extract_value_from_file(f, key, file_type='.dat'):
    """
    @param file_type:
    @param f:
    @param key:
    @return:
    """
    proceed = s.split('_')
    for i, k in enumerate(proceed):
        if key in k:
            v = parsed[i + 1]
            if file_type in v:
                return v[:-len(file_type)]
            else:
                return v


print(extract_value_from_file(s, "epoch"))
print(extract_value_from_file(s, "sample"))