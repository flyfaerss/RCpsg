import os
import torch
import numpy as np
import pickle


def resampling_dict_generation(dataset, resampling_method, repeat_factor, predicate_list, output_dir):
    print("\n using resampling method: " + resampling_method)
    curr_dir_repeat_dict = os.path.join(output_dir, "repeat_dict.pkl")
    if os.path.exists(curr_dir_repeat_dict):
        print("load repeat_dict from " + curr_dir_repeat_dict)
        with open(curr_dir_repeat_dict, 'rb') as f:
            repeat_dict = pickle.load(f)
        return repeat_dict
    else:
        if resampling_method == "faip":
            print("generate the balance sample by recurrent the predicate")

            predicate_times = np.zeros(len(predicate_list) + 1)
            for i in range(len(dataset)):
                relation_anno = dataset[i]['relations']
                for each_rel in relation_anno:
                    predicate_times[each_rel[2]] += 1

            predicate_total = sum(predicate_times)
            predicate_frequency = predicate_times[1:] / (predicate_total + 1e-11)

            predicate_dict = [[] for _ in range(len(predicate_list) + 1)]
            image2predicate = np.zeros(len(dataset))
            for i in range(len(dataset)):
                relation_anno = dataset[i]['relations']
                min_predicate = relation_anno[0][2]
                for each_rel in relation_anno:
                    if predicate_frequency[min_predicate - 1] > predicate_frequency[each_rel[2] - 1]:
                        min_predicate = each_rel[2]
                image2predicate[i] = min_predicate
                predicate_dict[min_predicate].append(i)
            predicate_length = [len(predicate_dict[i + 1]) for i in range(len(predicate_list))]
            repeat_dict = []
            predicate_end_count = np.zeros(len(predicate_list))
            predicate_loc = np.zeros(len(predicate_list))

            end_count = 28

            while int(np.sum(predicate_end_count)) < end_count:
                new_turn = [predicate_dict[i + 1][int(predicate_loc[i])] for i in range(len(predicate_list))]
                for i in range(len(predicate_list)):
                    if predicate_loc[i] + 1 < predicate_length[i]:
                        predicate_loc[i] = predicate_loc[i] + 1
                    else:
                        predicate_loc[i] = 0
                        predicate_end_count[i] = 1
                repeat_dict.extend(new_turn)

            rest_start_index = predicate_loc[predicate_end_count != 1]
            rest_index = [i for i, x in enumerate(predicate_end_count) if x != 1]
            for i in range(len(rest_index)):
                repeat_dict.extend(predicate_dict[rest_index[i] + 1][int(rest_start_index[i]):])
            # when we use the lvis sampling method,
            # global_rf = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR
            # logger.info(f"global repeat factor: {global_rf};  ")
        elif resampling_method == 'bilvl':
            repeat_dict = None
        else:
            raise NotImplementedError(resampling_method)

        return repeat_dict


def generate_resample_dataset(dataset, repeat_dict):
    resample_dataset = [dataset[i] for i in repeat_dict]
    return resample_dataset


def apply_dropping(data, drop_rate):
    pass


