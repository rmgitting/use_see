from builtins import print

import tasea.tasea_machinelearning.similarity_measures as sm
import sys
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def check_performance(list_multivariate_timeseries, list_sequences, distance_measure='brute', key='closest|majority'):
    tc = fc = 0
    avg_length = 0
    y_pred_maj = []
    y_true = []
    y_pred = []
    instance = 1
    for_app = 0
    key = key.split('|')
    for a_multivariate_timeseries in list_multivariate_timeseries:
        avg_f_dict = defaultdict(int)

        ts_class = a_multivariate_timeseries.class_timeseries
        min_seq_length = sys.maxsize
        true_classification = false_classification = 0
        predicted_class_distance = ''
        min_distance = sys.maxsize
        y_true.append(ts_class)

        tp_bool = False
        for a_sequence in list_sequences:
            sequence_class = a_sequence.sequence[0].class_shapelet
            seq_found, win_length, min_dist = pattern_found(a_multivariate_timeseries, a_sequence,
                                                            distance_measure=distance_measure)
            if sequence_class == ts_class:
                if seq_found:  # TP
                    # For Accuracy
                    true_classification += 1

                    avg_f_dict[sequence_class] += 1

                    # For Earliness
                    min_seq_length = min(min_seq_length, win_length + a_sequence.length_of_shapelets())

                    tp_bool = True

                    if min_dist <= min_distance:
                        min_distance = min_dist
                        predicted_class_distance = sequence_class

                else:  # FN
                    # For Accuracy
                    false_classification += 1

            else:
                if seq_found:  # FP
                    # For Accuracy
                    false_classification += 1

                    avg_f_dict[sequence_class] += 1

                    if min_dist < min_distance:
                        min_distance = min_dist
                        predicted_class_distance = sequence_class

                else:  # TN
                    # For Accuracy
                    true_classification += 1

        if (key[1] and key[1] == 'majority') or key[0] == 'majority':
            predicted_class = ""
            predicted = 0
            total = 0
            for aKey in avg_f_dict:
                if avg_f_dict[aKey] > predicted:
                    predicted_class = aKey
                    predicted = avg_f_dict[aKey]
                total += avg_f_dict[aKey]

            y_pred_maj.append(predicted_class)
            print("*" * 80)
            print("Time series number", instance, "with name:", a_multivariate_timeseries.name)
            print("True Class:", ts_class)
            print("Predicted Classes:")
            for aKey in avg_f_dict:
                print("\t", aKey, ":", round(avg_f_dict[aKey] / total, 2) * 100, "%")
            print("*" * 80)
            instance += 1

        if key[0] == 'closest':
            y_pred.append(predicted_class_distance)

        if not predicted_class_distance and not predicted_class:
            for_app += 1

        if true_classification >= false_classification:
            tc += 1
            if tp_bool:
                avg = min_seq_length / a_multivariate_timeseries.length()
                if avg > 1:
                    avg = 1
                avg_length += avg
        else:
            fc += 1

    app = (len(list_multivariate_timeseries) - for_app) / float(len(list_multivariate_timeseries))
    avg_length /= len(list_multivariate_timeseries)
    # acc = float(tc) / float(tc + fc)
    sk_acc = sk_report = sk_acc_maj = sk_report_maj = 0
    if y_pred:
        sk_acc = accuracy_score(y_true, y_pred)
        sk_precision, sk_recall, sk_fscore, sk_support = precision_recall_fscore_support(y_true, y_pred,
                                                                                         average='macro')
        sk_report = classification_report(y_true, y_pred)
    if y_pred_maj:
        sk_acc_maj = accuracy_score(y_true, y_pred_maj)
        sk_precision_maj, sk_recall_maj, sk_fscore_maj, sk_support_maj = precision_recall_fscore_support(y_true,
                                                                                                         y_pred_maj,
                                                                                                         average=
                                                                                                         'macro')

    acc = sk_acc
    if sk_acc < sk_acc_maj:
        acc = sk_acc_maj
        sk_report = classification_report(y_true, y_pred_maj)

    return acc * 100, avg_length * 100, sk_acc * 100, sk_report, sk_acc_maj * 100, sk_report_maj, app * 100


def pattern_found(a_multivariate_timeseries, a_sequence, distance_measure='brute'):
    step_index = -1
    match_index = None
    window_length = 0
    min_distance = 0
    for aShapelet in a_sequence.sequence:
        distances = sm.calculate_distances(
            a_multivariate_timeseries.timeseries[aShapelet.dimension_name],
            aShapelet.subsequence, key=distance_measure)
        min_distance += distances.min()
        matching_indices = [index for index, item in
                            enumerate(distances) if item <= aShapelet.dist_threshold]
        if not matching_indices:
            return False, None, 0

        if step_index == -1:
            match_index = matching_indices[0]

        else:
            if len(a_sequence.time_steps) <= step_index:
                a_sequence.time_steps.append(sys.maxsize)
            temp_indices = [i for i in matching_indices if
                            i >= match_index and i - match_index <= a_sequence.time_steps[step_index]]
            if not temp_indices:
                return False, None, 0
            window_length += temp_indices[0] - match_index
            match_index = temp_indices[0]

        step_index += 1

    return True, window_length, min_distance
