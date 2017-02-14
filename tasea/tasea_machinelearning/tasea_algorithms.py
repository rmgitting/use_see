from tasea.tasea_machinelearning.timeseries import TimeSeries
from tasea.tasea_machinelearning.timeseries import MultivariateTimeSeries
from tasea.utils import Utils

from tasea.ust_thread.thread_learning import LearningThread
from tasea.ust_thread.thread_gui import GUIThread
import tasea.tasea_machinelearning.learning_algorithms as lrn
import tasea.tasea_machinelearning.sequences as seq
import itertools
import operator
import threading
import gc
import sys


def use_v4(list_multivariate_timeseries, min_length=None, max_length=None, pruning='cover', k=10
           , distance_measure='brute'):
    # USE with psutil support
    if not min_length:
        length = Utils.min_length_dataset(list_multivariate_timeseries)
        min_length = int(length * 0.3)
        if length >= 40:
            max_length = int(length * 0.5)
        else:
            max_length = length - 1

    dict_timeseries_by_dimension = TimeSeries.generate_from_multivariate_timeseries(
        list_multivariate_timeseries, key='dimension')
    dict_timeseries_by_dimension_class = TimeSeries.generate_from_multivariate_timeseries(
        list_multivariate_timeseries, key='dimension|class')
    list_all_shapelets_pruned = []
    print("Detecting " + str(len(list_multivariate_timeseries[0].timeseries.keys())) + " dimensions")

    list_timeseries_by_dimension = dict_timeseries_by_dimension.values()

    for listTimeSeries in list_timeseries_by_dimension:
        done = False
        list_remaining_cands = None
        while not done:
            list_done_shapelets, list_remaining_cands = lrn.uts_brute_force_v3(listTimeSeries, min_length, max_length,
                                                                               distance_measure=distance_measure,
                                                                               list_remaining_cands=
                                                                               list_remaining_cands)
            if not list_remaining_cands:
                done = True
            grouped_shapelets = itertools.groupby(list_done_shapelets, lambda shapelet: shapelet.class_shapelet)
            print("Starting the pruning procedure...")
            i = 0
            length = len(list_done_shapelets)
            Utils.print_progress(i, length)
            for keyShapelet, groupShapelet in grouped_shapelets:
                list_shapelet_group = list(groupShapelet)
                dimension = list_shapelet_group[0].dimension_name
                the_class = list_shapelet_group[0].class_shapelet
                list_all_shapelets_pruned += lrn.pruning(list_shapelet_group, algorithm=pruning,
                                                         training_data=dict_timeseries_by_dimension_class[
                                                             (dimension, the_class)])
                i += len(list_shapelet_group)
                Utils.print_progress(i, length)
            print("Pruning complete")
            print("*************************")
            print()
            print()
            list_done_shapelets = None
            gc.collect()

    print("Calculating the matching indices...")
    i = 0
    length = len(list_all_shapelets_pruned)
    Utils.print_progress(i, length)
    for aShapelet in list_all_shapelets_pruned:
        aShapelet.build_matching_indices()
        i += 1
        Utils.print_progress(i, length)
    print("Calculation complete...")
    print("*************************")
    print()
    return list_all_shapelets_pruned


def use_v3(list_multivariate_timeseries, min_length=None, max_length=None, pruning='cover', k=10, multi_threading=False
           , distance_measure='brute'):
    if not min_length:
        length = Utils.min_length_dataset(list_multivariate_timeseries)
        min_length = int(length * 0.3)
        if length >= 40:
            max_length = int(length * 0.5)
        else:
            max_length = length

    dict_timeseries_by_dimension = TimeSeries.generate_from_multivariate_timeseries(
        list_multivariate_timeseries, key='dimension')
    dict_timeseries_by_dimension_class = TimeSeries.generate_from_multivariate_timeseries(
        list_multivariate_timeseries, key='dimension|class')
    list_all_shapelets_pruned = []
    print("Detecting " + str(len(list_multivariate_timeseries[0].timeseries.keys())) + " dimensions")
    # Run the Brute Force Algorithm and do the pruning for each sigma unid in order to learn the shapelets
    #  for this specific dimension
    list_timeseries_by_dimension = dict_timeseries_by_dimension.values()
    for listTimeSeries in list_timeseries_by_dimension:
        list_all_shapelets = lrn.uts_brute_force_v2(listTimeSeries, min_length, max_length,
                                                    distance_measure=distance_measure)
        grouped_shapelets = itertools.groupby(list_all_shapelets, lambda shapelet: shapelet.class_shapelet)
        print("Starting the pruning procedure...")
        i = 0
        length = len(list_all_shapelets)
        Utils.print_progress(i, length)
        for keyShapelet, groupShapelet in grouped_shapelets:
            list_shapelet_group = list(groupShapelet)
            dimension = list_shapelet_group[0].dimension_name
            the_class = list_shapelet_group[0].class_shapelet
            list_all_shapelets_pruned += lrn.pruning(list_shapelet_group, algorithm=pruning,
                                                     training_data=dict_timeseries_by_dimension_class[
                                                         (dimension, the_class)])
            i += len(list_shapelet_group)
            Utils.print_progress(i, length)
        print("Pruning complete")
        print("*************************")
        print()
        print()
        list_all_shapelets = None
        gc.collect()

    print("Calculating the matching indices...")
    i = 0
    length = len(list_all_shapelets_pruned)
    Utils.print_progress(i, length)
    for aShapelet in list_all_shapelets_pruned:
        aShapelet.build_matching_indices()
        i += 1
        Utils.print_progress(i, length)
    print("Calculation complete...")
    print("*************************")
    print()
    return list_all_shapelets_pruned


def use_v2(list_multivariate_timeseries, min_length=50, max_length=50, pruning='cover', k=10, multi_threading=False
           , distance_measure='brute'):
    dict_timeseries = TimeSeries.generate_from_multivariate_timeseries(
        list_multivariate_timeseries, key='dimension')

    list_all_shapelets = []
    print("Detecting " + str(len(list_multivariate_timeseries[0].timeseries.keys())) + " dimensions")
    # Run the Brute Force Algorithm for each sigma unid in order to learn the shapelets for this specific dimension
    list_timeseries_by_dimension = dict_timeseries.values()
    if multi_threading:
        thread_gui = GUIThread()
        thread_gui.start()
        lock = threading.Lock()
        threads = []
        for listTimeSeries in list_timeseries_by_dimension:
            t = LearningThread(listTimeSeries, min_length, max_length, list_all_shapelets, lock, thread_gui,
                               distance_measure=distance_measure)
            t.start()
            threads.append(t)
        print("Waiting all threads to finish...")
        for t in threads:
            t.join()
        print("Waiting complete")
        print("*************************")
        print()

    else:
        for listTimeSeries in list_timeseries_by_dimension:
            list_all_shapelets += lrn.uts_brute_force_v2(listTimeSeries, min_length, max_length,
                                                         distance_measure=distance_measure)

    dict_timeseries = TimeSeries.generate_from_multivariate_timeseries(
        list_multivariate_timeseries, key='dimension|class')

    grouped_shapelets = itertools.groupby(list_all_shapelets,
                                          lambda shapelet: (shapelet.dimension_name, shapelet.class_shapelet))

    list_all_shapelets_pruned = []
    print("Starting the pruning procedure...")
    i = 0
    length = len(list_all_shapelets)
    Utils.print_progress(i, length)
    for keyShapelet, groupShapelet in grouped_shapelets:
        list_shapelet_group = list(groupShapelet)
        dimension = list_shapelet_group[0].dimension_name
        the_class = list_shapelet_group[0].class_shapelet
        list_all_shapelets_pruned += lrn.pruning(list_shapelet_group, algorithm=pruning,
                                                 training_data=dict_timeseries[(dimension, the_class)])
        i += len(list_shapelet_group)
        Utils.print_progress(i, length)

    print("Pruning complete")
    print("*************************")
    print()
    print()

    print("Calculating the matching indices...")
    i = 0
    length = len(list_all_shapelets_pruned)
    Utils.print_progress(i, length)
    for aShapelet in list_all_shapelets_pruned:
        aShapelet.build_matching_indices()
        i += 1
        Utils.print_progress(i, length)
    print("Calculation complete...")
    print("*************************")
    print()
    return list_all_shapelets_pruned


def see(list_all_shapelets, training_dataset, accuracy_threshold=80, earliness_threshold=0, multi_threading=True):
    # The algorithm needs first to create the PreSequences in order for the combinations and permutations
    #  to be computed
    list_presequences = []

    print("Preparing the shapelets...")
    # First, create a list of lists, each inner list contains the shapelets that belong to the same class
    get_attr = operator.attrgetter('class_shapelet')
    list_of_lists_shapelet = [list(g) for k, g in
                              itertools.groupby(sorted(list_all_shapelets, key=get_attr), get_attr)]

    # Second, further divide the inner lists from above, to create inner lists of shapelets that belong to the
    #  same class, but divided regarding the dimension
    get_attr = operator.attrgetter('dimension_name')
    for inner_list in list_of_lists_shapelet:
        presequence = seq.PreSequence()
        presequence.class_name = inner_list[0].class_shapelet
        presequence.inner_lists = [list(g) for k, g in
                                   itertools.groupby(sorted(inner_list, key=get_attr), get_attr)]
        list_presequences.append(presequence)
    print("Preparation complete...")
    print("*************************")
    print()
    # The list of presequences is now ready for the permutation and combination
    # In each presequence, the inner_lists is a list
    # that contains lists of shapelets that belong to the same dimension

    list_all_accepted_sequences = []

    for aPresequence in list_presequences:
        print("Building sequences for the class:" + aPresequence.class_name + "...")
        # For the inner lists in each presequence object create all the possible combinations
        # Getting one shapelet from each dimension
        # And they all belong to the same class, by the definition of the PreSequence class
        list_already_tested_sequences = []
        list_accepted_sequences = []
        all_possible_combinations = itertools.product(*aPresequence.inner_lists)
        all_possible_combinations_list = list(all_possible_combinations)
        length = len(all_possible_combinations_list)
        j = 0
        Utils.print_progress(j, length)
        for aCombination in all_possible_combinations_list:
            for i in range(len(aCombination)):
                # Get the permutations that are possible for each combination
                all_possible_permutations = itertools.permutations(aCombination, i + 1)

                for aSequence in all_possible_permutations:
                    # Get the sequence at this stage and start working with it
                    sequence_as_list = list(aSequence)
                    if seq.already_tested(list_already_tested_sequences, sequence_as_list):
                        # If this sequence is already tested then discard
                        continue
                    list_already_tested_sequences.append(sequence_as_list)

                    if seq.already_prefixed(list_accepted_sequences, sequence_as_list):
                        # If an accepted shorter sequence prefixes this one then discard
                        continue
                    time_sequence = seq.accepted(training_dataset, sequence_as_list,
                                                 accuracy_threshold=accuracy_threshold)
                    if time_sequence:
                        # The actual testing of the sequence against
                        # the training dataset is in the accepted function
                        list_accepted_sequences.append(time_sequence)
            j += 1
            Utils.print_progress(j, length)
        print("Building complete...")
        print("*************************")
        print()
        list_all_accepted_sequences += list_accepted_sequences
    return list_all_accepted_sequences


def see_v2(list_all_shapelets, training_dataset, accuracy_threshold=80, earliness_threshold=0, multi_threading=True):
    training_dataset = MultivariateTimeSeries.encode_all_with_shapelets(training_dataset, list_all_shapelets)

    list_all_accepted_sequences = []

    for aClass in MultivariateTimeSeries.CLASS_NAMES:
        list_already_tested_sequences = []
        list_accepted_sequences = []

        list_multivariate_timeseries = sorted(training_dataset, key=lambda ts: ts.encoded_sequence_gain, reverse=True)
        print("Building sequences for the class: " + aClass + "...")

        length = len(list_multivariate_timeseries)
        j = 0
        Utils.print_progress(j, length)
        for aMultivariateTimeseries in list_multivariate_timeseries:
            if aMultivariateTimeseries.class_timeseries == aClass:
                for i in range(len(aMultivariateTimeseries.encoded_sequence)):
                    # Get the permutations that are possible for each combination
                    all_possible_permutations = itertools.permutations(aMultivariateTimeseries.encoded_sequence, i + 1)
                    for aSequence in all_possible_permutations:
                        # Get the sequence at this stage and start working with it

                        sequence_as_list = list(aSequence)

                        if seq.already_tested(list_already_tested_sequences, sequence_as_list):
                            # If this sequence is already tested then discard
                            continue
                        list_already_tested_sequences.append(sequence_as_list)

                        if seq.already_prefixed(list_accepted_sequences, sequence_as_list):
                            # If an accepted shorter sequence prefixes this one then discard
                            continue
                        time_sequence = seq.accepted(training_dataset, sequence_as_list,
                                                     accuracy_threshold=accuracy_threshold)

                        if time_sequence:
                            # The actual testing of the sequence against
                            # the training dataset is in the accepted function
                            list_accepted_sequences.append(time_sequence)
            j += 1
            Utils.print_progress(j, length)

        print("Building complete...")
        print("*************************")
        print()
        list_all_accepted_sequences += list_accepted_sequences

    return list_all_accepted_sequences


def taqe(list_sequences, list_multivariate_timeseries):
    return seq.TimeSequence.filter_sequences(list_sequences)
