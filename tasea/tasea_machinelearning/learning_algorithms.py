from tasea.tasea_machinelearning.shapelet import *


def uts_brute_force_v3(list_time_series, min_length, max_length, distance_measure='brute', skip=False,
                       list_remaining_cands=None):

    if not list_remaining_cands:
        print("Generating candidate shapelets for dimension: " + list_time_series[0].dimension_name + " ...")
        candidate_shapelets = Shapelet.generate_candidates(list_time_series, min_length, max_length, skip=skip)
        print("Generation complete")
    else:
        print("Continue the learning on dimension: " + list_time_series[0].dimension_name + " ...")
        candidate_shapelets = list_remaining_cands

    i = 0
    length = len(candidate_shapelets)
    print("Calculating the different features of the candidate shapelets...")
    Utils.print_progress(i, length)
    for candidate in candidate_shapelets:
        candidate.gain, candidate.dist_threshold = candidate.check(list_time_series, distance_measure=distance_measure)
        i += 1
        Utils.print_progress(i, length)
        good = Utils.check_memory()
        if not good:
            print()
            print("Memory usage is more that 90%")
            print("Forcing a prune phase, and then the learning will continue from where it stopped")
            return candidate_shapelets[:i], candidate_shapelets[i:]
    print("Calculation complete for dimension: ", list_time_series[0].dimension_name)
    print("*************************")
    # Return all the candidates
    return candidate_shapelets, None


def uts_brute_force_v2(list_time_series, min_length, max_length, list_all_shapelets=[], lock=None, label=None,
                       progress=None, distance_measure='brute', skip=True):

    if label:
        lock.acquire()
        label['text'] = "Generating candidate shapelets for dimension: " + list_time_series[0].dimension_name
        lock.release()

    if not label:
        print("Generating candidate shapelets for dimension: " + list_time_series[0].dimension_name + " ...")
    candidate_shapelets = Shapelet.generate_candidates(list_time_series, min_length, max_length, skip=skip)
    if not label:
        print("Generation complete")

    i = 0
    length = len(candidate_shapelets)
    if not label:
        print("Calculating the different features of the candidate shapelets...")
        Utils.print_progress(i, length)
    else:
        lock.acquire()
        progress['value'] = i
        progress['maximum'] = length
        lock.release()
    for candidate in candidate_shapelets:
        candidate.gain, candidate.dist_threshold = candidate.check(list_time_series, distance_measure=distance_measure)
        i += 1
        if not label:
            Utils.print_progress(i, length)
        else:
            lock.acquire()
            progress['value'] = i
            lock.release()
    if not label:
        print("Calculation complete for dimension: ", list_time_series[0].dimension_name)
        print("*************************")
    # Return all the candidates
    if not lock:
        return candidate_shapelets

    # For multi-threading use
    lock.acquire()
    list_all_shapelets += candidate_shapelets
    lock.release()


def pruning(list_shapelets, algorithm='top-k', k=6, training_data=None):
    """
    :param list_shapelets: takes input as a list of shapelets
    :param algorithm: a parametrisation in order to add additional pruning algorithms
    :param k: if the algorithm is top_k we need to specify how much k we have to obtain from this pruning
    :param training_data: the training data set as a dict of timeseries by dimension
    :return: a list of pruned shapelets
    """
    if not list_shapelets:
        return []
    list_shapelets.sort(key=lambda x: x.gain, reverse=True)

    if algorithm == "top-k":
        k /= 2
        return list_shapelets[:int(k)]

    if algorithm == "cover":
        if not training_data:
            return []

        list_selected_shapelets = []
        for aShapelet in list_shapelets:
            cover, remaining = aShapelet.cover(training_data)
            if cover:
                list_selected_shapelets.append(aShapelet)
            if not remaining:
                break
        return list_selected_shapelets

    else:
        return []
