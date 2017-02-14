from tasea.utils import Utils


def already_tested(list_sequences, a_sequence):
    return a_sequence in list_sequences


def already_prefixed(list_sequences, a_sequence):
    for one_sequence in list_sequences:
        if one_sequence.sequence == a_sequence[:len(one_sequence.sequence)]:
            return True


def accepted(training_data, a_sequence, accuracy_threshold=80, earliness_threshold=0):
    true_classification = false_classification = 0
    final_time_steps = []
    covered_timeseries = []
    for aMultivariateTimeseries in training_data:
        tp, fp, tn, fn, time_steps = check_pattern(aMultivariateTimeseries, a_sequence)
        if tp or fp:
            if not final_time_steps:
                final_time_steps = time_steps
            else:
                final_time_steps = [max(x, y) for x, y in zip(time_steps, final_time_steps)]
            if tp:
                covered_timeseries.append(aMultivariateTimeseries)

        true_classification += tp + tn
        false_classification += fn + fp
    acc = float(true_classification) / float(true_classification + false_classification)
    if acc * 100 >= accuracy_threshold:
        time_sequence = TimeSequence()
        time_sequence.class_sequence = a_sequence[0].class_shapelet
        time_sequence.sequence = a_sequence
        time_sequence.time_steps = final_time_steps
        time_sequence.covered_instances = len(covered_timeseries)
        time_sequence.covered_instance_names = [t.name for t in covered_timeseries]
        for aMultivariateTimeseries in covered_timeseries:
            aMultivariateTimeseries.matched = True
        return time_sequence
    return False


def check_pattern(multivariate_timeseries, sequence):
    match_index = old_match_index = None
    pattern_found = True
    sequence_class = sequence[0].class_shapelet
    time_steps = []
    for aShapelet in sequence:

        if multivariate_timeseries.name not in aShapelet.matching_indices:
            pattern_found = False
            break

        if not aShapelet.matching_indices[multivariate_timeseries.name]:
            pattern_found = False
            break

        if not match_index:
            match_index = aShapelet.matching_indices[multivariate_timeseries.name][0]
            old_match_index = match_index
        else:
            indices = [index for index in aShapelet.matching_indices[multivariate_timeseries.name] if
                       index > match_index]
            if not indices:
                pattern_found = False
                break
            temp = match_index
            match_index = indices[0]
            time_steps.append(match_index - old_match_index)
            old_match_index = temp

    if sequence_class == multivariate_timeseries.class_timeseries:
        if pattern_found:
            return 1, 0, 0, 0, time_steps
        return 0, 0, 0, 1, []
    else:
        if pattern_found:
            return 0, 1, 0, 0, time_steps
        return 0, 0, 1, 0, []
    return 0, 0, 0, 0, []


class TimeSequence(object):
    def __init__(self):
        self.name = id(self)
        self.dimension_name = "dummy"
        self.class_sequence = ''
        self.sequence = []
        self.time_steps = []
        self.covered_instances = 0
        self.covered_instance_names = []

    def __repr__(self):
        representation = "{"  # Outermost
        representation += '"name": ' + str(self.name) + ','
        # representation += '"dimension_name": "' + str(self.dimension_name) + '",'
        representation += '"class_sequence": "' + str(self.class_sequence) + '",'
        representation += '"sequence":['
        for s in self.sequence:
            representation += s.__repr__()
            representation += ','
        representation = representation[:-1]
        representation += ']' # list of Shapelets
        representation += ','
        representation = Utils.json_list(representation, 'time_steps', self.time_steps)
        representation += '}' # Outermost
        return representation


    def __str__(self):
        representation = "Sequence with class: " + self.class_sequence
        representation += " with sequences: " + str(self.sequence)
        representation += " with time steps: " + str(self.time_steps)
        return representation

    def length_of_shapelets(self):
        l = 0
        for aShapelet in self.sequence:
            l += len(aShapelet.subsequence)
        return l

    @staticmethod
    def filter_sequences(list_time_sequences):
        return list_time_sequences


class PreSequence(object):
    def __init__(self):
        self.class_name = ''
        self.inner_lists = []
