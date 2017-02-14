import numpy as np
from collections import defaultdict
import sys


class TimeSeries(object):
    def __init__(self):
        self.class_timeseries = ''
        self.dimension_name = ''
        self.timeseries = None
        self.matched = False
        self.name = ''

    def __repr__(self):
        representation = "Timeseries with dimension: " + self.dimension_name
        representation += " with class: " + self.class_timeseries
        representation += " with series: " + str(self.timeseries)
        return representation

    def __str__(self):
        representation = "Timeseries with dimension: " + self.dimension_name
        representation += " with class: " + self.class_timeseries
        representation += " with series: " + str(self.timeseries)
        return representation

    @staticmethod
    def generate_from_multivariate_timeseries(list_multivariate_timeseries, key='dimension'):
        dict_timeseries = defaultdict(list)
        for aMultivariateTimeseries in list_multivariate_timeseries:
            the_class = aMultivariateTimeseries.class_timeseries
            the_name = aMultivariateTimeseries.name
            for dimension, relatedTimeseries in aMultivariateTimeseries.timeseries.items():
                timeseries = TimeSeries()
                timeseries.name = the_name
                timeseries.dimension_name = dimension
                timeseries.class_timeseries = the_class
                timeseries.timeseries = relatedTimeseries
                if key == 'dimension':
                    dict_timeseries[dimension].append(timeseries)
                elif key == 'class':
                    dict_timeseries[the_class].append(timeseries)
                else:
                    dict_timeseries[(dimension, the_class)].append(timeseries)

        return dict_timeseries

    @staticmethod
    def generate_timeseries(unid):
        unid_timeseries = []
        for cuts in unid:
            target_class = cuts[-1]
            ts_str = cuts[:-1]
            ts = [float(element) for element in ts_str]
            timeseries = TimeSeries()
            timeseries.class_timeseries = target_class
            timeseries.timeseries = np.array(ts)
            unid_timeseries.append(timeseries)
        return unid_timeseries


class MultivariateTimeSeries(object):

    DIMENSION_NAMES = np.array([])
    CLASS_NAMES = set()

    def __init__(self):
        self.class_timeseries = ''
        self.timeseries = {}
        self.name = ''
        self.encoded_sequence = []
        self.encoded_sequence_gain = 0.0
        self.matched = False

    def length(self, key='min'):
        if key == 'min':
            min_l = sys.maxsize
            for dim in MultivariateTimeSeries.DIMENSION_NAMES:
                if len(self.timeseries[dim]) < min_l:
                    min_l = len(self.timeseries[dim])
            return min_l
        return len(self.timeseries[MultivariateTimeSeries.DIMENSION_NAMES[0]])

    @staticmethod
    def generate_timeseries(list_cmts, dimension_names=np.array([])):
        multivariate_timeseries = []
        if dimension_names.size == 0:
            dimension_names = list_cmts[0][0, 1:]
        for cmts in list_cmts:
            timeseries = MultivariateTimeSeries()
            timeseries.class_timeseries = cmts[-1, 0]
            for i in range(len(dimension_names)):
                timeseries.timeseries[dimension_names[i]] = cmts[1:-1, i + 1].astype(float)
            multivariate_timeseries.append(timeseries)
        return multivariate_timeseries

    @staticmethod
    def generate_from_file(directory, file):
        mts = np.genfromtxt(directory + "/" + file, delimiter=';', dtype=str)
        if MultivariateTimeSeries.DIMENSION_NAMES.size == 0:
            MultivariateTimeSeries.DIMENSION_NAMES = mts[0, 1:]
        timeseries = MultivariateTimeSeries()
        timeseries.name = file
        timeseries.class_timeseries = mts[-1, 0]
        MultivariateTimeSeries.CLASS_NAMES.add(timeseries.class_timeseries)
        for i in range(len(MultivariateTimeSeries.DIMENSION_NAMES)):
            timeseries.timeseries[MultivariateTimeSeries.DIMENSION_NAMES[i]] = mts[1:-1, i+1].astype(float)
        return timeseries

    @staticmethod
    def encode_all_with_shapelets(list_multivariate_timeseries, list_shapelets):
        for aTimeseries in list_multivariate_timeseries:
            aTimeseries.encode_with_shapelets(list_shapelets)

        return list_multivariate_timeseries

    def encode_with_shapelets(self, list_shapelets, n=1):
        # Encode a multivariate time series with a single list that contains a tuple (shapelet, where it occurs)
        # This list is sorted at the end regarding the occurring time
        encoding_list = []

        for aShapelet in list_shapelets:
            encoding_list += [(aShapelet, x) for x in aShapelet.matching_indices[self.name]
                              if aShapelet.class_shapelet == self.class_timeseries]

        temp_encoded_sequence = [shapelet for (shapelet, time) in sorted(encoding_list, key=lambda x: x[1])]

        added_dimensions = defaultdict(lambda: 0)

        if self.encoded_sequence:
            self.encoded_sequence = []
        for aShapelet in temp_encoded_sequence:

            dim = aShapelet.dimension_name

            if added_dimensions[dim] == n:
                continue

            added_dimensions[dim] += 1
            self.encoded_sequence.append(aShapelet)

        self.encoded_sequence_gain = sum(shapelet.gain for shapelet in self.encoded_sequence)

    @staticmethod
    def get_timeseries_by_name(list_multivariate_timeseries, name):
        for t in list_multivariate_timeseries:
            if t.name == name:
                return t
        return None