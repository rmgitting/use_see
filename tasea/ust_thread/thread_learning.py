import threading
import tasea.tasea_machinelearning.learning_algorithms as lrn


class LearningThread(threading.Thread):
    def __init__(self, list_time_series, min_length=3, max_length=5, list_all_shapelets=[], lock=None, thread_gui=None,
                 distance_measure='brute'):
        threading.Thread.__init__(self)
        self.list_time_series = list_time_series
        self.min_length = min_length
        self.max_length = max_length
        self.list_all_shapelets = list_all_shapelets
        self.lock = lock
        self.thread_gui = thread_gui
        self.distance_measure = distance_measure

    def run(self):
        self.lock.acquire()
        label, progress = self.thread_gui.add_progress_bar()
        self.lock.release()
        lrn.uts_brute_force_v2(self.list_time_series, self.min_length, self.max_length, self.list_all_shapelets,
                               self.lock, label, progress, distance_measure=self.distance_measure)
