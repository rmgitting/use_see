import matplotlib.pyplot as plt
import threading
from tasea.utils import Utils
from tasea.tasea_machinelearning.timeseries import MultivariateTimeSeries


# TO UPDATE


class PlottingThread(threading.Thread):
    def __init__(self, list_multivariate_timeseries, list_all_shapelets=[], list_all_sequences=[]):
        threading.Thread.__init__(self)
        self.list_all_shapelets = list_all_shapelets
        self.list_all_Sequences = list_all_sequences
        self.list_multivariate_timeseries = list_multivariate_timeseries

    def run(self):
        if self.list_all_shapelets:
            PlottingThread.plot_shapelets(self.list_all_shapelets, self.list_multivariate_timeseries)
        elif self.list_all_Sequences:
            PlottingThread.plot_sequences(self.list_all_Sequences, self.list_multivariate_timeseries)

    @staticmethod
    def plot_shapelets(list_all_shapelets, list_multivariate_timeseries):
        colors = ['navy', 'red']
        classes = MultivariateTimeSeries.CLASS_NAMES
        if len(classes) < len(colors):
            colors = colors[:len(classes)]
        elif len(classes) > len(colors):
            for i in range(len(colors), len(classes)):
                colors.append(Utils.generate_new_color(colors))
        color_dict = dict(zip(classes, colors))

        matching_timeseries_index = 0
        not_matching_timeseries_index = 0
        shapelet_index = 0

        def plot_all_plots():
            nonlocal matching_timeseries_index
            nonlocal not_matching_timeseries_index

            shapelet_to_draw = list_all_shapelets[shapelet_index]
            matching_timeseries, matching_timeseries_index = shapelet_to_draw.get_matching_timeseries(
                list_multivariate_timeseries, matching_timeseries_index)
            not_matching_timeseries, not_matching_timeseries_index = shapelet_to_draw.get_not_matching_timeseries(
                list_multivariate_timeseries, not_matching_timeseries_index)
            parent_timeseries, occ_index = shapelet_to_draw.get_parent_timeseries(list_multivariate_timeseries)
            lines = []
            labels = []
            the_class = shapelet_to_draw.class_shapelet

            ax1.cla()
            ax1.set_title("The Shapelet [dim="+ shapelet_to_draw.dimension_name+ "] (press left and right to navigate)", fontsize=12)
            # ax1.set_yticks([])
            ax1.set_xticks([])
            line, = ax1.plot(shapelet_to_draw.subsequence, c=color_dict[the_class])

            ax2.cla()
            ax2.set_title("Parent Timeseries:" + parent_timeseries.name, fontsize=12)
            # ax2.set_yticks([])
            ax2.set_xticks([])
            ax2.plot(parent_timeseries.timeseries[shapelet_to_draw.dimension_name],
                     c=color_dict[the_class])
            x = list(range(occ_index, occ_index + len(shapelet_to_draw.subsequence)))
            ax2.plot(x, shapelet_to_draw.subsequence, linewidth=8, c='green')

            ax3.cla()
            ax3.set_title("Samples of timeseries with same class as the shapelet", fontsize=12)
            ax3.set_xticks([])
            # ax3.set_yticks([])
            for aTimeseries in matching_timeseries:
                ax3.plot(aTimeseries.timeseries[shapelet_to_draw.dimension_name], c=color_dict[the_class])

            lines.append(line)
            labels.append(the_class)
            temp_classes = set(classes)
            ax4.cla()
            ax4.set_title("Sample of Timeseries from other classes", fontsize=12)
            ax4.set_xticks([])
            # ax4.set_yticks([])

            for aTimeseries in not_matching_timeseries:
                a_class = aTimeseries.class_timeseries
                if a_class in temp_classes:
                    line, = ax4.plot(aTimeseries.timeseries[shapelet_to_draw.dimension_name], c=color_dict[a_class])
                    labels.append(a_class)
                    lines.append(line)
                    temp_classes.remove(a_class)
                else:
                    ax4.plot(aTimeseries.timeseries[shapelet_to_draw.dimension_name], c=color_dict[a_class])
            plt.figlegend(lines, labels, loc='lower center')
            max_x = max(ax1.get_xlim()[1], ax2.get_xlim()[1], ax3.get_xlim()[1], ax4.get_xlim()[1])
            ax1.set_xlim([-1, max_x + 10])
            ax2.set_xlim([-1, max_x + 10])
            ax3.set_xlim([-1, max_x + 10])
            ax4.set_xlim([-1, max_x + 10])

            max_y = max(ax1.get_ylim()[1], ax2.get_ylim()[1], ax3.get_ylim()[1], ax4.get_ylim()[1])
            min_y = min(ax1.get_ylim()[0], ax2.get_ylim()[0], ax3.get_ylim()[0], ax4.get_ylim()[0])
            ax1.set_ylim([min_y - 50, max_y + 100])
            ax2.set_ylim([min_y - 50, max_y + 100])
            ax3.set_ylim([min_y - 50, max_y + 100])
            ax4.set_ylim([min_y - 50, max_y + 100])
            fig.canvas.draw()

        def key_event(e):
            nonlocal shapelet_index
            nonlocal matching_timeseries_index
            nonlocal not_matching_timeseries_index

            if e.key == "right":
                shapelet_index += 1
                if shapelet_index == len(list_all_shapelets):
                    shapelet_index = 0
                matching_timeseries_index = 0
                not_matching_timeseries_index = 0
                plot_all_plots()
            elif e.key == "left":
                shapelet_index -= 1
                if shapelet_index < 0:
                    shapelet_index = len(list_all_shapelets) - 1
                matching_timeseries_index = 0
                not_matching_timeseries_index = 0
                plot_all_plots()
            # elif e.key == 'a':
            #     matching_timeseries_index -= k * 2
            #     if matching_timeseries_index < 0:
            #         matching_timeseries_index = 0
            #     plot_all_plots()
            # elif e.key == 'd':
            #     plot_all_plots()
            # elif e.key == 'q':
            #     not_matching_timeseries_index -= k * 2
            #     if not_matching_timeseries_index < 0:
            #         not_matching_timeseries_index = 0
            #     plot_all_plots()
            # elif e.key == 'e':
            #     plot_all_plots()
            else:
                return

        fig = plt.figure(1)
        plt.title("Shapelets Explorer", fontsize=24)
        fig.canvas.mpl_connect('key_press_event', key_event)
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224)
        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        plot_all_plots()
        plt.show()

    @staticmethod
    def plot_sequences(list_all_sequence, list_multivariate_timeseries):
        colors = ['navy', 'red', 'green', 'purple', 'black', 'magenta']
        dims = MultivariateTimeSeries.DIMENSION_NAMES
        if len(dims) < len(colors):
            colors = colors[:len(dims)]
        elif len(dims) > len(colors):
            for i in range(len(colors), len(dims)):
                colors.append(Utils.generate_new_color(colors))
        color_dict = dict(zip(dims, colors))

        sequence_index = 0
        matching_timeseries_index = 0

        def plot_all_plots():

            sequence_to_draw = list_all_sequence[sequence_index]
            name = sequence_to_draw.covered_instance_names[matching_timeseries_index]
            timeseries_to_draw = MultivariateTimeSeries.get_timeseries_by_name(list_multivariate_timeseries, name)
            ax1.cla()
            ax1.set_title("The Rule [class=" + sequence_to_draw.class_sequence + "] (press left and right to navigate)",
                          fontsize=12)
            # ax1.set_yticks([])
            ax1.set_xticks([])

            x_old = 0
            for j in range(len(sequence_to_draw.sequence)):
                shapelet = sequence_to_draw.sequence[j]
                if j == 0:
                    x = list(range(len(shapelet.subsequence)))
                else:
                    x = list(range(x_old, x_old + len(shapelet.subsequence)))
                ax1.plot(x, shapelet.subsequence, c=color_dict[shapelet.dimension_name])
                if len(sequence_to_draw.time_steps) > j:
                    x1 = x[0]
                    y = -120
                    x2 = x[-1] + sequence_to_draw.time_steps[j]
                    ax1.annotate('', xy=(x1, y), xycoords='data', xytext=(x2, y), textcoords='data',
                                 arrowprops={'arrowstyle': '<->'})
                    ax1.annotate("maxlag:" + str(sequence_to_draw.time_steps[j]), xy=((x2 + x1) / 2, y + 2),
                                 xycoords='data',
                                 xytext=(-5, 5), textcoords='offset points')
                    x_old = x2

            lines = []
            labels = []
            ax2.cla()
            ax2.set_title("Covered Timeseries:" + timeseries_to_draw.name + " (press a and d to navigate)", fontsize=12)
            # ax2.set_yticks([])
            ax2.set_xticks([])
            for dimension in timeseries_to_draw.timeseries:
                line, = ax2.plot(timeseries_to_draw.timeseries[dimension], c=color_dict[dimension])
                lines.append(line)
                labels.append(dimension)
            old_occ = 0
            for j in range(len(sequence_to_draw.sequence)):
                shapelet = sequence_to_draw.sequence[j]
                occ = [index for index in shapelet.matching_indices[timeseries_to_draw.name] if index >= old_occ]
                x = list(range(occ[0], occ[0] + len(shapelet.subsequence)))
                ax2.plot(x, shapelet.subsequence, linewidth=5, c=color_dict[shapelet.dimension_name])

            plt.figlegend(lines, labels, loc='lower right')
            max_x = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
            ax1.set_xlim([-1, max_x + 10])
            ax2.set_xlim([-1, max_x + 10])

            max_y = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
            min_y = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
            ax1.set_ylim([min_y - 50, max_y + 100])
            ax2.set_ylim([min_y - 50, max_y + 100])
            fig.canvas.draw()

        def key_event(e):
            nonlocal sequence_index
            nonlocal matching_timeseries_index

            sequence_to_draw = list_all_sequence[sequence_index]
            if e.key == "right":
                sequence_index += 1
                if sequence_index == len(list_all_sequence):
                    sequence_index = 0
                matching_timeseries_index = 0
                plot_all_plots()
            elif e.key == "left":
                sequence_index -= 1
                if sequence_index < 0:
                    sequence_index = len(list_all_sequence) - 1
                matching_timeseries_index = 0
                plot_all_plots()
            elif e.key == "d":
                matching_timeseries_index += 1
                matching_timeseries_index %= len(sequence_to_draw.covered_instance_names)
                plot_all_plots()
            elif e.key == "a":
                matching_timeseries_index -= 1
                if matching_timeseries_index < 0:
                    matching_timeseries_index = len(sequence_to_draw.covered_instance_names) - 1
                plot_all_plots()
            else:
                return

        fig = plt.figure(2)
        plt.title("Rules Explorer", fontsize=24)
        fig.canvas.mpl_connect('key_press_event', key_event)
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        plot_all_plots()
        plt.show()

# now the real code :)
