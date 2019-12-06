import numpy as np
import matplotlib.pyplot as plt


class ResponseMeasurements:
    def __init__(self, signal, t, gradient_points=8, percentage=5, hop_size=1):
        self.signal = signal
        self.t = t
        self.gradient_points = gradient_points
        self.percentage = percentage / 100
        self.hop_size = hop_size
        self.dt = abs(self.t[len(self.t) - 1] - self.t[len(self.t) - 2])
        self.__get_measurements()

    def __get_measurements(self):
        self.__get_inflection_time_index()
        self.__get_settling_max_index()
        self.__get_settling_time_index()
        self.__get_ss_high_value()
        self.__get_ss_low_value()
        self.__get_sp()
        self.__get_rise_time()
        self.__get_overshoot()

    def __get_sp(self):
        self.sp = SetPoint(self)

    def __get_inflection_time_index(self):
        grad = []
        n = len(self.signal)
        for i in range(n):
            if i + self.gradient_points < n:
                grad.append(abs(self.signal[i] - self.signal[i + self.gradient_points]))
            else:
                break

        self.inflection_time_index = np.argmax(grad)

    def __get_settling_max_index(self):
        self.settling_max_index = np.argmax(self.signal)

    def __get_settling_time_index(self):
        ss_high = self.signal[self.settling_max_index]
        n = len(self.signal)
        signal_end = self.signal[len(self.signal) - 10]
        for i in range(n):
            if (
                np.max(abs(self.signal[i:] - signal_end))
                <= self.percentage * signal_end
            ):
                self.settling_time_index = i
                self.settling_time = (i - self.inflection_time_index) * abs(
                    self.t[0] - self.t[1]
                )
                break

            else:
                self.settling_time_index = len(self.signal) - 1
                self.settling_time = 1000  # signal never settles

    def __get_ss_high_value(self):
        self.ss_high_value = np.mean(
            self.signal[self.settling_time_index : len(self.signal) - 10]
        )

    def __get_ss_low_value(self):

        self.ss_low_value = np.mean(self.signal[: self.inflection_time_index])

    def __get_rise_time(self):
        off_set = self.sp.sp[0]  # get amount signal is offset from 0 by
        i = self.inflection_time_index
        prev_diff = abs(
            (self.signal.copy()[i] - off_set) - (((self.ss_high_value - off_set) * 0.1))
        )
        curr_diff = abs(
            (self.signal.copy()[i + 1] - off_set)
            - (((self.ss_high_value - off_set) * 0.1))
        )
        while curr_diff < prev_diff:
            prev_diff = abs(
                (self.signal.copy()[i] - off_set)
                - (((self.ss_high_value - off_set) * 0.1))
            )
            curr_diff = abs(
                (self.signal.copy()[i + 1] - off_set)
                - (((self.ss_high_value - off_set) * 0.1))
            )
            i += 1
        self.idx_ten = i - 1

        i = self.inflection_time_index
        prev_diff = abs(
            (self.signal.copy()[i] - off_set) - (((self.ss_high_value - off_set) * 0.9))
        )
        curr_diff = abs(
            (self.signal.copy()[i + 1] - off_set)
            - (((self.ss_high_value - off_set) * 0.9))
        )
        while curr_diff < prev_diff:
            prev_diff = abs(
                (self.signal.copy()[i] - off_set)
                - (((self.ss_high_value - off_set) * 0.9))
            )
            curr_diff = abs(
                (self.signal.copy()[i + 1] - off_set)
                - (((self.ss_high_value - off_set) * 0.9))
            )
            i += 1
        self.idx_ninety = i - 1
        time_ten = self.t[self.idx_ten]
        time_ninety = self.t[self.idx_ninety]
        self.rise_time = time_ninety - time_ten

    def __get_overshoot(self):
        self.overshoot = abs(
            float(
                (self.signal[self.settling_max_index] - self.ss_high_value)
                / self.ss_high_value
            )
        )


class SetPoint:
    def __init__(self, response):
        self.response = response
        self.__getSetPoint()

    def __getSetPoint(self):
        self.sp = np.zeros(len(self.response.signal))
        self.inflection_time_index = self.response.inflection_time_index
        self.sp[: self.inflection_time_index] = self.response.ss_low_value
        self.sp[self.inflection_time_index :] = self.response.ss_high_value
