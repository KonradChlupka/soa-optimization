import numpy as np


class StepInfo:
    def __init__(
        self,
        y,
        t,
        ss_low=None,
        ss_high=None,
        n_ss=10,
        rise_time_percentage=10,
        settling_time_percentage=5,
        inflection_point_percentage=10,
        step_length=None,
    ):
        """Performs the analysis of a step signal.

        Args:
            y (Iterable[float]): The step signal to be analyzed. Must
                contain only the step. Optinally can contain signal
                after the step, indicated by step_length.
            t (Iterable[float]): Time.
            ss_low (float): Value of the output well before the rising
                edge. If None, it will be calculated as the average of
                the first n_ss points of y.
            ss_high (float): Value of the output well after the rising
                edge. I None, it will be calculated as the average of
                the last n_ss points of y (optionally clipped if
                step_length is provided).
            rise_time_percentage (number): Percentage value to be used
                in calculating rise time. Using "10" will result in
                calculating 10-90% rise time, using "5" will correspond
                to 5-95%, etc.
            settling_time_percentage (number): Range within which the
                signal must be contained to count as settled.
            inflection_point_percentage (number): Percentage of the
                rising edge used for inflection point when calculating
                mse.
            step_length (int): If y contains more than just the step,
                this indicates how many samples from the start contain
                the step. Setting this parameter creates two properties,
                y_unclipped and t_unclipped.
        """
        self.y = y
        self.t = t
        self.rise_time_percentage = rise_time_percentage
        self.settling_time_percentage = settling_time_percentage
        self.inflection_point_percentage = inflection_point_percentage
        self.inflection_point_index = None  # calculated inside _mse()
        self.step_length = step_length

        if step_length:
            self.y_unclipped = self.y
            self.t_unclipped = self.t
            self.y = self.y[:step_length]
            self.t = self.t[:step_length]

        if ss_low:
            self.ss_low = ss_low
        else:
            self.ss_low = float(np.mean(self.y[:n_ss]))
        if ss_high:
            self.ss_high = ss_high
        else:
            self.ss_high = float(np.mean(self.y[-n_ss:]))

        self.rise_time = self._rise_time()
        self.settling_time = self._settling_time()
        self.overshoot = self._overshoot()
        self.mse = self._mse()

    def _rise_time(self):
        """Calculates rise time.

        Calculates rise time as specified by rise_time_percentage. If
        cannot be calculated, positive infinity is assigned.

        Returns:
            float
        """
        amplitude = self.ss_high - self.ss_low
        low_inflection = self.ss_low + self.rise_time_percentage / 100 * amplitude
        high_inflection = self.ss_high - self.rise_time_percentage / 100 * amplitude

        for index, t in enumerate(self.t):
            if self.y[index] >= high_inflection:
                t_high_inflection = t
                break
        for index, t in enumerate(self.t):
            if self.y[index] >= low_inflection:
                t_low_inflection = t
                break

        try:
            return t_high_inflection - t_low_inflection
        except UnboundLocalError:
            return float("inf")

    def _settling_time(self):
        """Calculates settling time.

        The time after which the rising edge behins is determined by
        rise_time_percentage. settling_time_percentage determines the
        range within which the signal must be contained to count as
        settled.

        Returns:
            float
        """
        amplitude = self.ss_high - self.ss_low
        settled_threshold_high = (
            self.ss_high + self.settling_time_percentage / 100 * amplitude
        )
        settled_threshold_low = (
            self.ss_high - self.settling_time_percentage / 100 * amplitude
        )

        # find inflection point
        for i, t in enumerate(self.t):
            if self.y[i] >= self.ss_low + self.rise_time_percentage / 100 * (
                self.ss_high - self.ss_low
            ):
                t_low_inflection = t
                break

        # find time when signal settles
        for i, t in enumerate(self.t):
            if self.y[i] > settled_threshold_high or self.y[i] < settled_threshold_low:
                t_last_not_settled = t

        try:
            return t_last_not_settled - t_low_inflection
        except UnboundLocalError:
            return float("inf")

    def _overshoot(self):
        """Calculates % overshoot.

        Returns:
            float
        """
        return 100.0 * (max(self.y) - self.ss_high) / (self.ss_high - self.ss_low)

    def _mse(self):
        """Calculates mean squared error with a pure square wave.

        The point where pure square wave rises is determined by
        inflection_point_percentage. Low and high levels of pure square
        wave are determined by ss_low and ss_high, respectively.

        Returns:
            float
        """
        for index, t in enumerate(self.t):
            if self.y[index] >= self.ss_low + self.inflection_point_percentage / 100 * (
                self.ss_high - self.ss_low
            ):
                index_above_inflection = index
                self.inflection_point_index = index
                break

        try:
            pure_square = [self.ss_low] * index_above_inflection + [self.ss_high] * (
                len(self.y) - index_above_inflection
            )
            return np.mean(((np.array(pure_square) - np.array(self.y)) ** 2))
        except UnboundLocalError:
            return float("inf")
