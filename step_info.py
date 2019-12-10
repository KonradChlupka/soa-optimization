import numpy as np


class StepInfo:
    def __init__(
        self,
        y,
        t,
        ss_low,
        ss_high,
        rise_time_percentage=10,
        settling_time_percentage=5,
        inflection_point_percentage=50,
    ):
        """Performs a analysis of a step signal.

        Args:
            y (Iterable[float]): The step signal to be analyzed. Must
                contain only the step.
            t (Iterable[float]): Time.
            ss_low (float): Value of the output well before the rising
                edge.
            ss_high (float): Value of the output well after the rising
                edge.
            rise_time_percentage (number): Percentage value to be used
                in calculating rise time. Using "10" will result in
                calculating 10-90% rise time, using "5" will correspond
                to 5-95%, etc.
            settling_time_percentage (number): Range within which the
                signal must be contained to count as settled.
            inflection_point_percentage (number): Percentage of the
                rising edge used for inflection point when calculating
                mse.
        """
        self.y = y
        self.t = t
        self.ss_low = ss_low
        self.ss_high = ss_high
        self.rise_time_percentage = rise_time_percentage
        self.settling_time_percentage = settling_time_percentage
        self.inflection_point_percentage = inflection_point_percentage

        self.rise_time = self._rise_time()
        self.settling_time = self._settling_time()
        self.overshoot = self._overshoot()
        self.mse = self._mse()

    def _rise_time(self):
        """Calculates rise time, saves result to self.rise_time.

        Calculates rise time as specified by rise_time_percentage. If
        cannot be calculated, positive infinity is assigned.
        """
        amplitude = ss_high - ss_low
        low_inflection = ss_low + rise_time_percentage / 100 * amplitude
        high_inflection = ss_high - rise_time_percentage / 100 * amplitude

        for index, t in enumerate(self.t):
            if y[index] >= high_inflection:
                t_high_inflection = t
                break
        for index, t in enumerate(self.t):
            if y[index] >= low_inflection:
                t_low_inflection = t
                break

        try:
            self.rise_time = t_high_inflection - t_low_inflection
        except UnboundLocalError:
            self.rise_time = float("inf")

    def _settling_time(self):
        """Calculates settling time, saves result to self.settling_time.
        
        The time after which the rising edge behins is determined by
        rise_time_percentage. settling_time_percentage determines the
        range within which the signal must be contained to count as
        settled.
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

        # find time when signal settles
        for i, t in enumerate(self.t):
            if self.y[i] > settled_threshold_high or self.y[i] < settled_threshold_low:
                t_last_not_settled = t

        try:
            self.settling_time = t_last_not_settled - t_low_inflection
        except UnboundLocalError:
            self.settling_time = float("inf")

    def _overshoot(self):
        """Calculates % overshoot, saves result to self.overshoot.
        """
        self.overshoot = (
            100.0 * (max(self.y) - self.ss_high) / (self.ss_high - self.ss_low)
        )

    def _mse(self):
        """Calculates mean squared error with a pure square wave.

        Saves result to self.mse. The point where pure square wave rises
        is determined by inflection_point_percentage. Low and high
        levels of pure square wave are determined by ss_low and ss_high,
        respectively.
        """
        for index, t in enumerate(self.t):
            if y[index] >= ss_low + self.inflection_point_percentage / 100 * (
                self.ss_high - self.ss_low
            ):
                index_above_inflection = index
                break

        try:
            pure_square = [self.ss_low] * index_above_inflection + [self.ss_high] * (
                len(self.y) - index_above_inflection
            )
            self.mse = np.mean((np.array(pure_square - self.y) ** 2))
        except UnboundLocalError:
            self.mse = float("inf")