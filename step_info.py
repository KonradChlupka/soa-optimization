class StepInfo:
    def __init__(
        self,
        y,
        t,
        ss_low,
        ss_high,
        rise_time_percentage=10,
        settling_time_percentage=5,
        inflection_point_percentage=10,
    ):
        """Performs a full step analysis of a signal.

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
                rising edge used for inflection point.
        """
        self.y = y
        self.t = t
        self.ss_low = ss_low
        self.ss_high = ss_high
        self.rise_time_percentage = rise_time_percentage
        self.settling_time_percentage = settling_time_percentage
        self.inflection_point_percentage = inflection_point_percentage

        self.rise_time = self.rise_time()
        self.settling_time = self.settling_time()
        self.overshoot = self.overshoot()
        self.mse = self.mse()

    def rise_time():
        pass

    def settling_time():
        pass

    def overshoot():
        pass

    def mse():
        pass
