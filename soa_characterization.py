import time
import pickle
import csv
import random

import devices


class Experiment:
    def __init__(self):
        pass

    def waveform_delay(self, original, delayed):
        """Calculates index delay between signals.

        Requires that 'delayed' is the same signal as 'original' (or
        slightly changed due to ringing etc.), 'delayed' is inverted,
        falling edge of both signals is visible, at least one full
        period of each is visible, a centered square wave from -1 to 1
        is sent, and 'original' on the left side of the screen is high
        and close to the falling edge.

        Args:
            original (List or np.array)
            delayed (List or np.array)

        Returns:
            int: Index offset between delayed and original.
        """
        input(
            "Make sure the signals are positioned correctly for "
            "measurement. Refer to documentation of waveform_delay(). "
            "Press Enter..."
        )

        on_top = False
        for idx, el in enumerate(original):
            if el > 0:
                on_top = True
            if el < 0 and on_top is True:
                orig_crossover_idx = idx
                break

        on_top = False
        delayed = -1 * np.array(delayed)
        for idx, el in enumerate(delayed):
            if el > 0:
                on_top = True
            if el < 0 and on_top is True:
                delayed_crossover_idx = idx
                break

        if orig_crossover_idx > delayed_crossover_idx:
            raise ValueError(
                "Delayed signal seems to be before original, check if "
                "the signal is compliant with requirements in "
                "waveform_delay"
            )
        delay = delayed_crossover_idx - orig_crossover_idx
        print("Detected delay of {} points.".format(delay))
        return delay

    def save_to_pickle(self, name):
        """Saves self.results to a pickle file.

        Args:
            name (str): name to which the results should be saved,
                without extension.
        """
        p = open(name + ".pkl", "wb")
        pickle.dump(self.results, p)
        p.close()

    def save_to_csv(self, name):
        """Saves self.results to a csv file.

        Args:
            name (str): name to which the results should be saved,
                without extension.
        """
        c = open(name + ".csv", "w", newline="")
        w = csv.writer(c)
        if isinstance(self.results, dict):
            for key, val in self.results.items():
                w.writerow([*key, *val])
        if isinstance(self.results, list):
            for line in self.results:
                w.writerow([*line])
        c.close()


class Experiment_1(Experiment):
    def __init__(self):
        """Initializes the devices needed in the experiment.

        Initializes current source, optical spectrum analyzer, and
        the attenuator. Uses addresses from UCL CONNET lab. Edit source
        code of this class to change addresses.
        """
        self.current_source = devices.Lightwave3220("GPIB1::12::INSTR")
        self.osa = devices.AnritsuMS9740A("GPIB1::3::INSTR")
        self.att = devices.Agilent8156A("GPIB1::8::INSTR")

    def run(self, name):
        """Runs the experiment, saves the results.

        The results are saved to csv, pickle, and self.results in format
        Dict[tuple[int, int]: List[float]]:
        {(current, attenuation): list_of_results}. Sweeps current and
        attenuation, measures the output of SOA on the OSA. Returns the
        results, as well as save as pickle and csv.

        Args:
            name (str): name to which the results should be saved,
                without extension.
        """
        current_values = range(0, 151, 5)
        attenuation_values = range(0, 51, 2)

        # results: {(current, attenuation): list_of_results}
        self.results = {}

        # turn on both devices
        self.current_source.set_output(0, switch_output_on=True)
        self.att.switch_output(True)
        print("Starting experiment in 3 seconds...")
        time.sleep(3)

        for current in current_values:
            self.current_source.set_output(current)
            for attenuation in attenuation_values:
                self.att.set_output(attenuation)
                time.sleep(1)
                print(
                    "Measuring for {:3} mA {:2} dB attenuation".format(
                        current, attenuation
                    )
                )
                self.results[(current, attenuation)] = self.osa.screen_capture()

        self.current_source.switch_off()
        self.att.switch_output(False)

        super().save_to_pickle(name)
        super().save_to_csv(name)


class Experiment_2(Experiment):
    def __init__(self):
        """Initializes the devices needed in the experiment.

        Initializes arbitrary function generator, and oscilloscope. Uses
        addresses from UCL CONNET lab. Edit source code of this class to
        change addresses.
        """
        self.awg = devices.TektronixAWG7122B("GPIB1::1::INSTR")
        self.osc = devices.Agilent86100C("GPIB1::7::INSTR")

    def run(self, name):
        """Runs the experiment, saves the results.

        Sends a squarewave made of 240 points, and a MISIC-like signal.
        The results are saved to csv, pickle, and self.results as a
        list. Sweeps the signal amplitude.

        Args:
            name (str): name to which the results should be saved,
                without extension.
        """
        self.results = [
            [
                "signal_type",
                "amplitude_multiplier",
                "mean_squared_error",
                "",
                "direct_signal",
                "...",
                "amplifier_signal",
            ],
            [""],
        ]

        # create MISIC and square signals
        random.seed(0)
        misic = [random.randint(-2, 2) for i in range(60)]
        misic = np.array([el / 2 for el in misic for _ in range(4)])
        square = np.array([-1.0] * 120 + [1.0] * 120)

        # setup oscilloscope for measurement
        self.osc.set_acquire(points=1351)
        self.osc.set_timebase(position=2.4e-8, range_=30e-9)
        # time_step = 30e-9 / 1351

        # get delay between signals
        self.awg.send_waveform(square, suppress_messages=True)
        time.sleep(2.5)
        idx_delay = super().waveform_delay(
            self.osc.measurement(4), self.osc.measurement(2)
        )

        amplitude_multipliers = np.arange(0.05, 1.01, 0.05)

        # loop through square signals of different amplitudes
        for mult in amplitude_multipliers:
            print("Measuring for square wave with multiplier {:.3}".format(mult))
            self.awg.send_waveform(mult * square, suppress_messages=True)
            time.sleep(2.5)
            orig = self.osc.measurement(4)
            delayed = self.osc.measurement(2)

            # align both signals (delayed is also flipped back)
            del orig[-idx_delay:]
            delayed = delayed[idx_delay:]
            orig = np.array(orig)
            delayed = -1 * np.array(delayed)

            # normalize both signal
            rms_orig = np.sqrt(np.mean(orig ** 2))
            rms_delayed = np.sqrt(np.mean(delayed ** 2))
            orig_norm = orig / rms_orig
            delayed_norm = delayed / rms_delayed

            mean_square_error = np.mean((orig_norm - delayed_norm) ** 2)

            result = ["square", mult, mean_square_error, "", *orig, "", *delayed]
            self.results.append(result)
        self.results.append([""])

        # loop through MISIC signals of different amplitudes
        for mult in amplitude_multipliers:
            print("Measuring for misic with multiplier {:.3}".format(mult))
            self.awg.send_waveform(mult * misic, suppress_messages=True)
            time.sleep(2.5)
            orig = self.osc.measurement(4)
            delayed = self.osc.measurement(2)

            # align both signals (delayed is also flipped back)
            del orig[-idx_delay:]
            delayed = delayed[idx_delay:]
            orig = np.array(orig)
            delayed = -1 * np.array(delayed)

            # normalize both signal
            rms_orig = np.sqrt(np.mean(orig ** 2))
            rms_delayed = np.sqrt(np.mean(delayed ** 2))
            orig_norm = orig / rms_orig
            delayed_norm = delayed / rms_delayed

            mean_square_error = np.mean((orig_norm - delayed_norm) ** 2)

            result = ["misic", mult, mean_square_error, "", *orig, "", *delayed]
            self.results.append(result)

        self.save_to_csv(name)
        self.save_to_pickle(name)


class Experiment_3(Experiment):
    def __init__(self):
        """Initializes the devices needed in the experiment.

        Initializes arbitrary function generator, oscilloscope, osa, and
        current source. Uses addresses from UCL CONNET lab. Edit source
        code of this class to change addresses.
        """
        self.awg = devices.TektronixAWG7122B("GPIB1::1::INSTR")
        self.osc = devices.Agilent86100C("GPIB1::7::INSTR")
        self.current_source = devices.Lightwave3220("GPIB1::12::INSTR", current_limit=100)
        self.att = devices.Agilent8156A("GPIB1::8::INSTR")
        self.osa = devices.AnritsuMS9740A("GPIB1::3::INSTR")

    def run(self, name):
        """Runs the experiment, saves the results.

        Sends a squarewave made of 240 points, and a MISIC-like signal.
        The results are saved to csv, pickle, and self.results as a
        list. Sweeps the bias current and attenuation.

        Args:
            name (str): name to which the results should be saved,
                without extension.
        """
        self.results = [
            [
                "Signal type",
                "Bias current [mA]",
                "Attenuation [dB]",
                "mean_squared_error",
                "SMSR [dB]",
                "Peak wavelength [nm]",
                "Peak level [dBm]",
                "SNR (/1nm) [dB]",
                "SNR (Res 0.051 nm) [dB]",
                "",
                "direct_signal",
                "...",
                "amplifier_signal",
            ],
            [""],
        ]

        # create MISIC and square signals
        random.seed(0)
        signal_names = ["square", "MISIC"]
        square = np.array([-1.0] * 120 + [1.0] * 120)
        misic_a = [-1.0] * 50
        misic_b = np.array([random.randint(-2, 2) for _ in range(50)]) / 2
        misic_c = np.array([random.randint(0, 2) for _ in range(140)]) / 2
        misic = [*misic_a, *misic_b, *misic_c]

        # setup oscilloscope for measurement
        self.osc.set_acquire(average=True, count=100, points=1351)
        self.osc.set_timebase(position=2.4e-8, range_=30e-9)
        # time_step = 30e-9 / 1351

        # get delay between signals
        self.current_source.set_output(75)
        self.awg.send_waveform(square, suppress_messages=True)
        time.sleep(8)
        orig = self.osc.measurement(4)
        delayed = np.array(self.osc.measurement(1))
        delayed = delayed - np.mean(delayed)
        idx_delay = super().waveform_delay(orig, delayed)

        # turn off averaging on the oscilloscope
        self.osc.set_acquire(average=False)

        # setup OSA measurement (laser diode)
        self.osa.inst.write("AP LD")

        bias_currents = range(55, 96, 5)
        attenuation_values = range(0, 17, 2)

        for (signal_name, signal_type) in zip(signal_names, (square, misic)):
            self.awg.send_waveform(signal_type, suppress_messages=True)
            time.sleep(2)

            for current in bias_currents:
                self.current_source.set_output(current)

                for attenuation in attenuation_values:
                    self.att.set_output(attenuation)

                    print(
                        "Measuring for {}, wave with bias current {}, and attenuation "
                        "of {}".format(signal_name, current, attenuation)
                    )
                    time.sleep(1.5)

                    osa_measurements = self.osa.inst.query("APR?").strip().split(",")

                    # oscilloscope measurements
                    orig = self.osc.measurement(4)
                    delayed = np.array(self.osc.measurement(1))
                    delayed = delayed - np.mean(delayed)

                    # align both signals (delayed is also flipped back)
                    del orig[-idx_delay:]
                    delayed = delayed[idx_delay:]
                    orig = np.array(orig)
                    delayed = -1 * np.array(delayed)

                    # normalize both signal
                    rms_orig = np.sqrt(np.mean(orig ** 2))
                    rms_delayed = np.sqrt(np.mean(delayed ** 2))
                    orig_norm = orig / rms_orig
                    delayed_norm = delayed / rms_delayed

                    mean_squared_error = np.mean((orig_norm - delayed_norm) ** 2)

                    result = [
                        "{}".format(signal_name),
                        current,
                        attenuation,
                        mean_squared_error,
                        osa_measurements[5],  # SMSR
                        osa_measurements[6],  # Peak wavelength
                        osa_measurements[7],  # Peak level
                        osa_measurements[8],  # SNR (/1nm)
                        osa_measurements[9],  # SNR (Res 0.051 nm)
                        "",
                        *orig,
                        "",
                        *delayed,
                    ]
                    self.results.append(result)
                self.results.append([""])
            self.results.append([""])

        self.save_to_csv(name)
        self.save_to_pickle(name)


if __name__ == "__main__":
    pass