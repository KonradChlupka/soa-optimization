import time
import struct

import numpy as np
import visa


class Lightwave7900B:
    def __init__(self, address=None):
        """Initializes object to communicate with Lightwave7900B.

        If no address is specified, __init__ will look through all the
        avaliable addresses to find the device, otherwise, it will check
        only the supplied address.

        Args:
            address (str): GPIB address of the device, e.g.
                "GPIB1::1::INSTR".
        """
        # open resource manager
        self.rm = visa.ResourceManager()

        if address:
            addresses = [address]
        else:
            addresses = self.rm.list_resources()

        print("Looking for ILX Lightwave 7900")
        self.inst = None
        for inst_str in addresses:
            print("Checking resource at {}".format(inst_str))
            try:
                inst = self.rm.open_resource(inst_str)
                query = inst.query("*IDN?")
                if "ILX Lightwave,7900 System" in query:
                    print("Found {}".format(query))
                    self.inst = inst
                    break
            except Exception:
                pass
        if self.inst is None:
            print("Couldn't find ILX Lightwave 7900B")
            self.inst = None

    def start_channels(self, channels=(1,)):
        """Selects and starts specified channels.

        Args:
            channels: Tuple of integers between 1 and 8.
        """
        assert isinstance(channels, tuple), "Argument must be a tuple"

        # iterate through all the channels
        for i in range(1, 9):
            # select channel
            self.inst.write("CH {}".format(i))
            # turn on/off selected channel
            self.inst.write("OUT {}".format(int(i in channels)))

    def set_channel_power(self, channel, power):
        """Sets power on a specified channel.

        Args:
            channel (int)
            power (float)
        """
        assert isinstance(channel, int), "Channel must be an int"
        assert isinstance(power, float), "Power must be a float"

        if power < -2.0 or power > 13.0:
            print("Warning: you might be using power outside supported range")

        # select channel
        self.inst.write("CH {}".format(channel))
        # turn on/off selected channel
        self.inst.write("LEVEL {}".format(power))

    def sweep_channel_power(self, channel, start, stop, step, seconds):
        """Turns on a channel and sweeps the power output.

        Args:
            channel (int)
            start (number): Starting point of sweep.
            stop (number): Sweep does not include this value.
            step (number): Size of step.
            seconds (number): Time between steps.
        """
        assert isinstance(channel, int), "Channel must be an int"

        if seconds < 2:
            print("Warning: the chosen delay between steps might be too low")

        # select channel
        self.inst.write("CH {}".format(channel))
        # turn on/off selected channel
        self.inst.write("OUT 1")

        for power in np.arange(start, stop, step):
            self.set_channel_power(channel, power)
            time.sleep(seconds)

    def set_channel_wavelength(self, channel, wavelength):
        """Sets wavelength on a specified channel.

        Args:
            channel (int)
            wavelength (float)
        """
        assert isinstance(channel, int), "Channel must be an int"
        assert isinstance(wavelength, float), "Wavelength must be a float"

        default_wavelengths = (
            1544.53,
            1545.32,
            1546.92,
            1547.72,
            1555.72,
            1558.98,
            1561.42,
            1562.23,
        )

        if (wavelength - default_wavelengths[channel - 1]) ** 2 > 9:
            print(
                "Warning: you might be using a wavelength outside supported range, "
                "default is {} and you're using {}".format(
                    default_wavelengths[channel - 1], wavelength
                )
            )

        # select channel
        self.inst.write("CH {}".format(channel))
        # turn on/off selected channel
        self.inst.write("WAVE {}".format(wavelength))

    def close(self):
        """Closes instrument and resource manager
        """
        self.inst.close()
        self.rm.close()


class Lightwave3220:
    def __init__(self, address=None, current_limit=None):
        """Initializes object to communicate with Lightwave3220.

        If no address is specified, __init__ will look through all the
        avaliable addresses to find the device, otherwise, it will check
        only the supplied address.

        Args:
            address (str): GPIB address of the device, e.g.
                "GPIB1::1::INSTR".
            current_limit (number): urrent limit in mA.
        """
        # open resource manager
        self.rm = visa.ResourceManager()
        self.current_limit = current_limit

        if address:
            addresses = [address]
        else:
            addresses = self.rm.list_resources()

        print("Looking for ILX Lightwave LDX-3220")
        self.inst = None
        for inst_str in addresses:
            print("Checking resource at {}".format(inst_str))
            try:
                inst = self.rm.open_resource(inst_str)
                query = inst.query("*IDN?")
                if "ILX Lightwave,3220" in query:
                    print("Found {}".format(query))
                    self.inst = inst
                    break
            except Exception:
                pass
        if self.inst is None:
            print("Couldn't find ILX Lightwave LDX-3220")
            self.inst = None

    def set_output(self, current, switch_output_on=True):
        """Sets the current output to specified value.

        Args:
            current (number)
            switch_output_on (bool): If True, it will turn on the
                output after specifying the current, otherwise it will
                stay in the initial state, whether on or off.
        """
        assert isinstance(current, int) or isinstance(
            current, float
        ), "Current must be a number"
        assert isinstance(switch_output_on, bool), "switch_output_on must be a bool"
        assert current >= 0, "Current must be non-negative"

        if self.current_limit:
            assert (
                current <= self.current_limit
            ), "Selected current exceeds current_limit, use can adjust it"

        # set output in mA
        self.inst.write("LAS:LDI {}".format(current))
        # switch current source on (stays on if already on)
        self.inst.write("LAS:OUT 1")

    def sweep_current(self, start, stop, step, seconds):
        """Turns on the current source and sweeps the output current.

        Args:
            start (number): Starting point of sweep in mA.
            stop (number): Sweep does not include this value in mA.
            step (number): Size of step in mA.
            seconds (number): Time between steps.
        """
        assert isinstance(seconds, int) or isinstance(
            seconds, float
        ), "Seconds must be a number"

        if seconds < 0.1:
            print("Warning: the chosen delay between steps might be too low")

        # switch current source on (stays on if already on)
        self.inst.write("LAS:OUT 1")

        # sweep the output
        for current in np.arange(start, stop, step):
            self.set_output(float(current), switch_output_on=False)
            time.sleep(seconds)

    def switch_off(self):
        """Switches the current source off.
        """
        self.inst.write("LAS:OUT 0")

    def close(self):
        """Closes instrument and resource manager.
        """
        self.inst.close()
        self.rm.close()


class AnritsuMS9740A:
    def __init__(self, address=None):
        """Initializes object to communicate with Anritsu MS9740A.

        If no address is specified, __init__ will look through all the
        avaliable addresses to find the device, otherwise, it will check
        only the supplied address.

        Args:
            address (str): GPIB address of the device, e.g.
                "GPIB1::1::INSTR".
        """
        # open resource manager
        self.rm = visa.ResourceManager()

        if address:
            addresses = [address]
        else:
            addresses = self.rm.list_resources()

        print("Looking for Anritsu MS9740A")
        self.inst = None
        for inst_str in addresses:
            print("Checking resource at {}".format(inst_str))
            try:
                inst = self.rm.open_resource(inst_str)
                query = inst.query("*IDN?")
                if "ANRITSU,MS9740A" in query:
                    print("Found {}".format(query))
                    self.inst = inst
                    break
            except Exception:
                pass
        if self.inst is None:
            print("Couldn't find Anritsu MS9740A")
            self.inst = None

    def set_x(self, center=None, span=None, start=None, stop=None):
        """Sets parameters (in nm) related to x axis.

        Any one of these can be used, and the later parameters will
        overwrite the previous parameters.

        args:
            center (number)
            span (number)
            start (number)
            stop (number)
        """
        if center:
            assert 600 <= center <= 1750, "Parameter outside supported range"
        if span:
            assert span > 0, "Parameter outside supported range"
        if start:
            assert 600 <= start <= 1750, "Parameter outside supported range"
        if stop:
            assert 600 <= stop <= 1750, "Parameter outside supported range"

        if center:
            self.inst.write("CNT {}".format(center))
        if span:
            self.inst.write("SPN {}".format(span))
        if start:
            self.inst.write("STA {}".format(start))
        if stop:
            self.inst.write("STO {}".format(stop))

    def set_y(self, db_per_div=None, ref=None):
        """Sets parameter related to y axis.

        args:
            db_per_div (number): Distance between divs, in dB. Must be
                between 0.1 and 10.
            ref (number): At the time of setting the Log scale, this
                command sets the reference level. Must be between -100
                and 100.
        """
        if db_per_div:
            assert 0.1 <= db_per_div <= 10, "Parameter outside supported range"
        if ref:
            assert -100 <= ref <= 100, "Parameter outside supported range"

        if db_per_div:
            self.inst.write("LOG {}".format(db_per_div))
        if ref:
            self.inst.write("RLV {}".format(ref))

    def set_resolution(self, resolution):
        """Sets resolution.

        args:
            resolution (float): Resolution in nm must be only one of the
                following: 0.03|0.05|0.07|0.1|0.2|0.5|1.0.
        """
        allowed_resolutions = (0.03, 0.05, 0.07, 0.1, 0.2, 0.5, 1.0)

        assert (
            resolution in allowed_resolutions
        ), "Resolution must be one of the following: {}".format(allowed_resolutions)

        self.inst.write("RES {}".format(resolution))

    def set_VBW(self, VBW):
        """Sets VBW (video band width).

        args:
            VBW (int or str): VBW in Hz, must be one of the following:
                10|100|200|1000|2000|10000|100000|1000000, or
                10HZ|100HZ|200HZ|1KHZ|2KHZ|10KHZ|100KHZ|1MHZ.
        """
        allowed_VBW = (
            10,
            100,
            200,
            1000,
            2000,
            10000,
            100000,
            1000000,
            "10HZ",
            "100HZ",
            "200HZ",
            "1KHZ",
            "2KHZ",
            "10KHZ",
            "100KHZ",
            "1MHZ",
        )
        assert isinstance(VBW, (int, str)), "VBW must be int or str"
        assert VBW in allowed_VBW, "VBW must be one of the following: {}".format(
            allowed_VBW
        )

        self.inst.write("VBW {}".format(VBW))

    def set_sampling_points(self, n):
        """Sets the number of sampling points.

        args:
            n (int): number of sampling points, must be one of the
                numbers: 51|101|251|501|1001|2001|5001|10001|20001|50001
        """
        allowed_n = (51, 101, 251, 501, 1001, 2001, 5001, 10001, 20001, 50001)
        assert isinstance(n, int), "Number of sampling points must be int"
        assert n in allowed_n, "VBW must be one of the following: {}".format(allowed_n)

        self.inst.write("MPT {}".format(n))

    def ana_rms(self, spectrum_level, spectrum_deviation_factor):
        """Executes the RMS spectrum analysis method.

        args:
            spectrum_level (number): In dB, between 0.1 and 50.0.
            spectrum_deviation_factor (number): K: Standard deviation
                factor, between 1.0 and 10.0.

        returns:
            List[float, float, float]:
                center wavelength (nm)
                spectrum width (nm)
                standard deviation
        """
        assert isinstance(
            spectrum_level, (int, float)
        ), "spectrum_level must be a nuber"
        assert isinstance(
            spectrum_deviation_factor, (int, float)
        ), "spectrum_deviation_factor must be a nuber"
        assert (
            0.1 <= spectrum_level <= 50.0
        ), "spectrum_level must be between 0.1 and 50.0"
        assert (
            1.0 <= spectrum_deviation_factor <= 10.0
        ), "spectrum_deviation_factor must be between 1.0 and 10.0"

        self.inst.write(
            "ANA RMS,{},{}".format(spectrum_level, spectrum_deviation_factor)
        )
        res = self.inst.query("ANAR?").split()
        if res == [-1, -1, -1]:
            print("Warning: RMS Analysis failed")
        return res

    def screen_capture(self):
        """Takes a single sweep of the screen content and returns.

        returns:
            List[float]: Each number is a sample at a wavelength,
                depending on set_x, and length depends on
                set_sampling_points.
        """
        self.inst.write("SSI; *WAI")
        res = self.inst.query("DMA?")
        return [float(i) for i in res.split()]

    def close(self):
        """Closes instrument and resource manager
        """
        self.inst.close()
        self.rm.close()


class TektronixAWG7122B:
    def __init__(self, address=None):
        """Initializes object to communicate with Tektronix AWG7122B.

        If no address is specified, __init__ will look through all the
        avaliable addresses to find the device, otherwise, it will check
        only the supplied address.

        Args:
            address (str): GPIB address of the device, e.g.
                "GPIB1::1::INSTR".
        """
        # open resource manager
        self.rm = visa.ResourceManager()

        if address:
            addresses = [address]
        else:
            addresses = self.rm.list_resources()

        print("Looking for Tektronix AWG7122B")
        self.inst = None
        for inst_str in addresses:
            print("Checking resource at {}".format(inst_str))
            try:
                inst = self.rm.open_resource(inst_str)
                query = inst.query("*IDN?")
                if "TEKTRONIX,AWG7122B" in query:
                    print("Found {}".format(query))
                    self.inst = inst
                    # set parameters necessary for binary data sending
                    self.inst.read_termination = None
                    self.inst.write_termination = "\r\n"
                    self.inst.encoding = "utf-8"
                    self.inst.timeout = 10000
                    break
            except Exception:
                pass
        if self.inst is None:
            print("Couldn't find Tektronix AWG7122B")
            self.inst = None

    def send_waveform(
        self,
        signal,
        markers=None,
        sampling_freq=12e9,
        amplitude=1.0,
        channel=1,
        name="konrad",
        suppress_messages=False,
    ):
        """Sends waveform to the device and turns on specified channel.

        args:
            signal (Any[float]): List of at least 1 float values, each
                value must be between -1 and 1, where the max values are
                equivalent to max set amplitude.
            markers (Any[int]): List of integers, either 0 or 1, which
                will determine if the markers are high or low. For now,
                this code supports setting both markers to the same
                value only. Must be the same length as signal.
            sampling_freq (int): Sampling frequency for the AWG.
                Combined with the length of signal, it determines the
                output signal frequency, i.e.
                output_frequency = sampling_frequency/len(signal)
                Must be between 10 MHz and 12 GHz.
            name (str): Name of the waveform.
            amplitude (number): Sets Vpp range of the signal. E.g. if
                signal is [0.0, 1.0] and amplitude is 0.5, the output
                signal will be [0.0 V, 0.25 V].
            suppress_messages (bool): If True, info and warning messages
                will not be shown.
        """
        assert all(
            isinstance(i, float) for i in signal
        ), "Signal must be a list of floats"
        assert all(-1 <= i <= 1 for i in signal), "Signal must be between -1 and 1"
        if len(signal) > 100000:
            print(
                "Warning: signal length is very large, "
                "consider using a lower sampling_freq instead of long signal"
            )

        # create marker if isn't supplied
        if not markers:
            markers = [1] + [0] * (len(signal) - 1)
        assert len(signal) == len(
            markers
        ), "Signal and markers must have the same length"
        assert all(isinstance(i, int) for i in markers), "Markers must be ints"
        assert all(i == 0 or i == 1 for i in markers), "Marker can be 1 or 0 only"

        assert isinstance(sampling_freq, (int, float)), "sampling_freq must be a number"
        assert (
            10e6 <= sampling_freq <= 12e9
        ), "sampling_freq must be between 10 MHz and 12 GHz"

        assert isinstance(amplitude, (int, float)), "amplitude must be a number"
        assert 0.5 <= amplitude <= 1.0, "amplitude must be between 0.5 and 1.0"

        assert channel in (1, 2), "channel must be 1 or 2"
        assert isinstance(channel, int), "channel must be an in"

        assert isinstance(name, str), "name must be a string"

        # substitute 1 with 192 to set both markers on (see AWG7220B docs)
        markers = [192 * i for i in markers]

        n_points = len(signal)

        # each point is 4 bytes for the float signal and 1 byte for the markers
        n_bytes = 5 * n_points

        # header as defined by the IEEE 488.2 standard
        header = "#" + str(len(str(n_bytes))) + str(n_bytes)

        # combine signal and marker as follows:
        # [signal[0], marker[0], signal[1], marker[1], ...]
        combined_array = [el for pair in zip(signal, markers) for el in pair]

        # create format to parse combined_array into little-endian
        # byte format ("<fBfBfB...")
        fmt = "<" + "fB" * n_points
        byte_data_block = bytes(header, "utf-8") + struct.pack(fmt, *combined_array)

        # waveform needs to be deleted first due to a bug on AWG
        # (occurs when new waveform is shorter)
        if not suppress_messages:
            print(
                "Warning: If waveform {} already exists, it wil be "
                "deleted now. If you want the same waveform on both "
                "channels, you have to create it with different names.".format(name)
            )
        self.inst.write('WLISt:WAVeform:DELete "{}"'.format(name))
        # if waveform doesn't exist, above would produce an error
        self.check_for_errors()

        self.inst.write('WLISt:WAVeform:NEW "{}", {}, REAL'.format(name, n_points))
        self.inst.write_raw(
            'WLISt:WAVeform:DATA "{}", '.format(name).encode("utf-8") + byte_data_block
        )
        # frequency is common for both channels
        self.inst.write("SOURce1:FREQuency {}".format(sampling_freq))
        self.inst.write("SOURce{}:VOLTage {}".format(channel, amplitude))
        self.inst.write('SOURce{}:WAVeform "{}"'.format(channel, name))
        self.inst.write("OUTPut{} ON".format(channel))
        self.inst.write("AWGControl:RUN")
        if not suppress_messages:
            print(
                "Sampling frequency (common for both channels) is "
                "{:.3e} and length of the signal is {:.3e}, "
                "so the output frequency is {:.3e}".format(
                    sampling_freq, n_points, sampling_freq / n_points
                )
            )

    def restore_defaults(self):
        """Sends *CLS and *RST to the device, restoring most defaults.

        The *CLS command clears the following:
        - error queue,
        - standard event status register (ESR),
        - status byte register (STB).
        *RST clears most parameters apart from the GPIB settings.
        """
        self.inst.write("WLISt:WAVeform:DELete ALL")
        self.inst.write("*CLS")
        self.inst.write("*RST")

    def check_for_errors(self):
        return self.inst.query("SYSTem:ERRor?")

    def close(self):
        """Closes instrument and resource manager.
        """
        self.inst.close()
        self.rm.close()


class Agilent8156A:
    def __init__(self, address=None):
        """Initializes object to communicate with Agilent 8156A.

        If no address is specified, __init__ will look through all the
        avaliable addresses to find the device, otherwise, it will check
        only the supplied address.

        Args:
            address (str): GPIB address of the device, e.g.
                "GPIB1::1::INSTR".
        """
        # open resource manager
        self.rm = visa.ResourceManager()

        if address:
            addresses = [address]
        else:
            addresses = self.rm.list_resources()

        print("Looking for Agilent 8156A")
        self.inst = None
        for inst_str in addresses:
            print("Checking resource at {}".format(inst_str))
            try:
                inst = self.rm.open_resource(inst_str)
                query = inst.query("*IDN?")
                if "HEWLETT-PACKARD,HP8156A" in query:
                    print("Found {}".format(query))
                    self.inst = inst
                    break
            except Exception:
                pass
        if self.inst is None:
            print("Couldn't find Agilent 8156A")
            self.inst = None

    def restore_defaults(self):
        """Sends *CLS and *RST to the device, restoring most defaults.

        The *CLS command clears the following:
        - error queue,
        - standard event status register (ESR),
        - status byte register (STB).
        *RST clears most parameters apart from the GPIB settings.
        """
        self.inst.write("*CLS")
        self.inst.write("*RST")

    def set_wavelength(self, wavelength):
        """Sets the wavelength for the instrument.

        The value is used to make the compensation for the wavelength
        dependence of the filter, using the wavelength calibration data.

        args:
            wavelength (number): Wavelength in nm (from 1200 to 1650).
        """
        assert isinstance(wavelength, (int, float)), "wavelength must be a number"
        assert 1200 <= wavelength <= 1650, "wavelength must be between 1200 and 1650"

        self.inst.write("INPut:WAVelength {}nm".format(wavelength))

    def set_calibration_factor(self, calib=2.9):
        """Sets the calibration factor for the instrument.

        args:
            calib (number): Calibration factor in dB.
        """
        assert isinstance(calib, (int, float)), "calib must be a number"

        self.inst.write("INPut:OFFSet {}".format(calib))

    def get_calibration_factor(self):
        """Return the calibration factor.

        returns:
            float: Calibration factor in dB.
        """

        return float(self.inst.query("INPut:OFFSet?").strip())

    def switch_output(self, state):
        """Opens or closes shutter.

        args:
            state (bool): True opens the shutter (laser goes through),
                while False closes it (no light passes through).
        """
        assert isinstance(state, bool), "state must be a bool"

        self.inst.write("OUTPut:STATe {}".format(int(state)))

    def set_output(self, attenuation):
        """Sets the output attenuation (without changing shutter).

        args:
            attenuation (number): Attenuation in dB, cannot be lower
                than calibration_factor.
        """
        assert isinstance(attenuation, (int, float)), "attenuation must be a number"
        assert (
            attenuation >= self.get_calibration_factor()
        ), "attenuation cannot be lower than calibration_factor"

        self.inst.write("INPut:ATTenuation {}".format(attenuation))

    def sweep(self, start, stop, step, seconds):
        """Sweeps the attenuation.

        args:
            start (number)
            stop (number)
            step (number): Change between steps.
            seconds (number): Delay between steps.
        """
        self.set_output(start)
        self.switch_output(True)

        for level in np.arange(start, stop, step):
            self.set_output(float(level))
            time.sleep(seconds)

    def close(self):
        """Closes instrument and resource manager.
        """
        self.inst.close()
        self.rm.close()


class Agilent86100C:
    def __init__(self, address=None):
        """Initializes object to communicate with Agilent 86100C.

        If no address is specified, __init__ will look through all the
        avaliable addresses to find the device, otherwise, it will check
        only the supplied address.

        Args:
            address (str): GPIB address of the device, e.g.
                "GPIB1::1::INSTR".
        """
        # open resource manager
        self.rm = visa.ResourceManager()

        if address:
            addresses = [address]
        else:
            addresses = self.rm.list_resources()

        print("Looking for Agilent 86100C")
        self.inst = None
        for inst_str in addresses:
            print("Checking resource at {}".format(inst_str))
            try:
                inst = self.rm.open_resource(inst_str)
                query = inst.query("*IDN?")
                if "Agilent Technologies,86100C" in query:
                    print("Found {}".format(query))
                    self.inst = inst
                    self.inst.timeout = 10000
                    self.inst.write("WAVeform:FORMat ASCii")
                    break
            except Exception:
                pass
        if self.inst is None:
            print("Couldn't find Agilent 86100C")
            self.inst = None

    def restore_defaults(self):
        """Sends *CLS and *RST to the device, restoring most defaults.

        The *CLS clears all status and error registers
        *RST clears most parameters apart from the GPIB settings.
        """
        self.inst.write("*CLS")
        self.inst.write("*RST")

    def autoscale(self):
        """Evaluates current input signal, autoscales, prints result.

        Autoscale sets the following:
        • Channel Display, Scale, and Offset
        • Trigger and Level
        • Time Base Scale and Position
        Autoscale turns off the following:
        • Measurements on sources that are turned off
        • Functions
        • Windows
        • Memories
        """
        self.inst.write("AUToscale")
        print(self.inst.query("AUToscale?"))

    def set_trigger(self, source="RMODule", level=10e-3):
        """Sets parameters related to trigger.

        args:
            source (str): One of FPANel, FRUN, LMODule, RMODule.
            level (number)
        """
        assert source in (
            "FPANel",
            "FRUN",
            "LMODule",
            "RMODule",
        ), "Source must be one of FPANel, FRUN, LMODule, RMODule"
        assert isinstance(level, (int, float)), "Level must be a number"

        self.inst.write("TRIGger:SOURce {}".format(source))
        self.inst.write("TRIGger:LEVel {}".format(level))

    def set_timebase(self, position=None, range_=None, reference=None):
        """Sets parameters related to timebase.

        args:
            position (number): in seconds - Sets the time interval
                between the trigger event and the delay reference point.
                The delay reference point is set with the
                TIMebase:REFerence command. The <position_value>
                argument’s maximum value depends on the
                time-per-division setting. Must be greater or equal
                2.4e-8.
            range_ (number): Sets the full-scale horizontal time in
                seconds. The range value is ten times the
                time-per-division value. Range is always set in units of
                time (seconds), not in bits. <full_scale_range> is the
                full-scale horizontal time in seconds.
            reference (str): Sets the delay reference to the left or
                center side of the display. Must be "left" or "center".
        """
        if position is not None:
            assert isinstance(position, (int, float)), "position must be a number"
            assert position >= 2.4e-8, "position must be greater or equal 2.4e-8 s"
            self.inst.write("TIMebase:POSition {}".format(position))

        if range_ is not None:
            assert isinstance(range_, (int, float)), "range_ must be a number"
            assert range_ >= 0, "range_ must be non-negative"
            self.inst.write("TIMebase:RANGe {}".format(range_))

        if reference is not None:
            assert reference.lower() in (
                "left",
                "center",
            ), "reference must be left or center (capitalization optional)"
            self.inst.write("TIMebase:REFerence {}".format(reference))

    def set_acquire(self, average=None, count=None, points=None):
        """Sets parameters related to acquiring data.

        args:
            average (bool): Enables or disables averaging. When ON, the
                analyzer acquires multiple data values for each time
                bucket, and averages them. When OFF, averaging is
                disabled. To set number of averages, use param count.
            count (int): Sets the number of averages for the waveforms.
                Specifies the number of data values to be averaged for
                each time bucket before the acquisition is considered
                complete for that time bucket. count must be from 1 to
                4096.
            points (int or str): Sets the requested memory depth for an
                acquisition. You can set the points value to AUTO, which
                allows the analyzer to select the number of points based
                upon the sample rate and time base scale. points range
                is 16 to 16,384 points.
        """
        if average is not None:
            assert isinstance(average, bool), "average must be a bool"
            self.inst.write("ACQuire:AVERage {}".format(int(average)))

        if count is not None:
            assert isinstance(count, int), "count must be an int"
            assert 1 <= count <= 4096, "count must be between 1 and 4096"
            self.inst.write("ACQuire:COUNt {}".format(count))

        if points is not None:
            assert isinstance(points, (int, str)), "points must be int or str"
            assert points == "AUTO" or (
                16 <= points <= 16384
            ), "points must be AUTO or int between 16 and 16384"
            self.inst.write("ACQuire:POINts {}".format(points))

    def set_channel(self, channel=4, display=None, offset=None, range_=None):
        """Sets parameters related to channels (and y-axis).

        args:
            channel (int): Channel to configure.
            display (bool): Turns the display of the specified channel
                on or off.
            offset (number): Sets the voltage that is represented at the
                center of the display for the selected channel.
            range_ (number): Defines the full-scale vertical axis of the
                selected channel in volts.
        """
        assert isinstance(channel, int), "Channel must be an int"
        assert channel in (1, 2, 3, 4), "Channel must be 1, 2, 3, or 4"

        if display is not None:
            assert isinstance(display, bool), "display must be bool"
            self.inst.write("CHANnel{}:DISPlay {}".format(channel, int(display)))

        if offset is not None:
            assert isinstance(offset, (int, float)), "offset must be a number"
            self.inst.write("CHANnel{}:OFFSet {}".format(channel, offset))

        if range_ is not None:
            assert isinstance(range_, (int, float)), "range_ must be a number"
            self.inst.write("CHANnel{}:RANGe {}".format(channel, range_))

    def measurement(self, channel=4):
        """Makes a measurement on a selected channel.

        args:
            channel (int): Channel to make measurement on.

        returns:
            List[float]: list of values in volts, where:
                “99.999E+36” represents a hole in the acquisition data,
                “99.999E+33” represents a clipped-high level,
                “99.999E+30” represents a clipped-low level.
        """
        assert isinstance(channel, int), "channel must be int"
        assert channel in (1, 2, 3, 4), "channel must be 1, 2, 3, or 4"
        self.inst.write("WAVeform:SOURce CHANnel{}".format(channel))

        return [float(i) for i in self.inst.query("WAVeform:DATA?").split(",")]

    def check_for_errors(self):
        return self.inst.query("SYSTem:ERRor?")

    def close(self):
        """Closes instrument and resource manager.
        """
        self.inst.close()
        self.rm.close()


if __name__ == "__main__":
    laser = Lightwave7900B("GPIB1::2::INSTR")
    current_source = Lightwave3220("GPIB1::12::INSTR")
    osa = AnritsuMS9740A("GPIB1::3::INSTR")
    awg = TektronixAWG7122B("GPIB1::1::INSTR")
    att = Agilent8156A("GPIB1::8::INSTR")
    osc = Agilent86100C("GPIB1::7::INSTR")
