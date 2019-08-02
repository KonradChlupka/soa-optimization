import time

import visa
import numpy as np


class Lightwave7900B:
    def __init__(self, address=None):
        """Initializes object to communicate with Lightwave7900B

        If no address is specified, __init__ will look through all the
        avaliable addresses to find the device, otherwise, it will check
        only the supplied address.

        Args:
            address (str): GPIB address of the device, e.g.
                "GPIB1::1::INSTR"
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
        """Selects and starts specified channels

        Args:
            channels: tuple of integers between 1 and 8
        """
        assert isinstance(channels, tuple), "Argument must be a tuple"

        # iterate through all the channels
        for i in range(1, 9):
            # select channel
            self.inst.write("CH {}".format(i))
            # turn on/off selected channel
            self.inst.write("OUT {}".format(int(i in channels)))
    
    def set_channel_power(self, channel, power):
        """Sets power on a specified channel

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
        """Turns on a channel and sweeps the power output

        Args:
            channel (int)
            start (number): starting point of sweep
            stop (number): sweep does not include this value
            step (number): size of step
            seconds (number): time between steps
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
        """Sets wavelength on a specified channel

        Args:
            channel (int)
            wavelength (float)
        """
        assert isinstance(channel, int), "Channel must be an int"
        assert isinstance(wavelength, float), "Wavelength must be a float"

        if (wavelength - default_wavelengths[channel - 1])**2 > 9:
            print("Warning: you might be using a wavelength outside supported range, default is {} and you're using {}".
                format(default_wavelengths[channel - 1], wavelength))

        default_wavelengths = (1544.53,
                               1545.32,
                               1546.92,
                               1547.72,
                               1555.72,
                               1558.98,
                               1561.42,
                               1562.23)

        # select channel
        self.inst.write("CH {}".format(channel))
        # turn on/off selected channel
        self.inst.write("WAVE {}".format(wavelength))

    def close(self):
        """Close resource manager
        """
        self.rm.close()

class Lightwave3220:
    def __init__(self, address=None):
        """Initializes object to communicate with Lightwave3220

        If no address is specified, __init__ will look through all the
        avaliable addresses to find the device, otherwise, it will check
        only the supplied address.

        Args:
            address (str): GPIB address of the device, e.g.
                "GPIB1::1::INSTR"
        """
        # open resource manager
        self.rm = visa.ResourceManager()

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
        """Sets the current output to specified value

        Args:
            current (number)
            switch_output_on (bool): if True, it will turn on the
                output after specifying the current, otherwise it will
                stay in the initial state, whether on or off
        """
        assert isinstance(current, int) or isinstance(current, float), "Current must be a number"
        assert isinstance(switch_output_on, bool), "switch_output_on must be a bool"
        assert current >= 0, "Current must be non-negative"

        # set output in mA
        self.inst.write("LAS:LDI {}".format(current))
        # switch current source on (stays on if already on)
        self.inst.write("LAS:OUT 1")
    
    def sweep_current(self, start, stop, step, seconds):
        """Turns on the current source and sweeps the output current

        Args:
            start (number): starting point of sweep in mA
            stop (number): sweep does not include this value in mA
            step (number): size of step in mA
            seconds (number): time between steps
        """
        assert isinstance(seconds, int) or isinstance(seconds, float), "Seconds must be a number"
        
        if seconds < 0.1:
            print("Warning: the chosen delay between steps might be too low")
        
        # switch current source on (stays on if already on)
        self.inst.write("LAS:OUT 1")

        # sweep the output
        for current in np.arange(start, stop, step):
            self.set_output(float(current), switch_output_on=False)
            time.sleep(seconds)

    def switch_off(self):
        """Switches the current source off
        """
        self.inst.write("LAS:OUT 0")

    def close(self):
        """Close resource manager
        """
        self.rm.close()
    
class AnritsuMS9740A:
    def __init__(self, address=None):
        """Initializes object to communicate with Anritsu MS9740A

        If no address is specified, __init__ will look through all the
        avaliable addresses to find the device, otherwise, it will check
        only the supplied address.

        Args:
            address (str): GPIB address of the device, e.g.
                "GPIB1::1::INSTR"
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
        """Sets parameters (in nm) related to x axis

        Any one of these can be used, and the later parameters will
        overwrite the previous parameters.

        args:
            center (number)
            span (number)
            start (number)
            stop (number)
        """
        assert center >= 600 and center <= 1750, "Parameter outside supported range"
        assert span >= 600 and span <= 1750, "Parameter outside supported range"
        assert start >= 600 and start <= 1750, "Parameter outside supported range"
        assert stop >= 600 and stop <= 1750, "Parameter outside supported range"

        if center:
            self.inst.write("CNT {}".format(center))
        if span:
            self.inst.write("SPN {}".format(span))
        if start:
            self.inst.write("STA {}".format(start))
        if stop:
            self.inst.write("STO {}".format(stop))


    
    def set_y(self, db_per_div=None, ref=None):
        """Sets parameter related to y axis

        args:
            db_per_div (number): distance between divs, in dB. Must be
                between 0.1 and 10
            ref (number): at the time of setting the Log scale, this command sets and queries the reference level 
        """
        assert db_per_div >= 0.1 and db_per_div <= 10,
            "Parameter outside supported range"
        assert ref >= 0.1 and ref <= 10, "Parameter outside supported range"

        if db_per_div:
            self.inst.write("LOG {}".format(db_per_div))
    
    def set_resolution(self, res):
        pass
    
    def set_VBW(self, VBW):
        pass
    
    def set_sampling_points(self, n):
        pass

    def close(self):
        """Close resource manager
        """
        self.rm.close()


if __name__ == "__main__":
    # laser = Lightwave7900B("GPIB1::2::INSTR")
    # current_source = Lightwave3220("GPIB1::12::INSTR")
    osa = AnritsuMS9740A("GPIB1::3::INSTR")