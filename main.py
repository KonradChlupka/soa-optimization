import visa

# TODO: make a function for sweeping

class Lightwave:
    def __init__(self):
        # open resource manager
        self.rm = visa.ResourceManager()

        print("Looking for ILX Lightwave 7900")
        self.inst = None
        for inst_str in self.rm.list_resources():
            print("Checking resource at {}".format(inst_str))
            try:
                inst = self.rm.open_resource(inst_str)
                query = inst.query("*IDN?")
                if "ILX Lightwave,7900 System" in query:
                    print("Found {}".format(query))
                    self.inst = inst
                    break
            except Exception: # TODO: less generic exception
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
    
    # def sweep_cu
    
    def set_channel_wavelength(self, channel, wavelength):
        """Sets wavelength on a specified channel

        Args:
            channel (int)
            wavelength (float)
        """
        assert isinstance(channel, int), "Channel must be an int"
        assert isinstance(wavelength, float), "Wavelength must be a float"

        default_wavelengths = (1544.53,
                               1545.32,
                               1546.92,
                               1547.72,
                               1555.72,
                               1558.98,
                               1561.42,
                               1562.23)

        if (wavelength - default_wavelengths[channel - 1])**2 > 9:
            print("Warning: you might be using a wavelength outside supported range, default is {} and you're using {}".
                format(default_wavelengths[channel - 1], wavelength))

        # select channel
        self.inst.write("CH {}".format(channel))
        # turn on/off selected channel
        self.inst.write("WAVE {}".format(wavelength))

    def close(self):
        """Close resource manager
        """
        self.rm.close()

if __name__ == "__main__":
    x = Lightwave()
    x.start_channels((1,))
