import time
import numpy as np

import devices
from step_info import StepInfo


def rising_edge_optimization():
    """Finds optimal rising edge.

    The function tests out different rising edges, as follows:
    each full wave is 240 points, first 80 are -0.75, last 80 are 0.75,
    and the points in between are either 0.75 or 1.0 (negative before
    the rising edge). Then rise times can be compared.
    """
    awg = devices.TektronixAWG7122B("GPIB1::1::INSTR")
    osc = devices.Agilent86100C("GPIB1::7::INSTR")

    # setup oscilloscope for measurement
    osc.set_acquire(average=True, count=30, points=1350)
    osc.set_timebase(position=4e-8, range_=12e-9)
    T = np.linspace(start=0, stop=12e-9, num=1350, endpoint=False)

    # find rise-time ref values
    awg.send_waveform([-0.75] * 120 + [0.75] * 120, suppress_messages=True)
    time.sleep(5)
    res = osc.measurement(channel=1)
    ss_low = res[0]
    ss_high = res[-1]

    results = []

    for n_high in range(9):
        for n_low in range(9):
            rising_edge = ([-0.75] * 5 * (8 - n_low) + [-1.0] * 5 * n_low) + (
                [1.0] * 5 * n_high + [0.75] * 5 * (8 - n_high)
            )
            awg.send_waveform(
                [-0.75] * 80 + rising_edge + [0.75] * 80, suppress_messages=True
            )
            time.sleep(5)
            result = osc.measurement(channel=1)
            my_rise_time = StepInfo(result, T, ss_low, ss_high).rise_time
            print(my_rise_time)
            results.append((rising_edge, result, my_rise_time))

    return results
