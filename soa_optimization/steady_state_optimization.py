import numpy as np
import time

import devices
from step_info import StepInfo


def steady_state_optimization():
    """Compares the effect of different steady-state on the rise time.
    """
    awg = devices.TektronixAWG7122B("GPIB1::1::INSTR")
    osc = devices.Agilent86100C("GPIB1::7::INSTR")

    # setup oscilloscope for measurement
    osc.set_acquire(average=True, count=30, points=1350)
    osc.set_timebase(position=4e-8, range_=12e-9)
    T = np.linspace(start=0, stop=12e-9, num=1350, endpoint=False)

    results = []

    for level in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        awg.send_waveform([-level] * 120 + [level] * 120, suppress_messages=True)
        time.sleep(5)
        res = osc.measurement(channel=1)
        rise_start = res[0]
        rise_end = res[-1]
        rise_time_pure = StepInfo(res, T, rise_start, rise_end).rise_time

        awg.send_waveform(
            [-level] * 110 + [-1.0] * 10 + [1.0] * 10 + [level] * 110,
            suppress_messages=True,
        )
        time.sleep(5)
        res = osc.measurement(channel=1)
        rise_time_optimized = StepInfo(res, T, rise_start, rise_end).rise_time

        results.append((level, rise_time_pure, rise_time_optimized))

    return results
