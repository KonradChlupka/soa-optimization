import devivces
import time

results = []
osc = devices.Agilent86100C("GPIB1::7::INSTR")

for n_of_averages in range(1, 101):
    osc.set_acquire(count=n_of_averages)
    dict_resp = {}
    dict_resp["n_of_averages"] = n_of_averages
    dict_resp["outputs"] = []
    for i in range(100):
        time.sleep(0.5 + n_of_averages / 10)
        dict_resp["outputs"].append(osc.measurement(channel=1))
    results.append(dict_resp)
