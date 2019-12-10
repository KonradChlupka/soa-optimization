import numpy as np
import matplotlib.pyplot as plt
from scipy import arange

#Get FOPDT parameters quickly from input (driving signal), output and a time axis.
#
#USE EXAMPLE:
#fopdtParams = FopdtMeasurements(signal,drive,t)
#-->
#Kc = fopdtParams.Kc
#TauP = fopdtParams.TauP
#ThetaP = fopdtParams.ThetaP

class ResponseMeasurements:

    def __init__(self,signal,t,gradientPoints=20,percentage=5):

        
        self.signal = signal
        self.t = t
        self.gradientPoints = gradientPoints
        self.percentage = percentage/100

        self.dt = abs(self.t[len(self.t)-1] - self.t[len(self.t)-2])

        self.__getMeasurements()

    def __getMeasurements(self):

        self.__getInflectionTimeIndex()
        self.__getSettlingMaxIndex()
        self.__getSettlingTimeIndex()
        self.__getSSHighValue()
        self.__getSSLowValue()
        self.__getRiseTime()
        self.__getOvershoot()

    def __getInflectionTimeIndex(self):

        grad = []
        n = len(self.signal)
        for i in range(n):
            if i + self.gradientPoints < n:
                grad.append(abs(self.signal[i] - self.signal[i+self.gradientPoints]))
            else:
                break
        self.inflectionTimeIndex = np.argmax(grad)

    def __getSettlingMaxIndex(self):
        self.settlingMaxIndex = np.argmax(self.signal)

    def __getSettlingTimeIndex(self):
        ss_high = self.signal[self.settlingMaxIndex] 
        n = len(self.signal)
        signal_end = self.signal[len(self.signal)-1]

        for i in range(n):
            if np.max(abs(self.signal[i:] - signal_end)) <= self.percentage * signal_end:
                self.settlingTimeIndex = i
                self.settlingTime = i*abs(self.t[0] - self.t[1])
                break

    def __getSSHighValue(self):
        self.SSHighValue = np.mean(self.signal[self.settlingTimeIndex:])

    def __getSSLowValue(self):
        self.SSLowValue = np.mean(self.signal[:self.inflectionTimeIndex])

    def __getRiseTime(self):

        i=self.inflectionTimeIndex
        prev_diff = abs((self.signal.copy()[i]) - (((self.SSHighValue)*0.1)))
        curr_diff = abs((self.signal.copy()[i+1]) - (((self.SSHighValue)*0.1)))
        while curr_diff < prev_diff:
            prev_diff = abs((self.signal.copy()[i]) - (((self.SSHighValue)*0.1)))
            curr_diff = abs((self.signal.copy()[i+1]) - (((self.SSHighValue)*0.1)))
            i += 1
        self.idxTen = i-1

        i=self.inflectionTimeIndex
        prev_diff = abs((self.signal.copy()[i]) - (((self.SSHighValue)*0.9)))
        curr_diff = abs((self.signal.copy()[i+1]) - (((self.SSHighValue)*0.9)))
        while curr_diff < prev_diff:
            prev_diff = abs((self.signal.copy()[i]) - (((self.SSHighValue)*0.9)))
            curr_diff = abs((self.signal.copy()[i+1]) - (((self.SSHighValue)*0.9)))
            i += 1
        self.idxNinety = i-1


        # self.idxTen = np.argmin(np.abs((self.signal.copy() - ((self.SSHighValue)*0.1 + off_set))))

        # self.idxNinety = np.argmin(np.abs(self.signal.copy() - ((self.SSHighValue)*0.9)))

        timeTen = self.t[self.idxTen]

        timeNinety = self.t[self.idxNinety]

        self.riseTime = (timeNinety - timeTen)

        '''        idxTen = np.argmin(np.abs(self.signal.copy() - self.SSHighValue*0.1))
        idxNinety = np.argmin(np.abs(self.signal.copy() - self.SSHighValue*0.9))
        timeTen = self.t[idxTen]
        timeNinety = self.t[idxNinety]
        self.riseTime = timeNinety - timeTen
        self.ten = idxTen
        self.ninety = idxNinety'''

    def __getOvershoot(self):
        self.overshoot = abs(float((self.signal[self.settlingMaxIndex] - self.SSHighValue)/self.SSHighValue))

class SetPoint:

    def __init__(self,response):

        self.response = response
        self.__getSetPoint()
    
    def __getSetPoint(self):

        self.sp = np.zeros(len(self.response.signal))
        self.inflectionTimeIndex = self.response.inflectionTimeIndex
        self.sp[:self.inflectionTimeIndex] = self.response.SSLowValue
        self.sp[self.inflectionTimeIndex:] = self.response.SSHighValue

# # TEMPLATE
# drive = np.genfromtxt('../fopdt_param_test_signals/input.csv')
# signal = np.genfromtxt('../fopdt_param_test_signals/output.csv')
# t = np.genfromtxt('../fopdt_param_test_signals/time.csv')

# # test = FopdtMeasurements(signal,drive,t)
# test = ResponseMeasurements(signal,t)
# # print(test.ThetaP)