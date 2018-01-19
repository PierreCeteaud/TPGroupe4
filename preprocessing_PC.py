# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:04:07 2018

@author: Pierre
"""

import librosa
import numpy as np

Signal,hz= librosa.load('06-11-22_16.wav',16000,True)
Ecart=np.mean(Signal)
Output=(Signal-Ecart)
librosa.output.write_wav('06-11-22_16c.wav',Output,hz)
Output=Output/np.max(np.abs(Signal))
librosa.output.write_wav('06-11-22_16cn.wav',Output,hz)