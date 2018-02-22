#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 12:36:22 2018

@author: lsm
"""
import pandas as pd
from emotions_classifier import *

ec = EmotionsClassifier()
ec.load_neural_net('default')
#ec.load_checkpoint()
excited_val = pd.read_csv('excited10.csv')
calm_val = pd.read_csv('calm10.csv')
angry_val = pd.read_csv('angry10.csv')
sad_val = pd.read_csv('depressed10.csv')

r_excited = excited_val.text.apply(ec.run_neural_network)
r_calm = calm_val.text.apply(ec.run_neural_network)
r_angry = angry_val.text.apply(ec.run_neural_network)
r_sad = sad_val.text.apply(ec.run_neural_network)
