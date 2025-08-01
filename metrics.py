# finding PCC,CPCC,SPCC,Jaccard,MSD,JMSD,COS and ACOS similarity metrics for a and b dicts {'item1':rating1,'item2':rating2..}

# !/usr/bin/env python
# -*- coding: utf-8 -*-
from math import *


# summ XY for 'both' common elements
def _scal(x, y, both):
    return sum(x[i] * y[i] for i in both)


# summ X for 'both' common elements
def _one(x, both):
    return sum(x[i] for i in both)


# calculating common elements
def _both_elem_calc(a, b):
    return dict([(i, 1) for i in a if i in b])


# PCC - Pearson correlation coefficient
def PCC(a, b):
    both = _both_elem_calc(a, b)
    if both == {}: return 0
    n = len(both)
    onea = _one(a, a) / len(a)
    oneb = _one(b, b) / len(b)
    up = sum((a[i] - onea) * (b[i] - oneb) for i in both)
    down = sqrt(sum((a[i] - onea) ** 2 for i in both)) * sqrt(sum((b[i] - oneb) ** 2 for i in both))
    if down == 0:
        return 0
    return up / down


# CPCC - Constrained Pearson correlation coefficient
def CPCC(a, b, rsm=3):  # rsm - rating scale median
    both = _both_elem_calc(a, b)
    if both == {}: return 0
    up = sum((a[i] - rsm) * (b[i] - rsm) for i in both)
    down = sqrt(sum((a[i] - rsm) ** 2 for i in both)) * sqrt(sum((b[i] - rsm) ** 2 for i in both))
    if down == 0:
        return 0
    return up / down


# SPCC - Sigmoid function based Pearson coefficient
def SPCC(a, b):
    both = _both_elem_calc(a, b)
    if both == {}: return 0
    return PCC(a, b) * 1 / (1 + exp(-len(both) / 2))


# COS - Cosine measure
def COS(a, b):
    both = _both_elem_calc(a, b)
    if both == {}: return 0
    return _scal(a, b, both) / (sqrt(_scal(a, a, a)) * sqrt(_scal(b, b, b)))


# ACOS - Adjusted cosine measure
def ACOS(a, b):
    al = set(a) & set(b)
    a2 = dict([(i, a[i]) if i in a else (i, 0) for i in al])
    b2 = dict([(i, b[i]) if i in b else (i, 0) for i in al])
    onea = _one(a, a) / len(a)
    oneb = _one(b, b) / len(b)
    up = sum((a2[i] - onea) * (b2[i] - oneb) for i in al)
    down = sqrt(sum((a2[i] - onea) ** 2 for i in al)) * sqrt(sum((b2[i] - oneb) ** 2 for i in al))
    if down == 0:
        return 0
    return up / down


# Jaccard - Jaccard similarity
def Jaccard(a, b):
    both = _both_elem_calc(a, b)
    if both == {}: return 0
    return len(both) / (len(a) + len(b) - len(both))


# MSD - Mean squared differences
def MSD(a, b, scale=5):
    both = _both_elem_calc(a, b)
    if both == {}: return 0
    return 1 - sum(((a[i] - b[i]) / scale) ** 2 for i in both) / len(both)


# JMSD - Jaccard and MSD combined
def JMSD(a, b, scale=5):
    return Jaccard(a, b) * MSD(a, b)
