# -*- coding: utf-8 -*-
import numpy as np
from sklearn.mixture import DPGMM
import load_data as ld

#ld.get_letor_3(64)
data_total = ld.load_pickle('','data_cluster.pickle')
print data_total.shape
# fit the data with DPGMM
dpgmm = DPGMM(covariance_type='diag', alpha=2., n_iter=1000, n_components=30)
dpgmm.fit(data_total)
print dpgmm.predict(data_total)
print 'aic criterion: %f' %dpgmm.aic(data_total)
print 'bic criterion: %f' %dpgmm.bic(data_total)

