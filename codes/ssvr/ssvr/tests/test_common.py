import pytest
from sklearn.utils.estimator_checks import check_estimator
from ssvr.classes import SSVR


check_estimator(SSVR)
