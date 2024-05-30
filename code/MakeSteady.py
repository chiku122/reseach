from GetPara import get_para
from sklearn import preprocessing

def make_steady(time_series, diff_num):
    diff = [[],[],[]]
    diff[0] = time_series
    diff[1] = time_series.diff(periods=1).fillna(0)
    diff[2] = diff[1].diff(periods=1).fillna(0)

    return(diff[diff_num])

