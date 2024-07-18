import numpy as np
import random
import time
import timeit

def perf_counter():#performance counter
    '''
    返回性能计数器的值（以小数秒为单位）作为浮点数，即具有最高可用分辨率的时钟，以测量短持续时间。 它确实包括睡眠期间经过的时间，并且是系统范围的。
通常perf_counter()用在测试代码时间上，具有最高的可用分辨率。不过因为返回值的参考点未定义，因此我们测试代码的时候需要调用两次，做差值。
    :return:
    '''
    try:
        return time.perf_counter()
    except:
        timeit.default_timer()