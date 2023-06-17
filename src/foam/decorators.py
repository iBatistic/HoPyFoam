"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Collection of function decorators
"""

from time import perf_counter
from functools import wraps

def timed(fn):

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        start = perf_counter()
        result = fn(self, *args, **kwargs)
        end = perf_counter()
        elapsed = end - start

        # Just for display purporse
        args_ = [str(a) for a in args]
        kwargs_ = ['{0}= {1}'.format(k, v) for (k, v) in kwargs.items()]
        all_args = args_ + kwargs_
        args_str = ','.join(all_args)

        # print('Function {0}({1}) took {2:.6f}s to run'.format(fn.__name__,
        #                                                 args_str, elapsed))

        print('{0}.{1} took {2:.6f}s to run'.format(self.__class__.__name__,
                                                    fn.__name__, elapsed))

        return result

    return wrapper
