import time


def get_time():
    return time.strftime("[%Y-%m-%d %H:%M:%S]", time.gmtime())


def fit_and_measure_time(func):
    def func_wrapper(self, X_train, y_train):
        start_time = time.time()
        func(self, X_train, y_train)
        self.train_time = time.time() - start_time

    return func_wrapper