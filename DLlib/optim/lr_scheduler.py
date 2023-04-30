class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        pass


class StepLR(_LRScheduler):
    """StepLR"""

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        pass


class MultiStepLR(_LRScheduler):
    """MultiStepLR"""

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        pass


class ExponentialLR(_LRScheduler):
    """ExponentialLR"""

    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        pass
