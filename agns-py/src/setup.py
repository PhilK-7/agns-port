import os


def setup_params(remote: bool, gpu_nrs: tuple = (0,)):
    """
    Sets necessary environment parameters for CUDA, and returns the path to the 'data' directory.

    :param remote: whether you use a program on the server (assuming an extra 'data' directory in 'storage-private),
        or locally
    :param gpu_nrs: a tuple stating which GPU(s) to use (by index)
    :return: the path to the 'data' directory
    """
    # set parameters for CUDA environment
    if remote:
        os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(list(map(lambda i: str(i), gpu_nrs)))
        dap = os.path.expanduser('~') + '/storage-private/data/'
    else:
        dap = '../data/'

    return dap
