from .client import Cluster, Job
from .context import Context, init


def connect(binary=None, home=None):
    return Cluster(binary, home)


__all__ = ["Cluster", "Job", "Context", "connect", "init"]
from .pool import PooledTrainer
__all__.append("PooledTrainer")
from .inference import OpenVINOModel, RemoteModel
__all__.extend(["RemoteModel", "OpenVINOModel"])
