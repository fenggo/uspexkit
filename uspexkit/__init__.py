"""USPeX Kit — USPEX crystal structure prediction post-processing toolkit."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "pred":
        from uspexkit.core import pred as _pred
        return _pred
    if name == "calc":
        from uspexkit.core import calc as _calc
        return _calc
    if name == "traj":
        from uspexkit.core import traj as _traj
        return _traj
    if name == "zmat":
        from uspexkit.core import zmat as _zmat
        return _zmat
    if name == "fdf":
        from uspexkit.core import fdf as _fdf
        return _fdf
    if name == "sample":
        from uspexkit.core import sample as _sample
        return _sample
    raise AttributeError(f"module 'uspexkit' has no attribute {name!r}")
