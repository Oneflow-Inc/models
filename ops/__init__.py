from .roi_align import RoIAlign
from .nms import nms


def lib_path():
    import os
    import glob

    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    libs = glob.glob(os.path.join(dir_path, "*.so"))
    assert len(libs) > 0, f"no .so found in {dir_path}"
    return libs[0]
