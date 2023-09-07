import re
import os
from colorama import init as color_init
from colorama import Fore, Style
import functools
import pyvips
import warnings
import contextlib
from collections import defaultdict

color_init()

def print_warning(msg, warning_type=UserWarning, rgb=Fore.RED):
    """Print warning message with color
    """
    warning_msg = f"{rgb}{msg}{Style.RESET_ALL}"
    if warning_type is None:
        print(warning_msg)
    else:
        warnings.simplefilter('always', warning_type)
        warnings.warn(warning_msg, warning_type)


def deprecated_args(**aliases):
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            rename_kwargs(f.__name__, kwargs, aliases)
            return f(*args, **kwargs)

        return wrapper

    return deco


def rename_kwargs(func_name, kwargs, aliases):
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise TypeError('{} received both {} and {}'.format(
                    func_name, alias, new))

            msg = f'{alias} is deprecated; use {new} instead'
            print_warning(msg, DeprecationWarning)

            kwargs[new] = kwargs.pop(alias)


@contextlib.contextmanager
def HiddenPrints():
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        yield


def get_name(f):
    """
    To get an object's name, remove image type extension from filename
    """
    if re.search("\.", f) is None:
        # Extension already removed
        return f

    f = os.path.split(f)[-1]

    if f.endswith(".ome.tiff") or f.endswith(".ome.tif"):
        back_slice_idx = 2
    else:
        back_slice_idx = 1

    img_name = "".join([".".join(f.split(".")[:-back_slice_idx])])

    return img_name


def sort_nicely(l):
    """Sort the given list in the way that humans expect.
    """
    l.sort(key=lambda s: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)])


def get_elapsed_time_string(elapsed_time, rounding=3):
    """Format elapsed time

    Parameters
    ----------
    elapsed_time : float
        Elapsed time in seconds

    rounding : int
        Number of decimal places to round

    Returns
    -------
    scaled_time : float
        Scaled amount of elapsed time

    time_unit : str
        Time unit, either seconds, minutes, or hours

    """

    if elapsed_time < 60:
        scaled_time = elapsed_time
        time_unit = "seconds"

    elif 60 <= elapsed_time < 60 ** 2:
        scaled_time = elapsed_time / 60
        time_unit = "minutes"

    else:
        scaled_time = elapsed_time / (60 ** 2)
        time_unit = "hours"

    scaled_time = round(scaled_time, rounding)

    return scaled_time, time_unit


def get_vips_version():
    v = f"{pyvips.vips_lib.VIPS_MAJOR_VERSION}.{pyvips.vips_lib.VIPS_MINOR_VERSION}.{pyvips.vips_lib.VIPS_MICRO_VERSION}"
    return v


def etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

