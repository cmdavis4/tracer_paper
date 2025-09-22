import re
import datetime as dt
import numpy as np
from pathlib import Path


def str_to_key(key_str, parse_ints=True):
    key = []
    for pair in key_str.split("_"):
        pair = pair.split("-")
        k = pair[0]
        v = pair[1]
        if parse_ints:
            try:
                v = int(v)
            except:
                pass
        key.append((k, v))
    return tuple(key)


def key_to_str(key, pair_sep="-", param_sep="_", strip_underscores=False):
    if isinstance(key, dict):
        key = tuple(key.items())
    pairs = [[str(x) for x in pair] for pair in key]
    if strip_underscores:
        pairs = [[x.replace("_", "") for x in pair] for pair in pairs]
    return "_".join(["-".join(pair) for pair in pairs])


def key_in_selector(key, selector):
    key = dict(key)
    selector = dict(selector)
    for k, v in selector.items():
        if isinstance(v, str):
            v = [v]
        if key.get(k) not in v:
            return False
    return True


def filter_paths_by_selector(paths, selector, parse_ints=True):
    filtered = []
    for dir in paths:
        if dir.name == "archive":
            continue
        # Pull out all of the keys from the directory name
        pairs = re.findall(r"[a-zA-Z0-9]+\-[a-zA-Z0-9]+", dir.name)
        try:
            key = str_to_key("_".join(pairs), parse_ints=parse_ints)
            if key_in_selector(key, selector):
                filtered.append(dir)
        except IndexError:
            # This filename doesn't have any key-value pairs, don't include it
            continue
    return filtered


def current_dt_str(format="%Y%m%d%H%M%S"):
    return dt.datetime.now().strftime(format)


def filter_to_points(da, as_dicts=True):
    # Assume da is a DataArray of the boolean we want to filter by
    # assert set(da.dims) == set(['x', 'y', 'z'])
    s = da.to_series()
    filtered_points = s.loc[s.astype(bool)]
    # Dumb that this is the only way I can figure out to convert the array of tuples to a 2D array
    if as_dicts:
        return [
            {dim_name: l[dim_ix] for dim_ix, dim_name in enumerate(da.dims)}
            for l in filtered_points.index
        ]
    else:
        return np.array([np.array(l) for l in filtered_points.index])


def raise_if_exists(fpath):
    if Path(fpath).exists():
        raise OSError(f"Output path {str(fpath)} exists and exist_ok=False was passed")


def append_to_stem(fpath, appendage):
    return Path(fpath).with_stem(fpath.stem + appendage)
