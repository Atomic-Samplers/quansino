from __future__ import annotations

import re


def get_auto_header_format(data_format: str) -> str:
    """
    Convert a data format string to a header format string.

    Parameters
    ----------
    data_format
        The data format string to convert.

    Returns
    -------
    str
        The converted header format string to maintain alignment.
        Returns '>10s' if the function fails to parse the data format.

    Examples
    --------
    get_header_format('10.3f') -> '>10s'
    get_header_format('4s') -> '>4s'
    """
    return re.sub(
        r":([<>^])?(\d+)?[^}]*",
        lambda m: f':{m.group(1) or ">"}{m.group(2) or "10"}s',
        data_format,
    )
