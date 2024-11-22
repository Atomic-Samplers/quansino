from __future__ import annotations


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
    data_format = data_format.lstrip(":")

    align = ">" if data_format[0] not in "<>^" else data_format[0]

    width = ""
    for char in data_format.lstrip("<>^"):
        if char.isdigit():
            width += char
        else:
            break

    return f"{align}{width}s" if width else ">10s"
