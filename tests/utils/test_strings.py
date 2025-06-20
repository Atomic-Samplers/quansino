from __future__ import annotations

from quansino.utils.strings import get_auto_header_format


def test_get_auto_header_format():
    """Test the `get_auto_header_format` function."""
    assert get_auto_header_format("{:10.3f}") == "{:>10s}"
    assert get_auto_header_format("{:4s}") == "{:>4s}"
    assert get_auto_header_format("{:s}") == "{:>10s}"
    assert get_auto_header_format("{:>s}") == "{:>10s}"
    assert get_auto_header_format("{:>10s}") == "{:>10s}"
    assert get_auto_header_format("{:>10}") == "{:>10s}"
    assert get_auto_header_format("{:>10.3f}{:<10.3f}") == "{:>10s}{:<10s}"
    assert get_auto_header_format("{:>10.3f}{:>10.3f}") == "{:>10s}{:>10s}"
