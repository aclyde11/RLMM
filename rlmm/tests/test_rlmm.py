"""
Unit and regression test for the rlmm package.
"""

# Import package, test suite, and other packages as needed
import rlmm
import pytest
import sys

def test_rlmm_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "rlmm" in sys.modules
