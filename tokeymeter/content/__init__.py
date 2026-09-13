"""Compatibility alias (K2 layout).
"""
import sys as _sys
from tokeymeter.engines.governance import content as _pkg
from tokeymeter.engines.governance.content import secrets as _se
_sys.modules[__name__+'.secrets']=_se
_sys.modules[__name__]=_pkg
