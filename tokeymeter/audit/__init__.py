"""Compatibility alias (K2 layout).
"""
import sys as _sys
from tokeymeter.engines.trust import audit as _pkg
from tokeymeter.engines.trust.audit import log as _l, proof as _p, signers as _s
_sys.modules[__name__+'.log']=_l; _sys.modules[__name__+'.proof']=_p; _sys.modules[__name__+'.signers']=_s
_sys.modules[__name__]=_pkg
