"""Compatibility alias (K2 layout).
"""
import sys as _sys
from tokeymeter.engines.execution import integrations as _pkg
from tokeymeter.engines.execution.integrations import openai as _o, anthropic as _a, openai_async as _oa, universal as _u
from tokeymeter.engines.economics import reconcile as _r
for _n,_m in (('openai',_o),('anthropic',_a),('openai_async',_oa),('universal',_u),('reconcile',_r)):
    _sys.modules[__name__+'.'+_n]=_m; setattr(_pkg,_n,_m)
_sys.modules[__name__]=_pkg
