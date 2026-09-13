"""Compatibility alias (K2 layout).
"""
import sys as _sys
from tokeymeter.engines.trust import cipher as _c
from tokeymeter.engines.optimization import redis_store as _rs
_sys.modules[__name__+'.cipher']=_c; _sys.modules[__name__+'.redis_store']=_rs
from tokeymeter.engines.trust.cipher import Cipher, FernetCipher, NoOpCipher
from tokeymeter.engines.optimization.redis_store import RedisStore
__all__=["RedisStore","Cipher","FernetCipher","NoOpCipher"]
cipher=_c; redis_store=_rs
