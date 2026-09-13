"""SecurityPolicy — opt-in, enforceable trust defaults.

Default policy is permissive (no behavior change). When a requirement is enabled,
the unsafe construction REFUSES to build (SecurityPolicyError).
"""
import os, tempfile
import pytest
import tokeymeter
from tokeymeter.policy import SecurityPolicy, SecurityPolicyError
from tokeymeter.storage import MemoryStore
from tokeymeter.privacy import DefaultRedactor

try:
    import redislite
    from cryptography.fernet import Fernet
    from tokeymeter.backends.redis_store import RedisStore
    from tokeymeter.backends.cipher import FernetCipher, NoOpCipher
    _REDIS = True
except Exception:
    _REDIS = False
redis_only = pytest.mark.skipif(not _REDIS, reason="redis/crypto not installed")


@pytest.fixture(autouse=True)
def _reset():
    tokeymeter.reset_security_policy()
    tokeymeter.set_default_store(MemoryStore())
    tokeymeter.set_default_redactor(None)
    yield
    tokeymeter.reset_security_policy()
    tokeymeter.set_default_redactor(None)


def _db():
    return os.path.join(tempfile.mkdtemp(), "p.rdb")


def test_default_policy_is_permissive():
    p = tokeymeter.get_security_policy()
    assert not any(vars(p).values())            # all False by default


@redis_only
def test_require_encryption_refuses_plaintext():
    tokeymeter.set_security_policy(require_encryption=True)
    with pytest.raises(SecurityPolicyError):
        RedisStore(client=redislite.Redis(_db()))
    with pytest.raises(SecurityPolicyError):
        RedisStore(client=redislite.Redis(_db()), cipher=NoOpCipher())
    # real cipher is accepted
    RedisStore(client=redislite.Redis(_db()), cipher=FernetCipher(key=Fernet.generate_key()))


@redis_only
def test_require_keyed_cache_refuses_unkeyed():
    tokeymeter.set_security_policy(require_keyed_cache=True)
    with pytest.raises(SecurityPolicyError):
        RedisStore(client=redislite.Redis(_db()),
                   cipher=FernetCipher(key=Fernet.generate_key()))


@redis_only
def test_secure_factory_satisfies_strict_policy():
    tokeymeter.set_security_policy(require_encryption=True, require_keyed_cache=True)
    s = RedisStore.secure(client=redislite.Redis(_db()),
                          encryption_key=Fernet.generate_key(), key_secret=b"k"*32)
    assert s is not None


def test_require_nonrepudiable_audit_refuses_hmac():
    from tokeymeter.audit import AuditLog, Ed25519Signer
    tokeymeter.set_security_policy(require_nonrepudiable_audit=True)
    d = tempfile.mkdtemp()
    with pytest.raises(SecurityPolicyError):
        AuditLog(path=os.path.join(d, "a.db"),
                 install_secret_path=os.path.join(d, "s"),
                 signing_key_path=os.path.join(d, "k"))     # default HMAC -> refused
    d2 = tempfile.mkdtemp()
    AuditLog(path=os.path.join(d2, "a.db"),
             install_secret_path=os.path.join(d2, "s"),
             signing_key_path=os.path.join(d2, "k"),
             signer=Ed25519Signer.generate())               # asymmetric -> accepted


def test_require_redaction_refuses_unredacted_calls():
    tokeymeter.set_security_policy(require_redaction=True)
    @tokeymeter.cache(model="m")
    def no_red(p): return "ok"
    with pytest.raises(SecurityPolicyError):
        no_red("x")
    @tokeymeter.cache(model="m", redactor=DefaultRedactor())
    def with_red(p): return "ok"
    assert with_red("contact a@b.com") == "ok"               # redactor present -> allowed


def test_high_stakes_honors_redaction_even_without_policy():
    """Redaction is an egress control, not an optimization: high_stakes strips
    PII before the verbatim model call when a redactor is configured."""
    seen = {}
    @tokeymeter.cache(model="m", high_stakes=True, redactor=DefaultRedactor())
    def hs(prompt):
        seen["p"] = prompt
        return "ok"
    hs("email me at alice@example.com please")
    assert "alice@example.com" not in seen["p"]


def test_policy_set_get_reset_and_describe():
    p = tokeymeter.set_security_policy(require_encryption=True)
    assert tokeymeter.get_security_policy().require_encryption
    assert "require_encryption" in p.describe()
    tokeymeter.reset_security_policy()
    assert not any(vars(tokeymeter.get_security_policy()).values())
    with pytest.raises(ValueError):
        tokeymeter.set_security_policy(bogus_flag=True)            # unknown option rejected
