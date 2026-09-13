"""Tokeymeter v0.12.0 — adversarial stress battery (review pass)."""
import asyncio, os, sqlite3, sys, tempfile, threading, time, traceback

import tokeymeter as tk

RESULTS = []
def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + ("  · " + detail if detail else ""))

def fresh():
    tk.reset()
    tk.reset_savings()
    tk.set_default_store(tk.MemoryStore())

# ── 1. Thread stampede: single-flight collapse ─────────────────────────
def t1():
    fresh()
    computed = []
    lock = threading.Lock()
    @tk.cache()
    def f(prompt: str) -> str:
        with lock: computed.append(1)
        time.sleep(0.05)
        return "answer:" + prompt
    errs, results = [], []
    def worker():
        try: results.append(f("same prompt"))
        except Exception as e: errs.append(e)
    threads = [threading.Thread(target=worker) for _ in range(64)]
    [t.start() for t in threads]; [t.join() for t in threads]
    check("T1 stampede no-errors", not errs, f"errs={len(errs)}")
    check("T1 single-flight collapse", len(computed) <= 3, f"computes={len(computed)} for 64 concurrent")
    check("T1 result consistency", len(set(results)) == 1 and len(results) == 64)

# ── 2. Async stampede ──────────────────────────────────────────────────
def t2():
    fresh()
    computed = []
    @tk.cache()
    async def f(prompt: str) -> str:
        computed.append(1)
        await asyncio.sleep(0.05)
        return "a:" + prompt
    async def main():
        return await asyncio.gather(*[f("same") for _ in range(300)])
    res = asyncio.run(main())
    check("T2 async stampede collapse", len(computed) <= 3, f"computes={len(computed)} for 300")
    check("T2 async consistency", len(set(res)) == 1)

# ── 3. Tenant isolation: same prompt must not bleed across tenants ─────
def t3():
    fresh()
    calls = []
    @tk.cache()
    def f(prompt: str) -> str:
        calls.append(tk.policy.current_tenant() if hasattr(tk.policy, "current_tenant") else "?")
        return f"resp-{len(calls)}:{prompt}"
    with tk.tenant_scope("tenant-A"):
        a = f("shared question")
    with tk.tenant_scope("tenant-B"):
        b = f("shared question")
    with tk.tenant_scope("tenant-A"):
        a2 = f("shared question")
    check("T3 tenant isolation (no cross-serve)", a != b, f"A={a!r} B={b!r}")
    check("T3 tenant cache works within tenant", a == a2)

# ── 4. Audit chain: tamper and deletion must be detected ──────────────
def t4():
    import sqlite3
    with tempfile.TemporaryDirectory() as d:
        dbp = os.path.join(d, "audit.db")
        alog = tk.audit.AuditLog(path=dbp, install_secret_path=os.path.join(d, "secret"),
                                 signing_key_path=os.path.join(d, "key"), durable=True)
        alog.attach()
        fresh()
        @tk.cache()
        def f(p): return "r:" + p
        for i in range(25): f(f"q{i}")
        f("q0")
        alog.flush(); alog.detach()
        v1 = alog.verify_chain()
        con = sqlite3.connect(dbp)
        con.execute("UPDATE audit_entries SET decision_type='cache_hit' WHERE seq=7"); con.commit()
        v2 = alog.verify_chain()
        con.execute("DELETE FROM audit_entries WHERE seq=12"); con.commit(); con.close()
        v3 = alog.verify_chain()
        check("T4 chain verifies clean", v1.valid, f"entries={v1.entries_verified}")
        check("T4 content tamper detected", not v2.valid, f"first_bad_seq={v2.first_bad_seq}")
        check("T4 deletion detected", not v3.valid, f"first_bad_seq={v3.first_bad_seq}")
        alog.close()

# ── 5. Restore safety: corrupt backup refused, live cache intact ──────
def t5():
    with tempfile.TemporaryDirectory() as d:
        tk.reset(); tk.reset_savings()
        tk.set_default_store(tk.SQLiteStore(os.path.join(d, "cache.db")))
        @tk.cache()
        def f(p): return "v:" + p
        for i in range(10): f(f"k{i}")
        exp = tk.admin.export_cache(os.path.join(d, "backup.export"))
        target = exp["path"]
        if os.path.isdir(target):
            target = max((os.path.join(target, x) for x in os.listdir(target)), key=os.path.getsize)
        with open(target, "r+b") as fh:
            fh.seek(os.path.getsize(target)//2); fh.write(b"CORRUPT!")
        r = tk.admin.import_cache(exp["path"], mode="replace")
        check("T5 corrupt backup refused", bool(r.get("aborted")) and r.get("imported", 1) == 0,
              f"integrity={r.get('integrity')}")
        check("T5 live cache intact after refusal", f("k3") == "v:k3")
        e2 = tk.admin.export_cache(os.path.join(d, "b2.export"))
        r2 = tk.admin.import_cache(e2["path"], mode="replace")
        check("T5 clean restore succeeds", not r2.get("aborted") and r2.get("imported") == 10,
              f"imported={r2.get('imported')}")

# ── 6. Fail-open: store whose OPERATIONS raise must not break calls ────
def t6():
    class BrokenStore:
        def get(self,k): raise RuntimeError("disk on fire")
        def set(self,k,v,**kw): raise RuntimeError("disk on fire")
        def delete(self,k): raise RuntimeError("disk on fire")
        def begin_singleflight(self,*a,**k): raise RuntimeError("disk on fire")
        def end_singleflight(self,*a,**k): raise RuntimeError("disk on fire")
    tk.reset(); tk.reset_savings()
    tk.set_default_store(BrokenStore())
    @tk.cache()
    def f(p: str) -> str: return "ok:" + p
    try:
        r1, r2 = f("x"), f("x")
        check("T6 fail-open (store ops raise)", r1 == r2 == "ok:x")
    except Exception as e:
        check("T6 fail-open (store ops raise)", False, repr(e)[:140])

# ── 7. Hostile payloads: huge values, weird unicode, null bytes ────────
def t7():
    fresh()
    @tk.cache()
    def f(p: str) -> str: return p[::-1]
    big = "x" * (5 * 1024 * 1024)
    weird = "ключ-\u0000-鍵-🔐-\ud800?".encode("utf-8", "surrogatepass").decode("utf-8", "replace")
    try:
        a = f(big); a2 = f(big)
        b = f(weird); b2 = f(weird)
        check("T7 5MB payload round-trip", a == a2 and len(a) == len(big))
        check("T7 hostile unicode round-trip", b == b2)
    except Exception as e:
        check("T7 hostile payloads", False, repr(e))

# ── 8. Bounded memory store under churn ────────────────────────────────
def t8():
    tk.reset(); tk.reset_savings()
    store = tk.MemoryStore(max_entries=1000)
    tk.set_default_store(store)
    @tk.cache()
    def f(p: str) -> str: return "v" + p
    for i in range(20000): f(f"key-{i}")
    n = len(getattr(store, "_data", {})) or getattr(store, "size", lambda: -1)()
    check("T8 memory store bounded", 0 < n <= 1100, f"entries={n} after 20k distinct keys")

for t in (t1, t2, t3, t4, t5, t6, t7, t8):
    try: t()
    except Exception:
        check(t.__name__ + " harness", False, traceback.format_exc(limit=2).replace("\n", " | ")[:200])

fails = [r for r in RESULTS if not r[1]]
print(f"\n==== {len(RESULTS)-len(fails)}/{len(RESULTS)} checks passed ====")
sys.exit(1 if fails else 0)
