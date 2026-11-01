# ABC adaptation: session headers and retained raw timestamps/text. Original harness SHA in README.
#!/usr/bin/env python3
"""Identical cells from the STEER_3 harness, against the live server.

Morrow: count / binary_search / hash-map prose. One 32-token warmup per type
(code uses warmup_noop). Users 1 through 8, three barrier-synchronized batches.
Temperature 0, top_p 1, both thinking flags false, cap 400, unique cache_salt.
S is (completion_tokens-1)/(last content - first content). Headline S is the
median of the three batch-mean stream rates. D is sum(N-1)/(latest last -
earliest first). E is sum(N)/(latest close - earliest send).

Cold prefill: filler lexicon seeded by Random(target), word+integer tokens,
unique front nonce and cache_salt. /tokenize brings the chat prompt within 16
of 2048, 8192, 32768, 65536, 131072. Three fresh requests. One output token.
Rate is prompt_tokens / (close - send).
"""
import json, random, statistics, threading, time, urllib.error, urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

HEADERS = {"Content-Type":"application/json"}
BASE = "http://127.0.0.1:8097"
MODEL = "deepseek-v4.1-flash"
OUT = "/mnt/roms/gnmac-exact-suite-20261031.json"
KW = {"thinking": False, "enable_thinking": False}
LEVELS = (1, 2, 3, 4, 5, 6, 7, 8)
REPS = 3
CAP = 400
WARM = 32
PREFILL = (2048, 8192, 32768, 65536, 131072)
FILLER_WORDS = (
    "amber basin cedar delta ember fjord garnet harbor iris juniper kestrel "
    "lumen meadow nimbus orchid pylon quartz raven sierra tundra umber vessel "
    "willow xenon yarrow zephyr"
).split()
CODE_TAIL = (
    "Output only Python source. No comments, no docstrings, no markdown fences. "
    "Then add tests and the helpers this needs. Keep writing code."
)
PROMPTS = {
    "count": "Count from 1 to 200. Output only the numbers, separated by spaces. No other text.",
    "code": "binary_search\ndef binary_search(nums, target) -> int: index of target in a sorted list, or -1.\n" + CODE_TAIL,
    "prose": (
        "Write a detailed step-by-step explanation of how a hash map works, "
        "including collision handling, resizing, and time complexity. Be thorough."
    ),
}
WARMUPS = {
    "count": PROMPTS["count"],
    "code": "warmup_noop\ndef warmup_noop(x): return x unchanged.\n" + CODE_TAIL,
    "prose": PROMPTS["prose"],
}


def post(path, body, timeout):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers=HEADERS,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def token_count(text):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "chat_template_kwargs": KW,
    }
    last = None
    for path in ("/tokenize", "/v1/tokenize"):
        try:
            got = post(path, body, 120)
        except urllib.error.HTTPError as exc:
            last = exc
            continue
        if "count" in got:
            return int(got["count"])
        if "tokens" in got:
            return len(got["tokens"])
        last = got
    raise RuntimeError("tokenize failed: %s" % last)


def filler(n_words, seed):
    rnd = random.Random(seed)
    return " ".join(
        "%s%d" % (rnd.choice(FILLER_WORDS), rnd.randint(0, 999))
        for _ in range(n_words)
    )


def calibrate(target, nonce):
    def text(n_words):
        return "[%s] %s" % (nonce, filler(n_words, target))

    lo, hi = 1, max(8, int(target * 0.9))
    best_n, best_c, best_gap = 1, token_count(text(1)), 10 ** 9
    for _ in range(28):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        count = token_count(text(mid))
        gap = abs(count - target)
        if gap < best_gap:
            best_n, best_c, best_gap = mid, count, gap
        if gap <= 16:
            return text(mid), count
        if count < target:
            lo = mid + 1
        else:
            hi = mid - 1
    # One-word walk from the closest binary-search hit.
    step = 1 if best_c < target else -1
    n_words = best_n
    for _ in range(64):
        nxt = n_words + step
        if nxt < 1:
            break
        count = token_count(text(nxt))
        if abs(count - target) < best_gap:
            best_n, best_c, best_gap = nxt, count, abs(count - target)
            n_words = nxt
        else:
            break
        if best_gap <= 16:
            break
    if best_gap > 16:
        raise RuntimeError("prefill %s calibrated to %s (off by %s)" % (target, best_c, best_gap))
    return text(best_n), best_c


def stream(prompt, max_tokens, timeout):
    salt = uuid.uuid4().hex
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": KW,
        "cache_salt": salt,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers=HEADERS,
    )
    send = time.perf_counter()
    first = last = None
    pieces = []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            now = time.perf_counter()
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for choice in ev.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning") or "")
                if piece:
                    pieces.append(piece)
                    if first is None:
                        first = now
                    last = now
    close = time.perf_counter()
    usage = usage or {}
    return {
        "salt": salt, "content": "".join(pieces),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "send": send, "first": first, "last": last, "close": close,
    }


def morrow_batch(name, users):
    barrier = threading.Barrier(users)
    prompt = PROMPTS[name]

    def one(_i):
        barrier.wait()
        return stream(prompt, CAP, 900)

    with ThreadPoolExecutor(max_workers=users) as pool:
        rows = list(pool.map(one, range(users)))
    rates, ttfts = [], []
    for row in rows:
        n = row["completion_tokens"] or 0
        if row["first"] is not None and row["last"] is not None and n > 1 and row["last"] > row["first"]:
            rates.append((n - 1) / (row["last"] - row["first"]))
        ttft = (row["first"] - row["send"]) if row["first"] is not None else None
        row["ttft_s"] = ttft
        if ttft is not None:
            ttfts.append(ttft)
    usable = [r for r in rows if r["first"] is not None and r["last"] is not None and r["completion_tokens"]]
    decode_tokens = sum((r["completion_tokens"] - 1) for r in usable)
    span = (max(r["last"] for r in usable) - min(r["first"] for r in usable)) if usable else 0
    total_tokens = sum(r["completion_tokens"] for r in usable)
    wall = (max(r["close"] for r in rows) - min(r["send"] for r in rows)) if rows else 0
    return {
        "users": users, "requests": rows, "wall_s": wall,
        "batch_mean_s": statistics.mean(rates) if rates else None,
        "d_tok_s": (decode_tokens / span) if span else None,
        "e_tok_s": (total_tokens / wall) if wall else None,
        "ttft_median_s": statistics.median(ttfts) if ttfts else None,
        "completion_tokens": [r["completion_tokens"] for r in rows],
        "prompt_tokens": [r["prompt_tokens"] for r in rows],
        "cached_tokens": [r["cached_tokens"] for r in rows],
        "salts": [r["salt"] for r in rows],
    }


def median(values):
    got = [v for v in values if v is not None]
    return statistics.median(got) if got else None


def save(doc):
    with open(OUT, "w") as handle:
        json.dump(doc, handle)


def main():
    doc = {"model": MODEL, "morrow": [], "prefill": []}
    print("tokenize probe", token_count("hello"), flush=True)
    for name in ("count", "code", "prose"):
        stream(WARMUPS[name], WARM, 300)
        print("warmup", name, flush=True)
        for users in LEVELS:
            batches = [morrow_batch(name, users) for _ in range(REPS)]
            cell = {
                "prompt": name,
                "users": users,
                "s_tok_s": median([b["batch_mean_s"] for b in batches]),
                "d_tok_s": median([b["d_tok_s"] for b in batches]),
                "e_tok_s": median([b["e_tok_s"] for b in batches]),
                "ttft_s": median([b["ttft_median_s"] for b in batches]),
                "batches": batches,
            }
            doc["morrow"].append(cell)
            save(doc)
            print(
                "MORROW %s C%s S %.2f D %.2f E %.2f TTFT %.3f out %s"
                % (name, users, cell["s_tok_s"], cell["d_tok_s"], cell["e_tok_s"], cell["ttft_s"],
                   batches[-1]["completion_tokens"]),
                flush=True,
            )
    for target in PREFILL:
        reps = []
        for _ in range(REPS):
            nonce = uuid.uuid4().hex
            text, calibrated = calibrate(target, nonce)
            row = stream(text, 1, 1800)
            wall = row["close"] - row["send"]
            pt = row["prompt_tokens"]
            if pt is None or abs(pt - target) > 20:
                raise RuntimeError("prefill usage %s for target %s (calibrated %s)" % (pt, target, calibrated))
            if row["completion_tokens"] != 1:
                raise RuntimeError("prefill completion_tokens %s" % row["completion_tokens"])
            if isinstance(row["cached_tokens"], int) and row["cached_tokens"] > 0:
                raise RuntimeError("prefill cached %s" % row["cached_tokens"])
            rec = {
                "target": target,
                "calibrated_tokens": calibrated,
                "prompt_tokens": pt,
                "completion_tokens": row["completion_tokens"],
                "cached_tokens": row["cached_tokens"],
                "wall_s": wall,
                "ttft_s": (row["first"] - row["send"]) if row["first"] is not None else None,
                "prompt_tok_s_wall": pt / wall,
                "salt": row["salt"],
            }
            reps.append(rec)
            print("PREFILL %s tok wall %.1f" % (pt, rec["prompt_tok_s_wall"]), flush=True)
        doc["prefill"].append({
            "target": target,
            "prompt_tok_s_wall": median([r["prompt_tok_s_wall"] for r in reps]),
            "wall_s": median([r["wall_s"] for r in reps]),
            "prompt_tokens": [r["prompt_tokens"] for r in reps],
            "reps": reps,
        })
        save(doc)
        print("PREFILL %s median %.2f" % (target, doc["prefill"][-1]["prompt_tok_s_wall"]), flush=True)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
