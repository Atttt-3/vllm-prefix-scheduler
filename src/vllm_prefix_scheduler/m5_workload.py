"""Deterministic synthetic token inputs and absolute open-loop M5 schedules."""

import random
from fractions import Fraction

SCENARIOS = ("burst", "mixed", "ample", "all-cold", "serial", "mechanism")
MODES = ("native", "off", "prefix", "guarded")
NANOSECOND = 1_000_000_000


def _exact_tokens(encode, nonce, count):
    text = f"M5 independent synthetic reference {nonce:016x}. "
    tokens = encode(text + (text + "A quiet bridge crosses a narrow stream. ") * (count // 8 + 1))
    if len(tokens) < count or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("tokenizer did not supply the requested exact token sequence")
    return tokens[:count]


def prepare_workload(
    config, encode, scenario, seed, *, rate=None, interval_ms=None, count=None, mechanism="pressure"
):
    """Build immutable shared prefixes and per-request suffixes using the frozen tokenizer."""
    if scenario not in SCENARIOS:
        raise ValueError("unsupported M5 scenario")
    rules = config["scenarios"][scenario]
    block_size = config["block_size"]
    count = config["request_count"] if count is None else count
    if scenario == "mechanism":
        count = 2
        if rate is not None or interval_ms is not None or mechanism not in ("pressure", "ample"):
            raise ValueError("mechanism accepts only pressure/ample variant and no arrival rate")
        spacing = None
    elif scenario == "serial":
        if rate is not None or interval_ms is None or Fraction(str(interval_ms)) <= 0:
            raise ValueError("serial requires a positive fixed interval")
        spacing = Fraction(str(interval_ms)) * 1_000_000
    else:
        if interval_ms is not None or rate is None or Fraction(str(rate)) <= 0:
            raise ValueError("natural load requires a positive absolute rate")
        spacing = NANOSECOND / Fraction(str(rate))
    if type(count) is not int or count < 1:
        raise ValueError("request count must be a positive integer")

    rng = random.Random(seed)
    first_blocks = set()

    def disjoint(length):
        for _ in range(32):
            nonce = rng.getrandbits(64)
            token_ids = _exact_tokens(encode, nonce, length)
            first = tuple(token_ids[:block_size])
            if len(first) != block_size:
                raise ValueError("every independent prompt must contain a full first block")
            if first not in first_blocks:
                first_blocks.add(first)
                return token_ids, f"{nonce:016x}"
        raise ValueError("synthetic tokenizer failed to generate distinct first blocks")

    def entry(request_id, token_ids, salt, family, category, offset_ns=None):
        return {
            "request_id": request_id,
            "offset_ns": offset_ns,
            "class": category,
            "family": family,
            "cache_salt": salt,
            "prompt_token_ids": token_ids,
        }

    if scenario == "mechanism":
        prefix_length = rules.get("prefix_tokens", 2048)
        prefix, salt = disjoint(prefix_length)
        tail_length = rules.get("suffix_tokens", 1)

        def tail():
            nonce = rng.getrandbits(64)
            return encode(f"{nonce:016x}" * (tail_length + 1))[:tail_length]

        a_tail, c_tail = tail(), tail()
        for _ in range(32):
            if a_tail != c_tail:
                break
            c_tail = tail()
        else:
            raise ValueError("could not tokenize distinct mechanism tails")
        b_tokens, b_salt = disjoint(rules.get("cold_prompt_tokens", 3072))
        warmup = [entry("prime-A", prefix + a_tail, salt, "A", "hot")]
        fill_blocks = (
            rules.get("pressure_blocks", 1025) - 1 - 16 - (len(prefix + a_tail) // block_size)
        )
        usable_prompt = rules["max_model_len"] - rules["output_tokens"]
        if fill_blocks < 0 or max(len(prefix + a_tail), len(b_tokens)) > usable_prompt:
            raise ValueError("mechanism APC priming exceeds the pressure profile")
        while fill_blocks:
            blocks = min(fill_blocks, usable_prompt // block_size)
            tokens, fill_salt = disjoint(blocks * block_size)
            warmup.append(entry(f"prime-filler-{len(warmup):03}", tokens, fill_salt, None, "cold"))
            fill_blocks -= blocks
        requests = [
            entry("B", b_tokens, b_salt, None, "cold", 0),
            entry("C", prefix + c_tail, salt, "A", "hot", 0),
        ]
    else:
        prefix_length = rules.get("prefix_tokens", 2048)
        suffix_length = rules.get("suffix_tokens", 128)
        burst_size = rules.get("burst_size", 20)
        burst_gap_ns = rules.get("burst_gap_ns", 2_000_000)
        if scenario == "burst" and not 0 < burst_gap_ns * (burst_size - 1) < spacing * burst_size:
            raise ValueError("burst group duration must fit in its offered-rate period")
        family_count = rules.get(
            "families", {"burst": 64, "mixed": 64, "ample": 4, "all-cold": 0, "serial": 4}[scenario]
        )
        hot_percent = rules.get(
            "hot_percent",
            {"burst": 80, "mixed": 60, "ample": 60, "all-cold": 0, "serial": 60}[scenario],
        )
        if (
            prefix_length % block_size
            or burst_size < 2
            or not 0 <= hot_percent <= 100
            or family_count < 0
            or (hot_percent and not family_count)
        ):
            raise ValueError("invalid M5 workload parameters")
        families = [disjoint(prefix_length) for _ in range(family_count)]

        def hot(request_id, index, offset_ns=None):
            prefix, salt = families[index]
            # Put a separately tokenized nonce before the suffix text. Prefix identity
            # is by exact token slice, not by re-tokenizing the joined source text.
            suffix = _exact_tokens(encode, rng.getrandbits(64), suffix_length)
            return entry(request_id, prefix + suffix, salt, f"family-{index:03}", "hot", offset_ns)

        prime_count = rules.get(
            "prime_families", family_count if scenario in ("ample", "serial") else 8
        )
        sequence = list(range(family_count))
        rng.shuffle(sequence)
        warmup = [
            hot(f"prime-family-{i:03}", family) for i, family in enumerate(sequence[:prime_count])
        ]
        requests = []
        mixed_hot_index = 0
        jitter_rng = random.Random(seed ^ 0x4D350001)
        jitter_bound_ns = int(spacing / 8)
        for i in range(count):
            if scenario == "burst":
                group, within = divmod(i, burst_size)
                offset_ns = int(group * burst_size * spacing + within * burst_gap_ns)
                is_hot = (within * 100 // burst_size) >= 100 - hot_percent
                hot_index = within - (burst_size - (burst_size * hot_percent // 100))
                # Each burst mixes nearby reusable families and slower rotation.
                family_index = group % min(family_count, 8) if family_count else None
                if family_count > 8 and hot_index >= 3:
                    family_index = 8 + (group * 3 + (hot_index - 3) // 2) % (family_count - 8)
                family = sequence[family_index] if family_index is not None else None
            else:
                offset_ns = int(i * spacing)
                if scenario == "mixed" and 0 < i < count - 1:
                    offset_ns += jitter_rng.randint(-jitter_bound_ns, jitter_bound_ns)
                # Freeze classes before inference, with a consistent quota per 100.
                is_hot = ((i + 1) * hot_percent // 100) > (i * hot_percent // 100)
                # Two successive hot requests share a recent family. The other
                # two advance a longer-distance cycle through all configured families.
                reuse_group, reuse_slot = divmod(mixed_hot_index, 4)
                index = reuse_group if reuse_slot < 2 else (reuse_group * 2 + reuse_slot)
                family = sequence[index % family_count] if family_count else None
                if is_hot:
                    mixed_hot_index += 1
            request_id = f"request-{i:06}"
            if is_hot:
                requests.append(hot(request_id, family, offset_ns))
            else:
                prompt, salt = disjoint(prefix_length + suffix_length)
                requests.append(entry(request_id, prompt, salt, None, "cold", offset_ns))

    output_tokens = rules.get("output_tokens", 16)
    return {
        "schema_version": 1,
        "task_id": config["task_id"],
        "scenario": scenario,
        "seed": seed,
        "rate_rps": None if rate is None else float(rate),
        "interval_ms": None if interval_ms is None else float(interval_ms),
        "mechanism": mechanism if scenario == "mechanism" else None,
        "sampling": {**config["sampling"], "max_tokens": output_tokens},
        "tokenizer_revision": config.get("tokenizer_revision", config["model_revision"]),
        "block_size": block_size,
        "requests": requests,
        "warmup": warmup,
    }
