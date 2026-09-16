"""Deterministic generator for the DS-018 schema-corpus v2 expansion (RFC-004 track).

Builds 300 NEW synthetic samples of bare, pretty-printed JSON text (no
markdown, no code fences, no commentary) that continue the DS-002 schema
series (scripts/build_schema_corpus.py, SEED=4004, ids 0-99).  v2 uses
SEED=4006 and ids 100-399, extends the depth envelope to 1..8, and embeds
five NEW hazards INSIDE the valid JSON:

1. unicode_keys             -- object keys drawn from an accented/CJK pool.
2. number_heavy             -- the wide leaf (and decorations) are dominated
                               by numeric scalars.
3. deep_arrays_of_objects   -- the container spine alternates
                               array -> object -> array -> object ...
4. empty_containers         -- `{}` / `[]` appear as values/elements.
5. single_key_objects       -- one-key objects appear as values/elements.

The corpus is a certified fixture so that suppression / variety behavior on
structured generation can be MEASURED rather than pattern-guessed, exactly as
in the DS-002 corpus.

Mirrors the structure and discipline of scripts/build_schema_corpus.py: a
single fixed seed, an explicit random.Random instance, structural self-checks
at generation time, and a byte-identical fixture across runs.  The vocabulary
still avoids Track B trigger words and "//"; the production
is_code_syntax_context regex must return False on all 300 samples.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import List, Union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))  # parents[2] header; import AVG.* via /AVG

from AVG.core.metrics import is_code_syntax_context  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed seed provenance:
# SEED = 4006 continues the RFC-004 schema track ("4.004" schema corpus used
# SEED 4004; the DS-007 hazard siblings used 4005).  4006 is the v2 expansion
# track.  A single fixed seed and an explicit random.Random instance guarantee
# the corpus is byte-identical across runs and reviewers.
# ---------------------------------------------------------------------------
SEED = 4006
NUM_SAMPLES = 300
MIN_LINES = 120
MAX_DEPTH = 8
LABEL = "schema"
ID_BASE = 100  # v2 ids are 100..399, continuing the v1 series 0..99

# Track B trigger words (production regex in AVG.core.metrics): the vocabulary
# must not contain any of these as whole words, and no URL "//" may appear.
TRACK_B_TRIGGERS = frozenset({"let", "const", "var", "def", "struct", "typedef"})

# Closed English vocabulary: single words used both as JSON keys and as the
# word pool for string values.  Identical to scripts/build_schema_corpus.py.
# No Track B trigger words, no slashes, no code-syntax markers.
VOCABULARY: List[str] = [
    "name", "value", "item", "count", "total", "size", "length", "width",
    "height", "depth", "weight", "price", "cost", "rate", "score", "level",
    "rank", "index", "role", "title", "label", "status", "state", "order",
    "number", "amount", "sum", "mean", "range", "limit", "edge", "top",
    "bottom", "left", "right", "front", "back", "middle", "center", "side",
    "corner", "base", "head", "tail", "start", "end", "open", "close",
    "main", "minor", "major", "prime", "key", "lock", "door", "window",
    "room", "floor", "wall", "roof", "garden", "path", "road", "street",
    "lane", "bridge", "river", "lake", "ocean", "sea", "hill", "valley",
    "field", "farm", "town", "village", "city", "country", "world", "map",
    "plan", "idea", "thought", "word", "sign", "symbol", "mark", "note",
    "book", "page", "line", "text", "story", "song", "dance", "game",
    "play", "work", "task", "job", "duty", "rule", "law", "peace", "war",
    "love", "hate", "joy", "fear", "hope", "dream", "wish", "want", "need",
    "get", "take", "give", "make", "go", "come", "see", "look", "hear",
    "speak", "talk", "say", "tell", "ask", "answer", "call", "find",
    "keep", "hold", "put", "move", "turn", "run", "walk", "jump", "sit",
    "stand", "lie", "fall", "rise", "grow", "build", "break", "cut",
    "push", "pull", "carry", "bring", "send", "receive", "buy", "sell",
    "pay", "spend", "save", "lose", "win", "teach", "learn", "know",
    "think", "believe", "feel", "understand", "remember", "forget",
    "choose", "decide", "help", "try", "stop", "continue", "color",
    "shape", "material", "quality", "degree", "point", "part", "place",
    "person", "thing", "time", "day", "week", "month", "year", "hour",
    "minute", "second", "morning", "evening", "night", "spring", "summer",
    "autumn", "winter", "family", "friend", "child", "parent", "house",
    "home", "building", "office", "school", "store", "market", "block",
    "zone", "area", "region", "section", "quarter", "half", "third",
    "pair", "group", "team", "crowd", "series", "chain", "batch", "pack",
    "stack", "pile", "row", "column", "table", "chair", "desk", "shelf",
    "box", "bag", "bottle", "cup", "plate", "bowl", "fork", "spoon",
    "knife", "tool", "machine", "engine", "motor", "wheel", "gear",
    "lever", "switch", "button", "knob", "handle", "hook", "nail",
    "screw", "bolt", "nut", "wrench", "hammer", "saw", "drill", "ladder",
    "rope", "cord", "wire",
]

# Unicode key pool: accented Latin and CJK keys used by the "unicode_keys"
# hazard.  None of these contain Track B trigger words or "//".  The pool is
# large enough that even a sample that draws every key from this pool (the
# worst case) never exhausts it.
UNICODE_KEYS: List[str] = [
    # accented Latin
    "café", "naïve", "résumé", "jalapeño", "señor", "señora", "piñata",
    "façade", "décor", "crème", "français", "über", "münchen", "tête",
    "père", "mère", "hôtel", "forêt", "noël", "œuf", "château", "garçon",
    "mañana", "niño", "niña", "año", "día", "fiancé", "fiancée", "blasé",
    "protégé", "exposé", "séance", "rôle", "côté", "déjà", "voilà", "âme",
    "île", "hôte", "paella", "empanada", "tortilla", "olé", "pièce", "crû",
    # CJK (simplified Chinese)
    "名字", "价值", "项目", "数量", "总数", "大小", "长度", "宽度",
    "高度", "深度", "重量", "价格", "成本", "比率", "分数", "等级",
    "排名", "索引", "角色", "标题", "标签", "状态", "顺序", "数字",
    "金额", "总和", "均值", "范围", "极限", "边缘", "顶部", "底部",
    "左边", "右边", "前面", "后面", "中间", "中心", "侧面", "角落",
    "基础", "头部", "尾部", "开始", "结束", "打开", "关闭", "主要",
    "次要", "关键", "锁", "门", "窗户", "房间", "地板", "墙壁",
    "屋顶", "花园", "道路", "街道", "巷子", "桥", "河流", "湖泊",
    "海洋", "大海", "山丘", "山谷", "田地", "农场", "城镇", "村庄",
    "城市", "国家", "世界", "地图", "计划", "想法", "思想", "单词",
    "标志", "符号", "标记", "笔记", "书籍", "页面", "行", "文本",
    "故事", "歌曲", "舞蹈", "游戏", "播放", "工作", "任务", "职责",
    "规则", "法律", "和平", "战争", "爱", "恨", "快乐", "恐惧",
    "希望", "梦想", "愿望", "需要", "得到", "拿", "给予", "制作",
    "去", "来", "看见", "看", "听", "说话", "说", "告诉", "问",
    "回答", "呼叫", "找到", "保持", "持有", "放置", "移动", "转动",
    "跑", "走", "跳", "坐", "站", "躺", "掉落", "上升", "生长",
    "建造", "打破", "切割", "推", "拉", "携带", "带来", "发送",
    "接收", "买", "卖", "支付", "花费", "保存", "失去", "赢得",
    "教", "学习", "知道", "思考", "相信", "感觉", "理解", "记住",
    "忘记", "选择", "决定", "帮助", "尝试", "停止", "继续", "颜色",
    "形状", "材料", "质量", "程度", "点", "部分", "地方", "人",
    "东西", "时间", "天", "周", "月", "年", "小时", "分钟", "秒",
    "早上", "晚上", "夜晚", "春天", "夏天", "秋天", "冬天", "家庭",
    "朋友", "孩子", "父母", "房子", "家", "建筑", "办公室", "学校",
    "商店", "市场", "街区", "区域", "地区", "部分", "季度", "一半",
    "第三", "对", "组", "团队", "人群", "系列", "链条", "批次",
    "包", "堆", "排", "列", "桌子", "椅子", "书桌", "架子", "盒子",
    "袋子", "瓶子", "杯子", "盘子", "碗", "叉子", "勺子", "刀子",
    "工具", "机器", "引擎", "马达", "轮子", "齿轮", "杠杆", "开关",
    "按钮", "把手", "手柄", "钩子", "钉子", "螺丝", "螺栓", "螺母",
    "扳手", "锤子", "锯", "钻", "梯子", "绳子", "绳索", "电线",
]

# Hazard families embedded inside the valid JSON payloads.
HAZARD_UNICODE_KEYS = "unicode_keys"
HAZARD_NUMBER_HEAVY = "number_heavy"
HAZARD_DEEP_AO = "deep_arrays_of_objects"
HAZARD_EMPTY_CONTAINERS = "empty_containers"
HAZARD_SINGLE_KEY_OBJECTS = "single_key_objects"
HAZARDS: List[str] = [
    HAZARD_UNICODE_KEYS,
    HAZARD_NUMBER_HEAVY,
    HAZARD_DEEP_AO,
    HAZARD_EMPTY_CONTAINERS,
    HAZARD_SINGLE_KEY_OBJECTS,
]

JsonValue = Union[dict, list, str, int, float, bool, None]


def _lines_of_value(v: JsonValue) -> int:
    """Number of pretty-printed (indent 2) lines a value occupies.

    Mirrors scripts/build_schema_corpus.py but accounts for empty containers:
    json.dumps renders `{}` and `[]` on a single line.
    """
    if isinstance(v, dict):
        if not v:
            return 1
        return 1 + sum(_lines_of_value(x) for x in v.values()) + 1
    if isinstance(v, list):
        if not v:
            return 1
        return 1 + sum(_lines_of_value(x) for x in v) + 1
    return 1


def _max_depth(v: JsonValue) -> int:
    """Maximum container nesting depth; scalars are depth 0."""
    if isinstance(v, dict):
        return 1 + max((_max_depth(x) for x in v.values()), default=0)
    if isinstance(v, list):
        return 1 + max((_max_depth(x) for x in v), default=0)
    return 0


def _scalar(rng: random.Random, number_heavy: bool = False) -> JsonValue:
    """Draw a JSON scalar; number_heavy biases strongly toward numbers."""
    if number_heavy and rng.random() < 0.90:
        if rng.random() < 0.7:
            return rng.randint(0, 1000)
        return round(rng.uniform(0.0, 1000.0), 2)
    kind = rng.choice(["str", "str", "str", "int", "float", "bool", "null"])
    if kind == "str":
        n_words = rng.randint(1, 4)
        return " ".join(rng.choice(VOCABULARY) for _ in range(n_words))
    if kind == "int":
        return rng.randint(0, 1000)
    if kind == "float":
        return round(rng.uniform(0.0, 1000.0), 2)
    if kind == "bool":
        return rng.choice([True, False])
    return None


def _count_values(v: JsonValue) -> tuple:
    """Return (total_scalars, numeric_scalars) in ``v`` (bools excluded)."""
    if isinstance(v, dict):
        t = n = 0
        for x in v.values():
            tt, nn = _count_values(x)
            t += tt
            n += nn
        return t, n
    if isinstance(v, list):
        t = n = 0
        for x in v:
            tt, nn = _count_values(x)
            t += tt
            n += nn
        return t, n
    return 1, (1 if isinstance(v, (int, float)) and not isinstance(v, bool) else 0)


def _contains_empty_container(v: JsonValue) -> bool:
    """True if any dict or list in ``v`` is empty."""
    if isinstance(v, dict):
        if not v:
            return True
        return any(_contains_empty_container(x) for x in v.values())
    if isinstance(v, list):
        if not v:
            return True
        return any(_contains_empty_container(x) for x in v)
    return False


def _contains_single_key_object(v: JsonValue) -> bool:
    """True if any object in ``v`` has exactly one key."""
    if isinstance(v, dict):
        if len(v) == 1:
            return True
        return any(_contains_single_key_object(x) for x in v.values())
    if isinstance(v, list):
        return any(_contains_single_key_object(x) for x in v)
    return False


def _has_array_of_objects(v: JsonValue) -> bool:
    """True if any list in ``v`` has at least one object element."""
    if isinstance(v, list):
        if any(isinstance(x, dict) for x in v):
            return True
        return any(_has_array_of_objects(x) for x in v)
    if isinstance(v, dict):
        return any(_has_array_of_objects(x) for x in v.values())
    return False


def _hazards_for(idx: int) -> List[str]:
    """Deterministic per-sample hazard assignment (depth-aware).

    Depth-1 samples cannot host container-valued hazards (empty containers,
    single-key objects, arrays-of-objects) without exceeding the recorded
    depth, so they carry only unicode_keys or number_heavy.  Deeper samples
    cycle through all five hazard families so every hazard is represented at
    every depth >= 2.
    """
    depth = (idx % MAX_DEPTH) + 1
    if depth == 1:
        return (
            [HAZARD_UNICODE_KEYS] if (idx // MAX_DEPTH) % 2 == 0 else [HAZARD_NUMBER_HEAVY]
        )
    return [HAZARDS[(idx // MAX_DEPTH) % len(HAZARDS)]]


def _build_tree(
    depth: int,
    rng: random.Random,
    top_is_object: bool,
    hazards: List[str],
) -> JsonValue:
    """Build a JSON value whose max container depth is exactly ``depth`` and
    which manifests every hazard in ``hazards``.

    Structure: a spine of ``depth`` nested containers.  Each non-leaf level
    carries a small number of scalar/container "decoration" entries plus the
    spine child; the leaf is a wide container of scalar entries sized so the
    pretty-printed text has >= MIN_LINES non-empty lines.
    """
    ascii_keys = list(VOCABULARY)
    rng.shuffle(ascii_keys)
    akeys = iter(ascii_keys)

    uni_keys = list(UNICODE_KEYS)
    rng.shuffle(uni_keys)
    ukeys = iter(uni_keys)

    has_unicode = HAZARD_UNICODE_KEYS in hazards
    is_number_heavy = HAZARD_NUMBER_HEAVY in hazards
    is_ao = HAZARD_DEEP_AO in hazards
    has_empty = HAZARD_EMPTY_CONTAINERS in hazards
    has_single = HAZARD_SINGLE_KEY_OBJECTS in hazards

    # Spine container types: True = object, False = list.
    spine: List[bool] = []
    if is_ao:
        # Deeply nested arrays-of-objects: the spine alternates and starts
        # with an array so the outermost container is always an array.
        cur = False
        for _ in range(depth):
            spine.append(cur)
            cur = not cur
    else:
        cur = top_is_object
        for _ in range(depth):
            spine.append(cur)
            cur = rng.choice([True, False])

    used_unicode = False

    def _next_key() -> str:
        nonlocal used_unicode
        if has_unicode and (not used_unicode or rng.random() < 0.30):
            used_unicode = True
            return next(ukeys)
        return next(akeys)

    def _decoration_value() -> JsonValue:
        if has_empty and rng.random() < 0.25:
            return {} if rng.random() < 0.5 else []
        if has_single and rng.random() < 0.25:
            return {_next_key(): _scalar(rng, is_number_heavy)}
        return _scalar(rng, is_number_heavy)

    def _build(level: int, is_object: bool):
        """Return (node, leaf) where leaf is the deepest container."""
        if level == depth - 1:
            leaf: JsonValue = {} if is_object else []
            return leaf, leaf

        n_dec = rng.randint(0, 3)
        child_is_object = spine[level + 1]
        child, child_leaf = _build(level + 1, child_is_object)

        dec_values: List[JsonValue] = []
        # Force hazard manifestation at the first non-leaf level.
        if level == 0 and has_empty:
            dec_values.append({} if rng.random() < 0.5 else [])
        if level == 0 and has_single:
            dec_values.append({_next_key(): _scalar(rng, is_number_heavy)})
        for _ in range(n_dec):
            dec_values.append(_decoration_value())

        if is_object:
            obj: dict = {}
            for v in dec_values:
                obj[_next_key()] = v
            obj[_next_key()] = child
            return obj, child_leaf

        arr: list = list(dec_values)
        arr.append(child)
        return arr, child_leaf

    tree, leaf = _build(0, spine[0])

    # Pad the leaf with scalar entries until the pretty-printed text has at
    # least MIN_LINES non-empty lines (deterministic rng consumption).
    target_lines = MIN_LINES + rng.randint(0, 10)
    while _lines_of_value(tree) < target_lines:
        if isinstance(leaf, dict):
            leaf[_next_key()] = _scalar(rng, is_number_heavy)
        else:
            leaf.append(_scalar(rng, is_number_heavy))

    return tree


def _generate_corpus(rng: random.Random) -> List[dict]:
    """Generate the 300-sample DS-018 schema-corpus v2 expansion.

    Depths cycle 1..8 so every depth is represented (variety engagement).
    Top-level objects and arrays are mixed; unicode_keys samples force an
    object root (keys must exist) and arrays-of-objects samples force an
    array root (the hazard's defining container).
    """
    samples: List[dict] = []
    for idx in range(NUM_SAMPLES):
        depth = (idx % MAX_DEPTH) + 1
        hazards = _hazards_for(idx)
        top_is_object = (idx % 2 == 0)
        if HAZARD_UNICODE_KEYS in hazards:
            top_is_object = True
        if HAZARD_DEEP_AO in hazards:
            top_is_object = False
        tree = _build_tree(depth, rng, top_is_object, hazards)
        text = json.dumps(tree, indent=2, ensure_ascii=False)
        line_count = sum(1 for ln in text.splitlines() if ln.strip())
        samples.append(
            {
                "id": ID_BASE + idx,
                "seed": SEED,
                "depth": depth,
                "line_count": line_count,
                "hazards": hazards,
                "text": text,
                "label": LABEL,
            }
        )
    return samples


def _validate(samples: List[dict]) -> None:
    """Structural self-checks; raise AssertionError on any failure."""
    # Vocabulary hygiene: no Track B trigger word as a whole word.
    vocab_words = set(VOCABULARY)
    assert not (vocab_words & TRACK_B_TRIGGERS), (
        f"vocabulary contains Track B trigger(s): {vocab_words & TRACK_B_TRIGGERS}"
    )
    uni_words = set(UNICODE_KEYS)
    assert not (uni_words & TRACK_B_TRIGGERS), (
        f"unicode vocabulary contains Track B trigger(s): {uni_words & TRACK_B_TRIGGERS}"
    )

    assert len(samples) == NUM_SAMPLES, (
        f"expected {NUM_SAMPLES} samples, got {len(samples)}"
    )

    depths: set = set()
    top_levels: set = set()
    for sample in samples:
        sid = sample["id"]
        assert sample["label"] == LABEL, f"sample {sid} label is not '{LABEL}'"
        assert sample["seed"] == SEED, f"sample {sid} seed is not {SEED}"
        assert isinstance(sample["depth"], int) and 1 <= sample["depth"] <= MAX_DEPTH, (
            f"sample {sid} depth {sample['depth']} outside 1..{MAX_DEPTH}"
        )
        assert ID_BASE <= sid < ID_BASE + NUM_SAMPLES, (
            f"sample {sid} outside id range {ID_BASE}..{ID_BASE + NUM_SAMPLES - 1}"
        )
        assert isinstance(sample["line_count"], int) and sample["line_count"] >= MIN_LINES, (
            f"sample {sid} has {sample['line_count']} non-empty lines < {MIN_LINES}"
        )

        text = sample["text"]
        # Self-check 1: text parses as bare JSON.
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - diagnostic
            raise AssertionError(f"sample {sid} text is not valid JSON: {exc}") from exc

        # Self-check 2: non-empty line count matches record and is >= 120.
        non_empty = [ln for ln in text.splitlines() if ln.strip()]
        assert len(non_empty) == sample["line_count"], (
            f"sample {sid} recorded line_count {sample['line_count']} != actual {len(non_empty)}"
        )
        assert len(non_empty) >= MIN_LINES, (
            f"sample {sid} has {len(non_empty)} non-empty lines < {MIN_LINES}"
        )
        # Recorded depth must equal the parsed JSON's actual max depth.
        assert _max_depth(parsed) == sample["depth"], (
            f"sample {sid} recorded depth {sample['depth']} != actual {_max_depth(parsed)}"
        )
        # Analytic pretty-printed line count must match the actual dump.
        assert _lines_of_value(parsed) == len(non_empty), (
            f"sample {sid} analytic lines {_lines_of_value(parsed)} != actual {len(non_empty)}"
        )

        # Self-check 3: production Track B regex must not fire on any sample.
        assert not is_code_syntax_context(text), (
            f"sample {sid} triggered Track B code-syntax regex"
        )

        # Self-check 4: every recorded hazard is manifested inside the JSON.
        hazards = sample["hazards"]
        assert isinstance(hazards, list) and hazards, f"sample {sid} has no hazards"
        for hazard in hazards:
            assert hazard in HAZARDS, f"sample {sid} unknown hazard {hazard!r}"
        if HAZARD_UNICODE_KEYS in hazards:
            assert any(ord(c) > 127 for c in text), (
                f"sample {sid} hazard unicode_keys not manifested (no non-ASCII char)"
            )
        if HAZARD_NUMBER_HEAVY in hazards:
            total, num = _count_values(parsed)
            assert num / total > 0.5, (
                f"sample {sid} hazard number_heavy not dominant ({num}/{total})"
            )
        if HAZARD_DEEP_AO in hazards:
            assert _has_array_of_objects(parsed), (
                f"sample {sid} hazard deep_arrays_of_objects not manifested"
            )
        if HAZARD_EMPTY_CONTAINERS in hazards:
            assert _contains_empty_container(parsed), (
                f"sample {sid} hazard empty_containers not manifested"
            )
        if HAZARD_SINGLE_KEY_OBJECTS in hazards:
            assert _contains_single_key_object(parsed), (
                f"sample {sid} hazard single_key_objects not manifested"
            )

        depths.add(sample["depth"])
        top_levels.add(type(parsed))

    # Variety engagement: every depth 1..8 and both top-level kinds appear.
    assert depths == set(range(1, MAX_DEPTH + 1)), (
        f"depth coverage incomplete: {sorted(depths)}"
    )
    assert top_levels == {dict, list}, (
        f"top-level kind coverage incomplete: {top_levels}"
    )


def _report(samples: List[dict]) -> None:
    """Print the corpus summary required by the DS-018 task file."""
    depth_counts: dict = {}
    hazard_counts: dict = {}
    for s in samples:
        depth_counts[s["depth"]] = depth_counts.get(s["depth"], 0) + 1
        for h in s["hazards"]:
            hazard_counts[h] = hazard_counts.get(h, 0) + 1
    print(f"seed provenance: SEED={SEED}, explicit random.Random, "
          f"{len(samples)} samples, ids {ID_BASE}..{ID_BASE + len(samples) - 1}")
    print("depth distribution: " + ", ".join(
        f"d{d}={depth_counts.get(d, 0)}" for d in range(1, MAX_DEPTH + 1)
    ))
    print("hazard counts: " + ", ".join(f"{h}={hazard_counts.get(h, 0)}" for h in HAZARDS))


def build(output_path: str) -> List[dict]:
    """Build, validate, and write the corpus to ``output_path``."""
    rng = random.Random(SEED)
    samples = _generate_corpus(rng)
    _validate(samples)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    _report(samples)
    print(f"Wrote {NUM_SAMPLES} validated samples to {output_path}")
    print(f"fixture sha256: {digest}")
    return samples


def main() -> int:
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = str(ROOT / "tests" / "fixtures" / "schema_corpus_v2.jsonl")
    build(output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
