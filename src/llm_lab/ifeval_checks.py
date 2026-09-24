"""IFEval の 25 命令型の検証器と、日本語プローブ用の検証器。

google-research/instruction_following_eval の判定規則を移植したもの。公式との既知の差:

- 単語・文の数え方が nltk ではなく正規表現（nltk は punkt データの追加取得が要るため）。
  「300 語以上」のような境界付近の判定が数語ずれることがある
- strict / loose の両方を出す。loose は公式と同じ 9 通りの整形（先頭行・末尾行の除去、`*` の除去）
  を試して 1 つでも通れば可とする

したがって**公表値との直接比較には使えない**。同一の採点器で素のモデルと学習後を比べる
回帰検知として使う（判定規則が両者で同一であることが要件で、公式一致は要件ではない）。
"""

from __future__ import annotations

import json
import re
from collections import Counter

# 判定できない命令型があればここに入り、集計時に coverage として報告される
UNSUPPORTED: set[str] = set()

WORD_RE = re.compile(r"\w+", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _sentences(text: str) -> list[str]:
    return [s for s in SENT_SPLIT_RE.split(text) if s.strip()]


def _rel(count: int, relation: str, target: int) -> bool:
    """IFEval の relation は "less than" と "at least" の 2 値。"""
    if relation == "less than":
        return count < target
    if relation == "at least":
        return count >= target
    raise ValueError(f"未知の relation: {relation}")


# --------------------------------------------------------------------------- IFEval


def _keywords_existence(v: str, kw: dict) -> bool:
    return all(re.search(k, v, flags=re.IGNORECASE) for k in kw["keywords"])


def _keywords_frequency(v: str, kw: dict) -> bool:
    n = len(re.findall(kw["keyword"], v, flags=re.IGNORECASE))
    return _rel(n, kw["relation"], kw["frequency"])


def _keywords_forbidden(v: str, kw: dict) -> bool:
    return not any(re.search(r"\b" + re.escape(w) + r"\b", v, flags=re.IGNORECASE) for w in kw["forbidden_words"])


def _letter_frequency(v: str, kw: dict) -> bool:
    n = Counter(v.lower())[kw["letter"].lower()]
    return _rel(n, kw["let_relation"], kw["let_frequency"])


def _response_language(v: str, kw: dict) -> bool | None:
    try:
        from langdetect import DetectorFactory, detect
    except ImportError:
        UNSUPPORTED.add("language:response_language")
        return None
    DetectorFactory.seed = 0  # langdetect は既定で非決定的
    try:
        return detect(v) == kw["language"]
    except Exception:  # noqa: BLE001  空文字など判定不能な応答
        return False


def _number_sentences(v: str, kw: dict) -> bool:
    return _rel(len(_sentences(v)), kw["relation"], kw["num_sentences"])


def _number_words(v: str, kw: dict) -> bool:
    return _rel(len(_words(v)), kw["relation"], kw["num_words"])


def _number_paragraphs(v: str, kw: dict) -> bool:
    # この命令型の段落区切りは "***"（改行ではない）
    paragraphs = re.split(r"\s?\*\*\*\s?", v)
    n = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if not p.strip():
            if i in (0, len(paragraphs) - 1):
                n -= 1
            else:
                return False
    return n == kw["num_paragraphs"]


def _nth_paragraph_first_word(v: str, kw: dict) -> bool:
    paragraphs = re.split(r"\n\n", v)
    n = sum(1 for p in paragraphs if p.strip())
    if kw["nth_paragraph"] > n:
        return False
    para = paragraphs[kw["nth_paragraph"] - 1].strip()
    if not para:
        return False
    word = para.split()[0].strip().lstrip("'").lstrip('"')
    first = ""
    for ch in word:
        if ch in {".", ",", "?", "!", "'", '"'}:
            break
        first += ch.lower()
    return n == kw["num_paragraphs"] and first == kw["first_word"].lower()


def _number_placeholders(v: str, kw: dict) -> bool:
    return len(re.findall(r"\[.*?\]", v)) >= kw["num_placeholders"]


def _postscript(v: str, kw: dict) -> bool:
    marker = kw["postscript_marker"]
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + re.escape(marker.lower()) + r".*$"
    return bool(re.findall(pattern, v.lower(), flags=re.MULTILINE))


def _number_bullets(v: str, kw: dict) -> bool:
    n = len(re.findall(r"^\s*\*[^\*].*$", v, flags=re.MULTILINE)) + len(
        re.findall(r"^\s*-.*$", v, flags=re.MULTILINE)
    )
    return n == kw["num_bullets"]


def _constrained_response(v: str, _kw: dict) -> bool:
    return any(o in v.strip() for o in ("My answer is yes.", "My answer is no.", "My answer is maybe."))


def _highlighted_sections(v: str, kw: dict) -> bool:
    n = sum(1 for h in re.findall(r"\*[^\n\*]*\*", v) if h.strip("*").strip())
    n += sum(1 for h in re.findall(r"\*\*[^\n\*]*\*\*", v) if h.removeprefix("**").removesuffix("**").strip())
    return n >= kw["num_highlights"]


def _multiple_sections(v: str, kw: dict) -> bool:
    pattern = r"\s?" + re.escape(kw["section_spliter"]) + r"\s?\d+\s?"
    return len(re.split(pattern, v)) - 1 >= kw["num_sections"]


def _json_format(v: str, _kw: dict) -> bool:
    s = v.strip().removeprefix("```json").removeprefix("```Json").removeprefix("```JSON").removeprefix("```")
    s = s.removesuffix("```").strip()
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def _title(v: str, _kw: dict) -> bool:
    return any(t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", v))


def _two_responses(v: str, _kw: dict) -> bool:
    parts = v.split("******")
    valid = []
    for i, p in enumerate(parts):
        if not p.strip():
            if i not in (0, len(parts) - 1):
                return False
        else:
            valid.append(p)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def _repeat_prompt(v: str, kw: dict) -> bool:
    return v.strip().lower().startswith(kw["prompt_to_repeat"].strip().lower())


def _end_checker(v: str, kw: dict) -> bool:
    return v.strip().lower().endswith(kw["end_phrase"].strip().lower())


def _quotation(v: str, _kw: dict) -> bool:
    s = v.strip()
    return len(s) > 1 and s[0] == '"' and s[-1] == '"'


def _capital_word_frequency(v: str, kw: dict) -> bool:
    n = sum(1 for w in _words(v) if w.isupper())
    return _rel(n, kw["capital_relation"], kw["capital_frequency"])


IFEVAL_CHECKS = {
    "keywords:existence": _keywords_existence,
    "keywords:frequency": _keywords_frequency,
    "keywords:forbidden_words": _keywords_forbidden,
    "keywords:letter_frequency": _letter_frequency,
    "language:response_language": _response_language,
    "length_constraints:number_sentences": _number_sentences,
    "length_constraints:number_words": _number_words,
    "length_constraints:number_paragraphs": _number_paragraphs,
    "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
    "detectable_content:number_placeholders": _number_placeholders,
    "detectable_content:postscript": _postscript,
    "detectable_format:number_bullet_lists": _number_bullets,
    "detectable_format:constrained_response": _constrained_response,
    "detectable_format:number_highlighted_sections": _highlighted_sections,
    "detectable_format:multiple_sections": _multiple_sections,
    "detectable_format:json_format": _json_format,
    "detectable_format:title": _title,
    "combination:two_responses": _two_responses,
    "combination:repeat_prompt": _repeat_prompt,
    "startend:end_checker": _end_checker,
    "startend:quotation": _quotation,
    "change_case:capital_word_frequency": _capital_word_frequency,
    "change_case:english_capital": lambda v, _kw: v.isupper(),
    "change_case:english_lowercase": lambda v, _kw: v.islower(),
    "punctuation:no_comma": lambda v, _kw: "," not in v,
}


def loose_variants(response: str) -> list[str]:
    """公式 IFEval の loose 判定と同じ 9 通り。前置き行や強調記号で落ちるのを救う。"""
    lines = response.split("\n")
    base = [
        response,
        "\n".join(lines[1:]).strip(),
        "\n".join(lines[:-1]).strip(),
        "\n".join(lines[1:-1]).strip(),
    ]
    return base + [v.replace("*", "") for v in base] + [response.replace("*", "")]


def check_instruction(instruction_id: str, response: str, kwargs: dict, loose: bool = False) -> bool | None:
    """None は「この命令型は判定していない」を意味する（集計では coverage 側に出る）。"""
    fn = IFEVAL_CHECKS.get(instruction_id)
    if fn is None:
        UNSUPPORTED.add(instruction_id)
        return None
    candidates = loose_variants(response) if loose else [response]
    result: bool | None = False
    for cand in candidates:
        try:
            r = fn(cand, kwargs)
        except (KeyError, TypeError, ValueError, AttributeError):
            # kwargs が欠けている等。判定不能として扱い、偽陰性にしない
            UNSUPPORTED.add(instruction_id)
            return None
        if r is None:
            return None
        if r:
            return True
        result = False
    return result


# --------------------------------------------------------------------------- 日本語プローブ
#
# IFEval は英語のみ。学習データは日本語なので、日本語の指示追従が壊れていないかを別に測る。
# 公式ベンチではないので絶対値に意味はなく、素のモデルとの差分だけを見る。

JA_CHECKS = {
    "ja:bullets": lambda v, kw: len(re.findall(r"^\s*[-・*]\s*\S", v, flags=re.MULTILINE)) == kw["num"],
    "ja:sentences": lambda v, kw: len([s for s in re.split(r"[。！？]", v) if s.strip()]) == kw["num"],
    "ja:max_chars": lambda v, kw: len(v.strip()) <= kw["num"],
    "ja:keywords": lambda v, kw: all(k in v for k in kw["keywords"]),
    "ja:forbidden": lambda v, kw: not any(k in v for k in kw["forbidden"]),
    "ja:json": _json_format,
    "ja:end_phrase": lambda v, kw: v.strip().endswith(kw["end_phrase"]),
    "ja:no_touten": lambda v, _kw: "、" not in v and "," not in v,
    "ja:language": lambda v, kw: _response_language(v, {"language": kw["language"]}),
    "ja:numbered": lambda v, kw: len(re.findall(r"^\s*\d+[.．)]\s*\S", v, flags=re.MULTILINE)) == kw["num"],
}

# (prompt, [(instruction_id, kwargs), ...])
JA_PROBES: list[tuple[str, list[tuple[str, dict]]]] = [
    ("コーヒーのドリップ抽出の手順を、箇条書き 3 点で説明してください。各行は「- 」で始めてください。",
     [("ja:bullets", {"num": 3}), ("ja:language", {"language": "ja"})]),
    ("二分探索木とは何か、2 文で説明してください。", [("ja:sentences", {"num": 2}), ("ja:language", {"language": "ja"})]),
    ("日本の首都はどこですか。都市名だけを答えてください。", [("ja:keywords", {"keywords": ["東京"]}), ("ja:max_chars", {"num": 10})]),
    ("光合成を 50 文字以内で説明してください。", [("ja:max_chars", {"num": 60}), ("ja:language", {"language": "ja"})]),
    ("好きな季節について、読点（、）を一切使わずに 1 文で書いてください。", [("ja:no_touten", {}), ("ja:sentences", {"num": 1})]),
    ("次の 3 つの果物を番号付きで列挙してください: りんご、みかん、ぶどう。「1.」のような形式にしてください。",
     [("ja:numbered", {"num": 3}), ("ja:keywords", {"keywords": ["りんご", "みかん", "ぶどう"]})]),
    ('「名前」と「年齢」の 2 つのキーだけを持つ JSON だけを出力してください。他の文は書かないでください。', [("ja:json", {})]),
    ("富士山の標高を答え、最後を必ず「以上です。」で終えてください。",
     [("ja:end_phrase", {"end_phrase": "以上です。"}), ("ja:keywords", {"keywords": ["3776"]})]),
    ("再帰関数について説明してください。ただし「関数」という語を使わずに説明してください。", [("ja:forbidden", {"forbidden": ["関数"]})]),
    ("素数とは何ですか。1 文で答えてください。", [("ja:sentences", {"num": 1}), ("ja:language", {"language": "ja"})]),
    ("Answer in English only: what is the boiling point of water at sea level?",
     [("ja:language", {"language": "en"}), ("ja:keywords", {"keywords": ["100"]})]),
    ("電子レンジの使い方の注意点を箇条書き 4 点で書いてください。各行は「- 」で始めてください。", [("ja:bullets", {"num": 4})]),
    ("「こんにちは」をフランス語でなんと言いますか。単語だけ答えてください。", [("ja:max_chars", {"num": 20})]),
    ("17 かける 23 はいくつですか。数字だけ答えてください。", [("ja:keywords", {"keywords": ["391"]}), ("ja:max_chars", {"num": 10})]),
    ("機械学習の過学習について 3 文で説明してください。", [("ja:sentences", {"num": 3}), ("ja:language", {"language": "ja"})]),
]

# 学習データの出力契約が無関係な質問にも漏れ出していないかを測る印。
# SFT 後に「何を聞いても findings JSON を返す」状態になっていれば、ここが跳ね上がる。
#
# JSON のキーは必ず引用符付きで照合する。素の英語の散文にも "findings" や "severity" は
# 普通に現れるため（素の 14B で 200 問中 3 件が誤検出された）。camelCase の識別子と
# 日本語タグは散文に出ないのでそのまま照合する
BLEED_MARKERS = [
    '"findings"',
    '"severity"',
    '"expected"',
    "ruleId",
    "sourceOrderKeys",
    "probableCauses",
    "appliedRules",
    "【事実】",
    "【仕様】",
    "【推測】",
]


def format_bleed(response: str) -> list[str]:
    return [m for m in BLEED_MARKERS if m in response]
