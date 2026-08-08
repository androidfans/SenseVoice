import re
import unicodedata
import uuid


def _normalize_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?，。！？；：、])", r"\1", text)
    text = re.sub(r"([（(【\[])\s+", r"\1", text)
    text = re.sub(r"\s+([）)】\]])", r"\1", text)
    return text


def _is_pure_punctuation(text):
    characters = [char for char in text if not char.isspace()]
    return bool(characters) and all(
        unicodedata.category(char).startswith("P") for char in characters
    )


def _milliseconds(seconds):
    return max(int(round(float(seconds) * 1000)), 0)


def build_huoshan_result(raw_text, token_groups):
    tokens = [token for group in token_groups for token in group]
    aligned_text = _normalize_text("".join(token["text"] for token in tokens))

    if raw_text and not tokens:
        raise ValueError("recognized text has no aligned timestamp tokens")
    if re.sub(r"\s+", "", _normalize_text(raw_text)) != re.sub(
        r"\s+", "", aligned_text
    ):
        raise ValueError("recognized text does not match aligned timestamp tokens")

    utterances = []
    leading_punctuation = []
    for group in token_groups:
        words = []
        for token in group:
            if _is_pure_punctuation(token["text"]):
                if words:
                    words[-1]["end_time"] = _milliseconds(token["end"])
                continue
            word_text = _normalize_text(token["text"])
            if not word_text:
                continue
            words.append(
                {
                    "text": word_text,
                    "start_time": _milliseconds(token["start"]),
                    "end_time": _milliseconds(token["end"]),
                }
            )

        if not words:
            if utterances:
                punctuation = "".join(token["text"] for token in group)
                previous = utterances[-1]
                previous["text"] = _normalize_text(previous["text"] + punctuation)
                previous["end_time"] = _milliseconds(group[-1]["end"])
                previous["words"][-1]["end_time"] = _milliseconds(group[-1]["end"])
            else:
                leading_punctuation.extend(group)
            continue
        utterance_group = leading_punctuation + group
        leading_punctuation = []
        utterances.append(
            {
                "additions": {},
                "start_time": _milliseconds(utterance_group[0]["start"]),
                "end_time": _milliseconds(utterance_group[-1]["end"]),
                "text": _normalize_text(
                    "".join(token["text"] for token in utterance_group)
                ),
                "words": words,
            }
        )

    if leading_punctuation:
        utterances.append(
            {
                "additions": {},
                "start_time": _milliseconds(leading_punctuation[0]["start"]),
                "end_time": _milliseconds(leading_punctuation[-1]["end"]),
                "text": _normalize_text(
                    "".join(token["text"] for token in leading_punctuation)
                ),
                "words": [],
            }
        )

    return {
        "additions": {},
        "code": 1000,
        "id": str(uuid.uuid4()),
        "message": "Success",
        "text": _normalize_text(raw_text),
        "utterances": utterances,
    }
