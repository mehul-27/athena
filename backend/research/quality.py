LOW_QUALITY_MARKERS = (
    "insufficient to",
    "content is insufficient",
    "no substantive data",
    "does not contain",
    "not relevant to",
    "no relevant information",
    "unable to extract",
    "completely unrelated",
    "boilerplate",
    "footer text",
    "cookie consent",
    "cookie banner",
    "cookie notice",
    "copyright notice",
    "copyright footer",
    "all rights reserved",
)


def is_low_quality(summary: str) -> bool:
    if not isinstance(summary, str) or not summary:
        return True
    lowered = summary.lower()
    return any(marker in lowered for marker in LOW_QUALITY_MARKERS)
