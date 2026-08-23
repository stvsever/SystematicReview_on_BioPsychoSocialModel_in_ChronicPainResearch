"""Guard the packing of the knowledge-graph payload.

The graph bundle is the only copy of what it shows, so packing it is only
acceptable if it is exactly reversible. Every failure mode here is silent: a
string table that drops an entry, a reference that resolves to the wrong slot,
or a source string shaped like a reference that unpacks into something the
coder never wrote. Each would still produce a graph that opens and looks right.
"""

import json

import pytest

from bps_review.graph.builder import PAYLOAD_FORMAT, pack_payload, unpack_payload


PAYLOAD = {
    "meta": {"title": "A run", "n_nodes": 3},
    "nodes": [
        {
            "id": "n1",
            "label": "central sensitization",
            "type": "item",
            "size": 5.7,
            "level": 4,
            "article_title": "A systematic review of something",
            "provider": "DeepSeek-V4-Flash",
            "detail": {
                "Article": {"Record ID": "F001", "Title": "A systematic review of something"},
                "Recorded value": "elaborated",
                "Quote": "central sensitization amplifies nociceptive input",
            },
        },
        {
            "id": "n2",
            "label": "central sensitization",
            "type": "item",
            "size": 5.7,
            "level": 4,
            "article_title": "A systematic review of something",
            "provider": "DeepSeek-V4-Flash",
            "detail": {
                "Article": {"Record ID": "F002", "Title": "A systematic review of something"},
                "Recorded value": "mentioned",
                "Quote": "the social domain is named but not developed",
            },
        },
        {"id": "n3", "label": "root", "type": "run", "size": 30, "level": 0, "detail": {}},
    ],
    "edges": [{"source": "n3", "target": "n1"}, {"source": "n3", "target": "n2"}],
    "filters": {"providers": ["DeepSeek-V4-Flash"], "types": ["run", "item"]},
}


def test_packing_is_exactly_reversible():
    """The packed file is the only copy of the graph, so this has to hold."""
    restored = unpack_payload(pack_payload(PAYLOAD))
    assert restored == PAYLOAD


def test_repeated_strings_are_written_once():
    packed = pack_payload(PAYLOAD)
    assert packed["meta"]["payload_format"] == PAYLOAD_FORMAT
    table = packed["strings"]
    # Strings used by more than one node are in the table exactly once.
    assert table.count("central sensitization") == 1
    assert "A systematic review of something" in table
    # An object key is a string like any other, and repeats just as hard.
    assert "Record ID" in table
    # A string used once is not worth a reference and stays inline.
    assert "central sensitization amplifies nociceptive input" not in table
    # Neither is a string too short for a reference to be any shorter: "id" is
    # two characters, and so is the "~0" that would stand in for it.
    assert "id" not in table
    assert packed["nodes"][0]["id"] == "n1"
    # A repeated key long enough to be worth it is referenced.
    assert packed["nodes"][0]["~" + str(table.index("label"))] == "~" + str(
        table.index("central sensitization")
    )


def test_packing_actually_shrinks_a_repetitive_payload():
    compact = {"separators": (",", ":"), "ensure_ascii": False}
    before = len(json.dumps(PAYLOAD, **compact))
    after = len(json.dumps(pack_payload(PAYLOAD), **compact))
    assert after < before


def test_numbers_are_left_alone():
    """Only strings are interned, so a reference can never be read as a size."""
    packed = pack_payload(PAYLOAD)
    node = unpack_payload(packed)["nodes"][0]
    assert node["size"] == 5.7
    assert node["level"] == 4
    assert isinstance(node["size"], float)


def test_a_string_shaped_like_a_reference_is_refused():
    """Packing must fail loudly rather than produce a payload it cannot restore."""
    poisoned = json.loads(json.dumps(PAYLOAD))
    poisoned["nodes"][0]["label"] = "~7"
    with pytest.raises(ValueError, match="reference"):
        pack_payload(poisoned)


def test_meta_stays_readable():
    """The header is a dozen fields someone may read straight out of the file."""
    packed = pack_payload(PAYLOAD)
    assert packed["meta"]["title"] == "A run"
    assert packed["meta"]["n_nodes"] == 3
