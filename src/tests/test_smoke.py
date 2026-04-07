from bps_review.settings import ROOT, protocol_config


def test_root_exists() -> None:
    assert ROOT.exists()


def test_protocol_title() -> None:
    protocol = protocol_config()
    title = protocol.get("review", {}).get("title") or protocol.get("project", {}).get("title", "")
    assert "Biopsychosocial" in title


def test_operational_window_extends_to_march_2026() -> None:
    protocol = protocol_config()
    assert protocol["search"]["operational_date_window"]["end"] == "2026-03-31"
