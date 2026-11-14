"""
Tests for the dual-mode (browser-landmarks vs server-side-image) request
path added to /api/enroll and /api/predict, and for /api/capabilities.

This sandbox's mediapipe install has no Tasks-API model reachable (no
internet -- see backend/app/landmarks.py's module docstring), so these
tests exercise the *error-handling* for that path (a clean 503, not a
crash) rather than real server-side detection end-to-end. On a machine
with internet access, swap in a real base64 JPEG and these same requests
would return real predictions instead of server_side_detection_unavailable.
"""
import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNBRIDGE_DATA_DIR", str(tmp_path))
    from backend.app import main as main_module

    main_module._classifier = None
    main_module._server_extractor = None
    main_module._server_extractor_error = None
    main_module.app.config.update(TESTING=True)
    with main_module.app.test_client() as c:
        yield c


def test_capabilities_endpoint_shape(client):
    res = client.get("/api/capabilities")
    assert res.status_code == 200
    data = res.get_json()
    assert "server_side_detection_importable" in data
    assert "server_side_detection_ready" in data
    assert isinstance(data["server_side_detection_importable"], bool)


def test_predict_requires_hands_or_image(client):
    res = client.post("/api/predict", json={})
    assert res.status_code == 400
    assert "hands" in res.get_json()["error"] or "image" in res.get_json()["error"]


def test_enroll_requires_hands_or_image(client):
    res = client.post("/api/enroll", json={"label": "hello"})
    assert res.status_code == 400


TINY_JPEG_B64 = (
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgG"
    "BgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQ"
    "EBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCAABAAEDASIAAhEBAxEB/8QA"
    "FQABAQAAAAAAAAAAAAAAAAAAAAj/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/"
    "xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
)


def test_predict_with_server_side_image_reports_unavailable_in_this_sandbox(client):
    """In THIS sandbox (no internet, incomplete mediapipe install) the
    server-side path must fail cleanly with a 503 + explanatory error,
    never a 500 / unhandled exception. On a machine with internet access,
    this same request would instead return a real (likely no-hand, since
    the fixture is a blank 1x1 pixel) prediction.
    """
    res = client.post("/api/predict", json={"image": TINY_JPEG_B64})
    data = res.get_json()
    if res.status_code == 503:
        assert "server_side_detection_unavailable" in data["error"]
    else:
        # This environment DOES have working server-side detection --
        # a 1x1 pixel has no hand in it, so this is still the right answer.
        assert res.status_code == 200
        assert data["reason"] == "no_hand_detected"


def test_enroll_with_server_side_image_reports_unavailable_in_this_sandbox(client):
    res = client.post("/api/enroll", json={"label": "hello", "image": TINY_JPEG_B64})
    data = res.get_json()
    assert res.status_code in (503, 422, 200)
    if res.status_code == 503:
        assert "server_side_detection_unavailable" in data["error"]


def test_hands_field_wins_when_both_hands_and_image_sent(client):
    from backend.app.landmarks import NUM_LANDMARKS

    hand = {"label": "Right", "landmarks": [[0.1 * i, 0.2 * i, 0.0] for i in range(NUM_LANDMARKS)]}
    # A malformed image alongside valid hands: if "image" were used, this
    # would error on decoding; "hands" must win so this succeeds instead.
    res = client.post(
        "/api/enroll",
        json={"label": "hello", "hands": [hand], "image": "data:image/jpeg;base64,not-valid"},
    )
    assert res.status_code == 200
    assert res.get_json()["hands_detected"] == 1
