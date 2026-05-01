"""Unit tests for V2XAggregatorState (no ROS plumbing)."""
import pytest
from v2x_aggregator.aggregator_node import V2XAggregatorState


def make_point(x=1.0, y=2.0, z=3.0):
    class _P:
        pass
    p = _P()
    p.x, p.y, p.z = x, y, z
    return p


def make_header(frame_id="d1", sec=0, nanosec=0):
    class _H:
        class _S:
            pass
    h = _H()
    h.frame_id = frame_id
    h.stamp = _H._S()
    h.stamp.sec = sec
    h.stamp.nanosec = nanosec
    return h


def make_point_stamped(frame_id="d1", x=1.0, y=2.0, z=3.0, sec=0, nanosec=0):
    class _PS:
        pass
    ps = _PS()
    ps.header = make_header(frame_id, sec, nanosec)
    ps.point = make_point(x, y, z)
    return ps


def test_update_state_on_incoming():
    state = V2XAggregatorState(stale_timeout_s=1.0)
    msg = make_point_stamped(frame_id="d2")
    state.update("d2", msg, now_s=0.0)
    snap = state.snapshot(now_s=0.1)
    assert len(snap) == 1
    assert snap[0][0] == "d2"


def test_array_includes_all_active_entries():
    state = V2XAggregatorState(stale_timeout_s=1.0)
    state.update("d1", make_point_stamped("d1"), now_s=0.0)
    state.update("d2", make_point_stamped("d2"), now_s=0.0)
    state.update("d3", make_point_stamped("d3"), now_s=0.0)
    snap = state.snapshot(now_s=0.5)
    ids = sorted(item[0] for item in snap)
    assert ids == ["d1", "d2", "d3"]


def test_stale_entry_dropped():
    state = V2XAggregatorState(stale_timeout_s=1.0)
    state.update("d1", make_point_stamped("d1"), now_s=0.0)
    state.update("d2", make_point_stamped("d2"), now_s=0.5)
    snap = state.snapshot(now_s=1.4)
    ids = sorted(item[0] for item in snap)
    assert ids == ["d2"]


def test_vehicle_id_extracted_from_received_header():
    state = V2XAggregatorState(stale_timeout_s=1.0)
    msg = make_point_stamped(frame_id="d4")
    state.update_from_message(msg, now_s=0.0)
    snap = state.snapshot(now_s=0.1)
    assert snap[0][0] == "d4"


def test_position_header_frame_id_overwritten_to_map():
    from v2x_aggregator.aggregator_node import build_vehicle_position_payload
    msg = make_point_stamped(frame_id="d2", x=10.0, y=20.0, z=30.0)
    payload = build_vehicle_position_payload(vehicle_id="d2", source_msg=msg)
    assert payload["vehicle_id"] == "d2"
    assert payload["position_frame_id"] == "map"
    assert payload["x"] == 10.0
    assert payload["y"] == 20.0
    assert payload["z"] == 30.0


def test_header_frame_id_is_map():
    from v2x_aggregator.aggregator_node import ARRAY_HEADER_FRAME_ID
    assert ARRAY_HEADER_FRAME_ID == "map"
