# Archived Task 2 tests

These files preserve the assertions and fixtures from the earlier Task 2
prototype. Their `archived_test_*.py` names intentionally keep them out of the
current pytest collection. Some fixtures assert that a public pilot is open or
that participants can choose an A/B/C group; both assumptions conflict with the
v6 admission gate while Warehouse remains pending. Running the archived files
directly therefore does not describe v6 compatibility.

Current compatibility checks live in the ordinary `tests/test_*.py` suite,
including `test_study_v6_release_catalog.py`,
`test_study_v6_warehouse_frame_invariants.py`, and
`test_study_v6_pong_kitchen_frames.py`. Those exercise old-record retention,
explicit policy binding, replay, and fail-closed admission under the active
release. Keep the archived tests as source evidence for the historical behavior.
