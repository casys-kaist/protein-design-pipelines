from __future__ import annotations

def test_available_components_nonempty():
    import profile.cli.run_sweeps as rs

    components = rs.available_components()
    assert isinstance(components, list)
    assert components, "component registry should not be empty"
