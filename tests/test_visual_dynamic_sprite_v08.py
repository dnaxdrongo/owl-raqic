from owl.viz.sprite_state import build_sprite_state


def test_invalid_action_gets_unknown_debug_sprite():
    sprite = build_sprite_state(action=99999, health=1, resource=1)
    assert sprite.action_glyph == "unknown"
    assert sprite.debug_marker == "UNKNOWN_ACTION"
    assert sprite.ring_style == "invalid"


def test_dynamic_stress_and_readiness_flags():
    sprite = build_sprite_state(
        action=0,
        health=0.1,
        resource=0.1,
        toxin=0.8,
        reproduction_ready=True,
        selected=True,
    )
    assert sprite.cracked_outline
    assert sprite.hollow_center
    assert sprite.hazard_outline
    assert sprite.reproduction_ready
    assert sprite.selected
