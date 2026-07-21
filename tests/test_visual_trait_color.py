from __future__ import annotations

from owl.viz.sprite_state import build_sprite_state
from owl.viz.trait_color import (
    TraitVector,
    encode_trait_hex,
    trait_distance,
    transform_accessibility_color,
    transform_perceptual_color,
)


def test_trait_hex_exact_nibble_encoding() -> None:
    traits = TraitVector(0.0, 1.0, 0.5, 0.25, 0.75, 0.10)
    color = encode_trait_hex(traits)
    assert color.nibbles == (0, 15, 8, 4, 11, 2)
    assert color.raw_hex == "#0F84B2"
    assert color.rgb == (15, 132, 178)


def test_body_identity_is_independent_of_action_and_status() -> None:
    traits = TraitVector(0.2, 0.4, 0.6, 0.8, 0.3, 0.7)
    first = build_sprite_state(action=0, health=1.0, resource=1.0, ow_id=7, traits=traits)
    second = build_sprite_state(action=16, health=0.2, resource=0.1, ow_id=7, traits=traits)
    assert first.descriptor.trait_color.raw_hex == second.descriptor.trait_color.raw_hex
    assert first.descriptor.archetype == second.descriptor.archetype
    assert first.body_color[:3] == second.body_color[:3]


def test_trait_similarity_is_preserved() -> None:
    base = TraitVector(0.2, 0.4, 0.6, 0.8, 0.3, 0.7)
    similar = TraitVector(0.22, 0.39, 0.61, 0.78, 0.31, 0.69)
    distant = TraitVector(0.95, 0.05, 0.08, 0.12, 0.92, 0.10)
    assert trait_distance(base, similar) < trait_distance(base, distant)


def test_accessibility_transform_preserves_raw_identity_code() -> None:
    color = encode_trait_hex(TraitVector(0.1, 0.3, 0.5, 0.7, 0.9, 0.2))
    for mode in ("high_contrast", "deuteranopia", "protanopia", "tritanopia", "monochrome"):
        transformed = transform_accessibility_color(color, mode)
        assert transformed.raw_hex == color.raw_hex
        assert transformed.nibbles == color.nibbles
        assert transformed.rendered_rgb != ()


def test_perceptual_display_preserves_raw_trait_identity() -> None:
    color = encode_trait_hex(TraitVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    transformed = transform_perceptual_color(color)
    assert transformed.raw_hex == color.raw_hex
    assert transformed.nibbles == color.nibbles
    assert transformed.rendered_rgb != color.rgb


def test_offspring_mutation_changes_child_without_changing_parent() -> None:
    parent_traits = TraitVector(0.2, 0.4, 0.6, 0.8, 0.3, 0.7)
    child_traits = TraitVector(0.2, 0.4, 0.6, 0.8, 0.3, 0.95)
    parent_before = encode_trait_hex(parent_traits)
    child = encode_trait_hex(child_traits)
    parent_after = encode_trait_hex(parent_traits)
    assert parent_before.raw_hex == parent_after.raw_hex
    assert child.raw_hex != parent_before.raw_hex
