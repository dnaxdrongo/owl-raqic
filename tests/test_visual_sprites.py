from owl.core.actions import Action
from owl.viz.legend import action_legend_rows
from owl.viz.sprites import SPRITE_SPECS


def test_every_action_has_sprite_spec():
    assert set(SPRITE_SPECS) == set(Action)
    for _action, spec in SPRITE_SPECS.items():
        assert spec.name
        assert spec.glyph
        assert len(spec.color) == 4


def test_legend_rows_cover_actions():
    rows = action_legend_rows()
    assert len(rows) == len(Action)
