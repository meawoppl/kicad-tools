"""Per-footprint / per-LCSC-part XY and rotation corrections for CPL export.

The XY offset convention is footprint-local (KiCad footprint-editor axes,
+Y down), rotated into the board frame like a pad offset, with local Y
negated on ``B.Cu``.  The oracle below was captured from KiCad's own
``pcbnew`` 10.0.6: a footprint at (100, 100) with a pad at footprint-local
(1, 2), placed at each rotation on each side (bottom via
``FOOTPRINT.Flip(pos, FLIP_DIRECTION_TOP_BOTTOM)`` then
``SetOrientationDegrees``), reporting ``pad.GetPosition()``.  The same
values hold for ``FLIP_DIRECTION_LEFT_RIGHT`` once the orientation is set,
because KiCad stores flipped pads with local Y negated either way.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from kicad_tools.export.pnp import (
    GenericPnPFormatter,
    JLCPCBPnPFormatter,
    PlacementData,
    PnPExportConfig,
    export_pnp,
    extract_placements,
)
from kicad_tools.manufacturers import (
    PlacementCorrection,
    PlacementCorrections,
    get_profile,
    load_placement_corrections,
    load_rotation_corrections,
)
from kicad_tools.manufacturers import base as mfr_base

LOCAL_OFFSET = (1.0, 2.0)

# (layer, rotation) -> pcbnew world position of local (1, 2) at (100, 100).
PCBNEW_ORACLE: dict[tuple[str, float], tuple[float, float]] = {
    ("F.Cu", 0.0): (101.0, 102.0),
    ("F.Cu", 90.0): (102.0, 99.0),
    ("F.Cu", 180.0): (99.0, 98.0),
    ("F.Cu", 270.0): (98.0, 101.0),
    ("B.Cu", 0.0): (101.0, 98.0),
    ("B.Cu", 90.0): (98.0, 99.0),
    ("B.Cu", 180.0): (99.0, 102.0),
    ("B.Cu", 270.0): (102.0, 101.0),
}


def _placement(layer: str = "F.Cu", rotation: float = 0.0, **kw) -> PlacementData:
    fields = {
        "reference": "U1",
        "value": "X",
        "footprint": "Lib:PKG-1",
        "x": 100.0,
        "y": 100.0,
        "rotation": rotation,
        "layer": layer,
    }
    fields.update(kw)
    return PlacementData(**fields)


# ---------------------------------------------------------------------------
# Offset geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("layer", "rotation"), sorted(PCBNEW_ORACLE))
def test_footprint_offset_matches_pcbnew_pad_transform(layer: str, rotation: float):
    corrections = PlacementCorrections(
        footprint={"PKG-*": PlacementCorrection(0.0, *LOCAL_OFFSET)},
    )
    formatter = GenericPnPFormatter(placement_corrections=corrections)
    out = formatter.apply_transforms(_placement(layer, rotation))
    assert out.x == pytest.approx(PCBNEW_ORACLE[(layer, rotation)][0], abs=1e-9)
    assert out.y == pytest.approx(PCBNEW_ORACLE[(layer, rotation)][1], abs=1e-9)
    # The rotation correction is still added to the native angle.
    assert out.rotation == rotation


def test_offset_uses_native_angle_not_corrected_angle():
    """The offset follows the KiCad placement; the rotation fix is JLC-side."""
    corrections = PlacementCorrections(
        footprint={"PKG-*": PlacementCorrection(90.0, *LOCAL_OFFSET)},
    )
    out = GenericPnPFormatter(placement_corrections=corrections).apply_transforms(_placement())
    assert (out.x, out.y) == pytest.approx(PCBNEW_ORACLE[("F.Cu", 0.0)])
    assert out.rotation == 90.0


def test_offset_applied_before_global_offset_and_mirror():
    corrections = PlacementCorrections(
        footprint={"PKG-*": PlacementCorrection(0.0, *LOCAL_OFFSET)},
    )
    config = PnPExportConfig(x_offset=-100.0, y_offset=-100.0, mirror_x=True, mirror_y=True)
    out = GenericPnPFormatter(config, placement_corrections=corrections).apply_transforms(
        _placement()
    )
    assert (out.x, out.y) == pytest.approx((-1.0, -2.0))


def test_no_correction_leaves_placement_untouched():
    out = JLCPCBPnPFormatter(placement_corrections=PlacementCorrections()).apply_transforms(
        _placement(rotation=45.0, lcsc="C1")
    )
    assert (out.x, out.y, out.rotation, out.lcsc) == (100.0, 100.0, 45.0, "C1")


# ---------------------------------------------------------------------------
# Resolution precedence
# ---------------------------------------------------------------------------


@pytest.fixture
def corrections() -> PlacementCorrections:
    return PlacementCorrections(
        footprint={
            "SOT-23*": PlacementCorrection(180.0),
            "SW-*": PlacementCorrection(0.0, 0.5, 0.0),
        },
        lcsc={
            "C51118": PlacementCorrection(270.0),
            "C999": PlacementCorrection(None, 0.0, -1.0),
        },
    )


def test_lcsc_entry_overrides_footprint_glob(corrections):
    assert corrections.resolve("Lib:SOT-23-5", "C51118") == (270.0, 0.0, 0.0)
    assert corrections.resolve("Lib:SOT-23-5", " c51118 ") == (270.0, 0.0, 0.0)
    assert corrections.resolve("Lib:SOT-23-5", "C00000") == (180.0, 0.0, 0.0)
    assert corrections.resolve("Lib:SOT-23-5", "") == (180.0, 0.0, 0.0)


def test_lcsc_entry_without_rotation_inherits_glob_rotation(corrections):
    assert corrections.resolve("SOT-23-6", "C999") == (180.0, 0.0, -1.0)
    assert corrections.resolve("R_0402", "C999") == (0.0, 0.0, -1.0)


def test_lcsc_entry_replaces_glob_offset(corrections):
    assert corrections.resolve("SW-1", "") == (0.0, 0.5, 0.0)
    assert corrections.resolve("SW-1", "C999") == (0.0, 0.0, -1.0)


def test_legacy_rotation_dict_still_applies():
    formatter = JLCPCBPnPFormatter(rotation_corrections={"SOT-23*": 180.0})
    out = formatter.apply_transforms(_placement(footprint="SOT-23-3", rotation=90.0))
    assert out.rotation == 270.0
    assert (out.x, out.y) == (100.0, 100.0)


# ---------------------------------------------------------------------------
# YAML format
# ---------------------------------------------------------------------------


def _write_rotations(tmp_path: Path, monkeypatch, text: str) -> None:
    (tmp_path / "testfab_rotations.yaml").write_text(text)
    monkeypatch.setattr(mfr_base, "_DATA_DIR", tmp_path)


def test_yaml_scalar_and_mapping_entries(tmp_path, monkeypatch):
    _write_rotations(
        tmp_path,
        monkeypatch,
        """
rotation_corrections:
  "SOT-23*": 180
  "SW-*": {offset_x_mm: 0.5}
  "QFN-*": {rotation: 270, offset_y_mm: -0.1}
lcsc_corrections:
  c123: {offset_y_mm: -2.75}
""",
    )
    # Legacy loader: rotations only, mapping entries default to 0.
    assert load_rotation_corrections("testfab") == {
        "SOT-23*": 180.0,
        "SW-*": 0.0,
        "QFN-*": 270.0,
    }
    loaded = load_placement_corrections("testfab")
    assert loaded.footprint["SW-*"] == PlacementCorrection(0.0, 0.5, 0.0)
    assert loaded.footprint["QFN-*"] == PlacementCorrection(270.0, 0.0, -0.1)
    assert loaded.lcsc == {"C123": PlacementCorrection(None, 0.0, -2.75)}


def test_yaml_rejects_unknown_keys(tmp_path, monkeypatch):
    _write_rotations(tmp_path, monkeypatch, 'rotation_corrections:\n  "X*": {rotaton: 90}\n')
    with pytest.raises(ValueError, match="rotaton"):
        load_placement_corrections("testfab")


def test_missing_yaml_returns_empty():
    assert not load_placement_corrections("nonexistent_mfr")


def test_shipped_jlcpcb_lcsc_corrections():
    """Verified JLCPCB preview corrections (2026-09-26) resolve as documented."""
    pc = get_profile("jlcpcb").placement_corrections
    sot5 = "Package_TO_SOT_SMD:SOT-23-5"
    sot6 = "Package_TO_SOT_SMD:SOT-23-6"
    tssop = "Package_SO:TSSOP-20_4.4x6.5mm_P0.65mm"
    # 90 degrees clockwise == 270 CCW-positive.
    assert pc.resolve(tssop, "C113281") == (270.0, 0.0, 0.0)
    assert pc.resolve(sot5, "C51118") == (270.0, 0.0, 0.0)
    assert pc.resolve(sot6, "C7519") == (270.0, 0.0, 0.0)
    # The package-wide rules are unchanged for other parts.
    assert pc.resolve(sot5, "") == (180.0, 0.0, 0.0)
    assert pc.resolve(tssop, "") == (270.0, 0.0, 0.0)
    assert pc.resolve("Lib:JS102011SAQN", "C221660") == (0.0, 0.0, -2.75)
    assert pc.resolve("Lib:USB_C_Receptacle_x", "C165948") == (0.0, 0.0, -1.425)
    # The tier-1 profile shares the data.
    assert get_profile("jlcpcb-tier1").placement_corrections == pc


# ---------------------------------------------------------------------------
# End to end through a parsed board
# ---------------------------------------------------------------------------


def _footprint(ref: str, name: str, lcsc_prop: str, lcsc: str, x: float, y: float, rot: float):
    return f"""  (footprint "{name}"
    (layer "F.Cu")
    (uuid "00000000-0000-0000-0000-00000000000{ref[-1]}")
    (at {x} {y} {rot})
    (property "Reference" "{ref}" (at 0 0 0) (layer "F.SilkS"))
    (property "Value" "V" (at 0 0 0) (layer "F.Fab"))
    (property "{lcsc_prop}" "{lcsc}" (at 0 0 0) (layer "F.Fab") (hide yes))
    (attr smd)
    (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu"))
  )
"""


def test_export_pnp_applies_lcsc_corrections_from_board(tmp_path):
    from kicad_tools.schema.pcb import PCB

    pcb_text = (
        '(kicad_pcb (version 20240108) (generator "pcbnew")\n'
        '  (layers (0 "F.Cu" signal) (31 "B.Cu" signal))\n'
        + _footprint("SW1", "Lib:JS102011SAQN", "LCSC", "C221660", 10, 20, 0)
        + _footprint("J1", "Lib:USB_C_Receptacle_HRO", "LCSC Part", "C165948", 30, 40, 180)
        + _footprint("U1", "Package_TO_SOT_SMD:SOT-23-5", "JLC", "C51118", 50, 60, 90)
        + _footprint("U2", "Package_TO_SOT_SMD:SOT-23-5", "LCSC", "C1", 70, 80, 90)
        + ")\n"
    )
    path = tmp_path / "board.kicad_pcb"
    path.write_text(pcb_text)
    footprints = list(PCB.load(str(path)).footprints)

    placements = {p.reference: p for p in extract_placements(footprints)}
    assert placements["SW1"].lcsc == "C221660"
    assert placements["J1"].lcsc == "C165948"
    assert placements["U1"].lcsc == "C51118"

    profile = get_profile("jlcpcb")
    csv_text = export_pnp(
        footprints,
        "jlcpcb",
        rotation_corrections=profile.rotation_corrections,
        placement_corrections=profile.placement_corrections,
    )
    rows = {r["Designator"]: r for r in csv.DictReader(io.StringIO(csv_text))}

    # Switch at rotation 0: 2.75 mm up the board (-Y in KiCad's Y-down frame).
    assert rows["SW1"]["Mid X"] == "10.0000mm"
    assert rows["SW1"]["Mid Y"] == "17.2500mm"
    assert float(rows["SW1"]["Rotation"]) == 0.0
    # USB-C at rotation 180: local -Y becomes 1.425 mm *down* the board.
    assert rows["J1"]["Mid X"] == "30.0000mm"
    assert rows["J1"]["Mid Y"] == "41.4250mm"
    assert float(rows["J1"]["Rotation"]) == 180.0
    # LCSC-specific SOT-23-5 fix vs. the package-wide SOT-23* rule.
    assert float(rows["U1"]["Rotation"]) == 0.0  # 90 + 270
    assert float(rows["U2"]["Rotation"]) == 270.0  # 90 + 180
