"""The vocabulary: every column name, category value, site and sheet name the sources use.

The code never writes a source's column name, one of its category values, a
site's name or a worksheet's name. It refers to each by a *key* —
``col("vector_titer")``, ``value("coa_available_yes")``, ``site("a")`` — and a
vocabulary says what the key stands for. The real vocabulary is private: it is
the ``Names`` sheet of a workbook on SharePoint (``sharepoint.vocabulary`` in
config.yaml), next to the ``Corrections`` sheet that holds the known data
fixes. Each row of ``Names`` is one name:

=========  ===================  ==========================================
Kind       Key                  Name
=========  ===================  ==========================================
column     vector_titer         the column's name in the source
column     clump_bags           one row per name, in order, for a list
value      coa_available_yes    a category value the code compares against
site       a                    the first site's name (``b``: the second)
sheet      phf_site_a           the worksheet a source's data is on
=========  ===================  ==========================================

A name may carry a ``{placeholder}`` the code fills in, such as a day number.

The registries below list every key the code uses, with an invented name for
each. They are the demo's vocabulary and the description of each key at once.
A vocabulary that leaves a key out is refused before a stage starts, naming the
key and what it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

#: The workbook sheets of the vocabulary on SharePoint.
NAMES_SHEET = "Names"
CORRECTIONS_SHEET = "Corrections"
KINDS = ("column", "value", "site", "sheet")


class VocabularyError(ValueError):
    """A vocabulary that cannot be used, or none loaded when one is needed."""


COLUMNS: dict[str, str | tuple[str, ...]] = {
    "date_start": "Start date",
    "batch_type": "Batch kind",
    "pt_cd4_pct": "Thawed CD4 share (%)",
    "pt_cd8_pct": "Thawed CD8 share (%)",
    "pt_cd3_pct": "Thawed CD3 share (%)",
    "pt_viable_cells": "Thawed viable cells (cells)",
    "pw_cd4_pct": "Selected CD4 share (%)",
    "pw_cd8_pct": "Selected CD8 share (%)",
    "pw_cd3_pct": "Selected CD3 share (%)",
    "pw_viability": "Selected viability (%)",
    "f_log_pt_cd4_cd8": "Thawed CD4/CD8 log ratio",
    "f_log_pw_cd4_cd8": "Selected CD4/CD8 log ratio",
    "f_cd4_cd8_shift": "CD4/CD8 log ratio shift",
    "f_pt_cd3_cells": "Thawed CD3 cells",
    "f_pt_cd4_share": "Thawed CD3+CD4 share (%)",
    "f_pt_cd4_cells": "Thawed CD4 cells",
    "f_pw_cd4_share": "Selected CD3+CD4 share (%)",
    "f_pw_cd8_share": "Selected CD3+CD8 share (%)",
    "harvest_lactate": "Day 10 lactate (g/L)",
    "harvest_glucose": "Day 10 glucose (g/L)",
    "d6_lactate_a": "Day 6 lactate vessel A (g/L)",
    "d6_lactate_b": "Day 6 lactate vessel B (g/L)",
    "d8_lactate_a": "Day 8 lactate vessel A (g/L)",
    "d8_lactate_b": "Day 8 lactate vessel B (g/L)",
    "d6_glucose_a": "Day 6 glucose vessel A (g/L)",
    "d6_glucose_b": "Day 6 glucose vessel B (g/L)",
    "d8_glucose_a": "Day 8 glucose vessel A (g/L)",
    "d8_glucose_b": "Day 8 glucose vessel B (g/L)",
    "f_harvest_lactate_norm": "Day 10 lactate per seeded cell",
    "f_lactate_per_cell": (
        "D6 lactate per E6 cells",
        "D8 lactate per E6 cells",
        "D10 lactate per E6 cells",
    ),
    "f_glucose_per_cell": (
        "D6 glucose per E6 cells",
        "D8 glucose per E6 cells",
        "D10 glucose per E6 cells",
    ),
    "f_lactate_slope": "Lactate per cell slope D6-D10",
    "f_lactate_auc": "Lactate per cell area D6-D10",
    "f_glucose_slope": "Glucose per cell slope D6-D10",
    "f_glucose_auc": "Glucose per cell area D6-D10",
    "f_lactate_glucose_ratio": "Lactate to glucose ratio D{day}",
    "f_lactate_increment": ("Lactate per cell rise D6-D8", "Lactate per cell rise D8-D10"),
    "f_glucose_increment": ("Glucose per cell change D6-D8", "Glucose per cell change D8-D10"),
    "f_lactate_yield": "Lactate yield on glucose D{day}",
    "cpdl": "Doublings D3-D10",
    "f_growth_profile": "Growth class",
    "f_cd4_profile": "CD4 class",
    "d3_viability": "Day 3 viability (%)",
    "harvest_viability": "Day 10 viability before wash (%)",
    "washed_viability": "Day 10 viability after wash (%)",
    "f_viability_drop_pts": "Viability change D0-D3 (points)",
    "f_viability_d3_d10_rel": "Viability loss D3-D10 (%)",
    "f_viability_d3_d10_pts": "Viability change D3-D10 (points)",
    "f_viability_wash": "Viability change in wash (%)",
    "vessel_count": "Day 3 vessels seeded",
    "vsvg_cells": "Cells for expansion (cells)",
    "cells_available": "Day 3 cells available (cells)",
    "cells_seeded_a": "Day 3 cells seeded vessel A",
    "cells_seeded_b": "Day 3 cells seeded vessel B",
    "culture_bags": "Culture bags",
    "f_seeding_pct": "D3 seeding vs target (%)",
    "f_seeding_error": "D3 seeding error (%)",
    "f_seeding_within": "D3 seeding within tolerance",
    "f_cells_per_bag": "D3 cells per bag",
    "f_seeding_loss": "D3 seeding shortfall",
    "f_dead_cells_d3": "D3 dead cells carried",
    "f_seeded_spread": "Seeded cells {statistic} A-B",
    "f_vector_volume_spread": "Vector volume {statistic} A-B (mL)",
    "vector_titer": "Vector titer (IU/mL)",
    "vector_volume_a": "Vector volume vessel A (mL)",
    "vector_volume_b": "Vector volume vessel B (mL)",
    "f_moi_a": "MOI vessel A",
    "f_log_moi_a": "MOI vessel A (log)",
    "f_moi_b": "MOI vessel B",
    "f_log_moi_b": "MOI vessel B (log)",
    "f_moi_pool": "MOI pooled",
    "f_log_moi_pool": "MOI pooled (log)",
    "f_moi_quality": "MOI pooled, quality adjusted",
    "f_moi_per_cd3": "MOI per CD3 cell",
    "f_moi_vs_target": "MOI against plan",
    "contact_time": "Day 3 contact time (min)",
    "contact_time_a": "Day 3 contact time vessel A (min)",
    "contact_time_b": "Day 3 contact time vessel B (min)",
    "f_contact_deviation": (
        "D3 contact deviation",
        "D3 contact deviation A",
        "D3 contact deviation B",
    ),
    "f_contact_within": (
        "D3 contact within window",
        "D3 contact within window A",
        "D3 contact within window B",
    ),
    "incubator_in_d0": "Day 0 incubator in",
    "incubator_in_d1": "Day 1 incubator in",
    "incubator_out_d1": "Day 1 incubator out",
    "incubator_out_d3": "Day 3 incubator out",
    "incubator_out_d6": "Day 6 incubator out",
    "incubator_out_d8": "Day 8 incubator out",
    "incubator_out_d10": "Day 10 incubator out",
    "incubator_back_d3": "Day 3 incubator back",
    "incubator_back_d6": "Day 6 incubator back",
    "incubator_back_d8": "Day 8 incubator back",
    "f_incubation_d0_d1": "Incubator time days 0-1 (min)",
    "f_incubation_d1_d3": "Incubator time days 1-3 (min)",
    "f_incubation_d3_d6": "Incubator time days 3-6 (min)",
    "f_incubation_d6_d8": "Incubator time days 6-8 (min)",
    "f_incubation_d8_d10": "Incubator time days 8-10 (min)",
    "f_incubation_total": "Incubator time days 3-10 (min)",
    "f_thaw_to_incubator": "Thaw to incubator (min)",
    "f_out_of_incubator": (
        "Outside incubator D3 (min)",
        "Outside incubator D6 (min)",
        "Outside incubator D8 (min)",
    ),
    "clumps_pre_count": "Day 1 clumps before mixing (count)",
    "clumps_pre_size": "Day 1 clumps before mixing (size)",
    "clumps_post_count": "Day 1 clumps after mixing (count)",
    "clumps_post_size": "Day 1 clumps after mixing (size)",
    "clumps_mix_count": "Day 1 clumps after cycles (count)",
    "clumps_mix_size": "Day 1 clumps after cycles (size)",
    "mixing_cycles": "Day 1 mixing cycles",
    "d3_clump_pattern": "Day 3 clump observation",
    "clump_bags": (
        "Day 10 clumps in bag 1",
        "Day 10 clumps in bag 2",
        "Day 10 clumps in bag 3",
        "Day 10 clumps in bag 4",
        "Day 10 clumps in bag 5",
        "Day 10 clumps in bag 6",
    ),
    "f_clump_count_reduction": "Clump count reduction",
    "f_clump_size_reduction": "Clump size reduction",
    "f_clump_burden_pre": "Clump burden before mixing",
    "f_clump_burden_post": "Clump burden after mixing",
    "f_clump_burden_mix": "Clump burden after cycles",
    "f_mixing_x_clumps": "Mixing cycles x clumps",
    "f_d3_clumps": "D3 clumps present",
    "f_clump_bag_count": "D10 bags with clumps",
    "dose_per_kg": "Dose per kg (cells/kg)",
    "dose_cells": "Dose (cells)",
    "formulated_concentration": "Formulated viable concentration (cells/mL)",
    "formulated_viability": "Formulated viability (%)",
    "bag_volume": "Bag volume (mL)",
    "harvest_car_pct": "Day 10 CAR expression (%)",
    "subject_weight": "Patient weight (kg)",
    "washed_cells": "Day 10 cells after wash (cells)",
    "harvest_cells": "Day 10 cells before wash (cells)",
    "washed_concentration": "Day 10 concentration after wash (cells/mL)",
    "formulation_cell_volume": "Cell volume formulated (mL)",
    "formulation_medium_volume": "Medium volume formulated (mL)",
    "fp_car_pct": "Product CAR expression (%)",
    "vcn": "Vector copies per transduced cell",
    "f_dose_all": "Dose per kg, all batches",
    "f_dose_basis": "Dose kind",
    "f_post_formulation_dose": "Dose from formulation",
    "f_achievable_dose": "D10 achievable dose",
    "f_car_cells": "D10 CAR cells",
    "f_car_yield": "CAR yield per seeded cell",
    "f_vcn_per_cell": "Vector copies per cell",
    "f_flow_accuracy": "Product vs harvest CAR",
    "f_formulated_total_concentration": "Formulated total concentration (E6/mL)",
    "f_cells_into_formulation": "Cells into formulation",
    "f_cells_out_of_formulation": "Cells out of formulation",
    "f_cells_lost_formulation": "Cells lost in formulation",
    "f_pct_lost_formulation": "Share lost in formulation (%)",
    "f_cells_lost_wash": "Cells lost in wash",
    "f_wash_recovery": "Wash return share",
    "il2_activity_d6": "Day 6 cytokine activity (IU/mg)",
    "il2_content_d6": "Day 6 cytokine content (ug/vial)",
    "il2_activity_d8": "Day 8 cytokine activity (IU/mg)",
    "il2_content_d8": "Day 8 cytokine content (ug/vial)",
    "f_il2_d6": "D6 cytokine units",
    "f_il2_d8": "D8 cytokine units",
    "thaw_time": "Thaw start",
    "f_shift": "Thaw shift",
    "co2_marker": "CO2 (%)",
    "temperature_marker": "Temperature (C)",
    "coa_prefix": "Vector ",
    "coa_lot": "Certificate lot",
    "coa_lot_number": "Vector lot number",
    "coa_available": "Certificate available",
    "coa_made": "Made on",
    "coa_titer": "Titer",
    "coa_titer_unadjusted": "Titer raw",
    "coa_p24": "Physical titer",
    "coa_pi_ratio": "Particle ratio",
    "coa_car_metric": "Expression",
    "coa_ifng_metric": "Activity",
    "coa_donor": "{metric} donor {donor}",
    "coa_donor_average": "{metric} mean",
    "coa_ph": "Acidity",
    "coa_osmolality": "Osmolarity",
    "coa_volume": "Fill volume",
    "coa_dna_size": "Insert size",
    "coa_impurities": (
        "Host protein",
        "Plasmid DNA",
        "Residual DNA",
        "Residual DNA 1",
        "Residual DNA 2",
    ),
    "coa_calc_prefix": "Delivered ",
    "coa_site_format": "Site and fill",
    "coa_delivered_dose": "Delivered dose per vessel",
    "coa_delivered_dose_fallback": "Delivered dose used adjusted titer",
    "coa_titer_adjustment": "Titer correction factor",
    "coa_specific_infectivity": "Infectivity per particle",
    "coa_last_dna_fraction": "Residual DNA 2 share",
    "f_coa_impurity_index": "Vector impurity index",
    "f_coa_impurity_spec_index": "Vector impurity index vs spec",
    "f_vector_age": "Vector age at use (days)",
    "vector_lot": "Vector lot",
    "patient_id": "Patient code",
    "clinical_site": "Treatment centre",
    "clinical_site_id": "Treatment centre code",
    "country": "Nation",
    "site_map_acronym": "Short name",
    "site_map_institution": "Centre full name",
    "map_site_a_name": "Name at Northfield",
    "map_site_b_name": "Name at Southbank",
    "remfg_batch_keys": ("Tracking number", "Patient Lot/Batch #", "Lot number"),
    "prior_lines_source": "Prior therapies (source)",
    "prior_lines": "Prior therapies",
    "raw_materials_batch": "lotref",
    "raw_materials_batch_keys": ("Patient Lot/Batch #", "Lot", "Lot number", "Tracking number"),
    "endotoxin": "Endotoxin level",
    "pre_activation_clumps_yn": "Day 1 clumps seen?",
    "excluded_mapping_row": "Deviation class (derived)",
}

VALUES: dict[str, str | tuple[str, ...]] = {
    "coa_available_yes": "Yes",
    "endotoxin_unit": " U/mL",
    "patient_id_site_prefix": "NF-",
    "country_to_harmonise": "Southland",
    "country_harmonised": "Republic of Southland",
    "growth_profile_classes": ("slow", "fast", "typical"),
    "cd4_profile_classes": ("CD4 low", "CD4 high", "CD4 mid"),
    "dose_basis_classes": ("by weight", "fixed count"),
    "type_commercial": "Routine",
    "status_aliases": ("recalled|recall", "halted|halt"),
}

SITES: dict[str, str] = {
    "a": "Northfield",
    "b": "Southbank",
}

SHEETS: dict[str, str | int] = {
    "catalog": "Catalog",
    "batch_table": "Sheet1",
    "phf_site_a": "Batch Records",
    "phf_site_b": "Batch Records",
    "param_mapping": "Mapping",
    "remfg": "Treatment Lines",
    "lv_coa": "Certificates",
    "raw_materials": "Materials",
    "investigations": "Investigations",
    "site_maps": "Sheet1",
}


_REGISTRIES = {"column": COLUMNS, "value": VALUES, "site": SITES, "sheet": SHEETS}


@dataclass
class Vocabulary:
    """What every key stands for in one set of sources."""

    columns: dict[str, Any] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)
    sites: dict[str, str] = field(default_factory=dict)
    sheets: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def entries(self, kind: str) -> dict[str, Any]:
        return {
            "column": self.columns,
            "value": self.values,
            "site": self.sites,
            "sheet": self.sheets,
        }[kind]

    def missing(self) -> list[str]:
        """Every key the code needs that this vocabulary leaves out."""
        return [
            f"{kind} {key} ({_describe(registry[key])})"
            for kind, registry in _REGISTRIES.items()
            if kind != "sheet"
            for key in registry
            if key not in self.entries(kind)
        ]

    def names(self) -> list[str]:
        """Every name and value in it, for a leak check."""
        out: list[str] = []
        for kind in KINDS:
            for value in self.entries(kind).values():
                out += [str(v) for v in (value if isinstance(value, tuple) else (value,))]
        return out


def _describe(value: Any) -> str:
    return " / ".join(value) if isinstance(value, tuple) else str(value)


def demo() -> Vocabulary:
    """The invented vocabulary the demo and the tests run on."""
    return Vocabulary(dict(COLUMNS), dict(VALUES), dict(SITES), dict(SHEETS), "the demo")


def from_frame(frame: pd.DataFrame, source: str = "") -> Vocabulary:
    """A vocabulary from its ``Names`` sheet: Kind, Key, Name (and anything else, ignored)."""
    columns = {str(c).strip().lower(): c for c in frame.columns}
    absent = [c for c in ("kind", "key", "name") if c not in columns]
    if absent:
        raise VocabularyError(
            f"{source}: the {NAMES_SHEET} sheet has no {', '.join(absent)} column"
        )
    collected: dict[tuple[str, str], list[Any]] = {}
    errors: list[str] = []
    for number, row in enumerate(frame.itertuples(index=False), start=2):
        record = dict(zip(frame.columns, row))
        kind = str(record[columns["kind"]]).strip().lower()
        key = str(record[columns["key"]]).strip()
        name = record[columns["name"]]
        if not key or key == "nan" or kind == "nan":
            continue
        if kind not in KINDS:
            errors.append(f"row {number}: kind '{kind}' is not one of {', '.join(KINDS)}")
            continue
        if pd.isna(name):
            errors.append(f"row {number}: {kind} {key} has no name")
            continue
        if kind == "sheet" and isinstance(name, float) and name.is_integer():
            name = int(name)
        collected.setdefault((kind, key), []).append(name if kind == "sheet" else str(name))
    if errors:
        raise VocabularyError(f"{source} cannot be used:\n  - " + "\n  - ".join(errors))
    vocabulary = Vocabulary(source=source)
    for (kind, key), names in collected.items():
        listed = isinstance(_REGISTRIES[kind].get(key), tuple)
        vocabulary.entries(kind)[key] = tuple(names) if listed or len(names) > 1 else names[0]
    return vocabulary


def to_frame(vocabulary: Vocabulary) -> pd.DataFrame:
    """The ``Names`` sheet for a vocabulary, with what each key is."""
    rows = []
    for kind in KINDS:
        for key, value in vocabulary.entries(kind).items():
            what = _describe(_REGISTRIES[kind].get(key, ""))
            for name in value if isinstance(value, tuple) else (value,):
                rows.append(
                    {"Kind": kind, "Key": key, "Name": name, "What it is (demo name)": what}
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The vocabulary in use
# ---------------------------------------------------------------------------
_current: Vocabulary | None = None


def use(vocabulary: Vocabulary) -> Vocabulary:
    """Make ``vocabulary`` the one every key is resolved against, after checking it."""
    missing = vocabulary.missing()
    if missing:
        raise VocabularyError(
            f"The vocabulary from {vocabulary.source or 'the workbook'} leaves out "
            f"{len(missing)} names the code needs:\n  - " + "\n  - ".join(missing)
        )
    global _current
    _current = vocabulary
    return vocabulary


def current() -> Vocabulary:
    if _current is None:
        raise VocabularyError(
            "No vocabulary is loaded: a stage reads it from sharepoint.vocabulary first"
        )
    return _current


def _get(kind: str, key: str) -> Any:
    entries = current().entries(kind)
    # A key the workbook adds (one config.yaml names) is as good as a registered one.
    if key not in _REGISTRIES[kind] and key not in entries:
        raise KeyError(f"'{key}' is not a registered {kind} key (see ml.names)")
    if key not in entries:
        raise VocabularyError(f"The vocabulary has no {kind} '{key}'")
    return entries[key]


def col(key: str, **placeholders: Any) -> str:
    """The column a key stands for, with any ``{placeholder}`` filled in."""
    name = _get("column", key)
    if isinstance(name, tuple):
        raise KeyError(f"'{key}' is a list of columns; use cols()")
    return name.format(**placeholders) if placeholders else name


def cols(key: str) -> tuple[str, ...]:
    """The columns a list key stands for, in order."""
    names = _get("column", key)
    return names if isinstance(names, tuple) else (names,)


def value(key: str) -> str:
    """The category value a key stands for."""
    found = _get("value", key)
    if isinstance(found, tuple):
        raise KeyError(f"'{key}' is a list of values; use values()")
    return found


def values(key: str) -> tuple[str, ...]:
    found = _get("value", key)
    return found if isinstance(found, tuple) else (found,)


def site(which: str) -> str:
    """The name of site ``a`` or site ``b``."""
    return _get("site", which)


def sites() -> tuple[str, str]:
    return site("a"), site("b")


def sheet(label: str, default: Any = 0) -> Any:
    """The worksheet a source's data is on, per the vocabulary (else ``default``)."""
    if _current is None:
        return default
    return _current.sheets.get(label, default)
