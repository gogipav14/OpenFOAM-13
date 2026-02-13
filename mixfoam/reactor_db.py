"""
Parse MixIT Reactors .mdata archive and build a searchable reactor catalog.

Archive structure:
  MixIT Reactors.mdata (zip)
    ├── Binder.mdata (zip)
    │   ├── Binder.xml          (tank/impeller/process specs)
    │   ├── Binder.milib (zip)  (impeller STLs)
    │   └── BaffleSTLs/*.stl    (baffle geometry)
    └── ...
"""

import zipfile
import io
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class TankGeometry:
    type: str  # "Cylindrical" or "Rectangular"
    diameter: float  # m (for cylindrical: diameter, for rectangular: one side)
    length: float  # m (for rectangular)
    width: float  # m (for rectangular, or same as diameter for cylindrical)
    straight_side: float  # m (straight wall height)
    bottom_style: str  # "Flat", "2:1Elliptical", "6%Torispherical", "10%Torispherical", "Conical"
    bottom_depth: float  # m
    head_style: str
    head_depth: float  # m


@dataclass
class ImpellerSpec:
    type: str  # impeller type name (maps to STL filename)
    mounting: str  # "Top" or "Bottom"
    diameter: float  # m
    clearance: float  # m (from bottom for bottom-mounted, from top for top-mounted)
    off_center: float  # m (X offset from center)
    off_center_y: float  # m (Y offset from center)
    number_of_blades: int
    blade_width: float  # m
    angle1: float  # degrees
    angle2: float  # degrees
    angle3: float  # degrees
    angle4: float  # degrees


@dataclass
class BaffleSpec:
    style: str
    width: float  # m
    off_wall: float  # m
    off_bottom: float  # m
    off_top: float  # m
    stl_filename: Optional[str] = None


@dataclass
class OperatingCondition:
    temperature: float  # K
    volume_m3: float  # m³
    liquid_level: float  # m
    rpm: float
    direction: str = "Clockwise"


@dataclass
class ReactorConfig:
    reactor_id: str
    config_name: str
    process_name: str
    scale: str
    tank: TankGeometry
    impeller: ImpellerSpec
    baffles: List[BaffleSpec]
    operating: OperatingCondition
    motor_power: float  # kW
    available_impeller_stls: List[str] = field(default_factory=list)
    available_baffle_stls: List[str] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return f"{self.reactor_id}/{self.config_name}"

    @property
    def volume_liters(self) -> float:
        return self.operating.volume_m3 * 1000


@dataclass
class ReactorFamily:
    name: str
    configs: List[ReactorConfig]
    impeller_stl_names: List[str]  # all STLs available in .milib
    baffle_stl_names: List[str]  # all STLs available in BaffleSTLs/


def _parse_tank(tank_elem) -> TankGeometry:
    def val(tag, default="0"):
        el = tank_elem.find(tag)
        return el.get("value", default) if el is not None else default

    return TankGeometry(
        type=val("type", "Cylindrical"),
        diameter=float(val("diameter")),
        length=float(val("length", val("diameter"))),
        width=float(val("width", val("diameter"))),
        straight_side=float(val("straightSide")),
        bottom_style=val("bottomStyle", "Flat"),
        bottom_depth=float(val("bottomDepth")),
        head_style=val("headStyle", "Flat"),
        head_depth=float(val("headDepth")),
    )


def _parse_impeller(imp_elem) -> ImpellerSpec:
    def val(tag, default="0"):
        el = imp_elem.find(tag)
        return el.get("value", default) if el is not None else default

    return ImpellerSpec(
        type=val("type"),
        mounting=val("mounting", "Bottom"),
        diameter=float(val("diameter")),
        clearance=float(val("clearance")),
        off_center=float(val("offCenter")),
        off_center_y=float(val("offCenterY")),
        number_of_blades=int(val("numberOfBlades", "1")),
        blade_width=float(val("bladeWidth")),
        angle1=float(val("angle1")),
        angle2=float(val("angle2")),
        angle3=float(val("angle3")),
        angle4=float(val("angle4")),
    )


def _parse_baffles(geom_elem, config_name: str, reactor_id: str,
                   baffle_stl_names: List[str]) -> List[BaffleSpec]:
    baffles = []
    bc_elem = geom_elem.find("bafflesCustoms")
    if bc_elem is None:
        return baffles

    n = int(bc_elem.find("numberOfBafflesCustom").get("value", "0"))
    if n == 0:
        return baffles

    for b_elem in bc_elem.findall("bafflesCustom"):
        def val(tag, default="0"):
            el = b_elem.find(tag)
            return el.get("value", default) if el is not None else default

        # Try to find matching baffle STL
        stl_name = None
        for s in baffle_stl_names:
            if config_name in s or reactor_id in s:
                stl_name = s
                break

        baffles.append(BaffleSpec(
            style=val("baffleStyle", "Custom"),
            width=float(val("baffleWidth")),
            off_wall=float(val("baffleOffWall")),
            off_bottom=float(val("baffleOffBottom")),
            off_top=float(val("baffleOffTop")),
            stl_filename=stl_name,
        ))

    return baffles


def _parse_operating(proc_elem) -> OperatingCondition:
    oc = proc_elem.find(".//operatingCondition")
    if oc is None:
        return OperatingCondition(300, 0, 0, 0)

    def val(tag, default="0"):
        el = oc.find(tag)
        return el.get("value", default) if el is not None else default

    # Get direction from impellerSpeeds if available
    direction = "Clockwise"
    pi_elem = proc_elem.find(".//processImpeller/direction")
    if pi_elem is not None:
        direction = pi_elem.get("value", "Clockwise")

    return OperatingCondition(
        temperature=float(val("temperature", "300")),
        volume_m3=float(val("operatingVolume")),
        liquid_level=float(val("liquidLevel")),
        rpm=float(val("impellerSpeed")),
        direction=direction,
    )


def _parse_reactor_xml(xml_data: bytes, reactor_id: str,
                       impeller_stl_names: List[str],
                       baffle_stl_names: List[str]) -> List[ReactorConfig]:
    """Parse a reactor XML and return all configurations."""
    root = ET.fromstring(xml_data.decode("utf-8-sig"))
    configs = []

    for rpv in root.findall(".//ReactorProcessVariant"):
        ri = rpv.find("reactorInfo")
        if ri is None:
            continue

        def ri_val(tag, default=""):
            el = ri.find(tag)
            return el.get("value", default) if el is not None else default

        config_name = ri_val("configuration")
        process_name = ri_val("process")
        scale = ri_val("scale", "Lab")

        geom = rpv.find(".//geometry")
        if geom is None:
            continue

        tank = _parse_tank(geom.find("tank"))

        imp_elem = geom.find(".//impeller")
        if imp_elem is None:
            continue
        impeller = _parse_impeller(imp_elem)

        baffles = _parse_baffles(geom, config_name, reactor_id, baffle_stl_names)
        operating = _parse_operating(rpv.find("processData"))

        mech = rpv.find("mechanical")
        motor_power = float(mech.find("motorPower").get("value", "3")) if mech is not None else 3.0

        configs.append(ReactorConfig(
            reactor_id=reactor_id,
            config_name=config_name,
            process_name=process_name,
            scale=scale,
            tank=tank,
            impeller=impeller,
            baffles=baffles,
            operating=operating,
            motor_power=motor_power,
            available_impeller_stls=list(impeller_stl_names),
            available_baffle_stls=list(baffle_stl_names),
        ))

    return configs


def load_reactor_database(mdata_path: str) -> Dict[str, ReactorFamily]:
    """
    Load all reactor families from a MixIT Reactors .mdata archive.

    Returns dict mapping reactor_id -> ReactorFamily.
    """
    mdata_path = Path(mdata_path)
    database = {}

    outer = zipfile.ZipFile(mdata_path, "r")

    for entry_name in sorted(outer.namelist()):
        if not entry_name.endswith(".mdata"):
            continue

        reactor_id = entry_name.replace(".mdata", "")
        inner_data = outer.read(entry_name)
        inner = zipfile.ZipFile(io.BytesIO(inner_data), "r")

        # Collect STL names from .milib
        impeller_stl_names = []
        milib_files = [f for f in inner.namelist() if f.endswith(".milib")]
        for mf in milib_files:
            milib_data = inner.read(mf)
            milib_zip = zipfile.ZipFile(io.BytesIO(milib_data), "r")
            impeller_stl_names.extend(
                f for f in milib_zip.namelist() if f.lower().endswith(".stl")
            )

        # Collect baffle STL names
        baffle_stl_names = [
            f for f in inner.namelist()
            if f.lower().endswith(".stl") and "baffle" in f.lower()
        ]

        # Parse XML
        xml_files = [f for f in inner.namelist() if f.endswith(".xml")]
        # Also check inside milib for XML
        all_configs = []
        for xf in xml_files:
            xml_data = inner.read(xf)
            configs = _parse_reactor_xml(
                xml_data, reactor_id, impeller_stl_names, baffle_stl_names
            )
            all_configs.extend(configs)

        if all_configs:
            database[reactor_id] = ReactorFamily(
                name=reactor_id,
                configs=all_configs,
                impeller_stl_names=impeller_stl_names,
                baffle_stl_names=baffle_stl_names,
            )

    return database


def find_config(database: Dict[str, ReactorFamily],
                reactor_id: str,
                config_name: str) -> Optional[ReactorConfig]:
    """Find a specific configuration by reactor ID and config name."""
    family = database.get(reactor_id)
    if family is None:
        return None
    for cfg in family.configs:
        if cfg.config_name == config_name:
            return cfg
    return None


def print_reactor_summary(database: Dict[str, ReactorFamily]):
    """Print a formatted summary of all available reactors."""
    print(f"{'Reactor':<22} {'Config':<15} {'Tank':<13} {'D(m)':<7} "
          f"{'Bottom':<18} {'Impeller':<30} {'D_imp(m)':<9} {'Vol(L)':<8} {'RPM':<6}")
    print("-" * 150)

    for rid in sorted(database.keys()):
        family = database[rid]
        for cfg in family.configs:
            t = cfg.tank
            imp = cfg.impeller
            op = cfg.operating
            off = ""
            if abs(imp.off_center) > 0.001 or abs(imp.off_center_y) > 0.001:
                off = " *"
            print(f"{rid:<22} {cfg.config_name:<15} {t.type:<13} {t.diameter:<7.4f} "
                  f"{t.bottom_style:<18} {imp.type:<30} {imp.diameter:<9.4f} "
                  f"{op.volume_m3*1000:<8.1f} {op.rpm:<6.0f}{off}")

    print(f"\n* = off-center impeller")
    print(f"Total: {sum(len(f.configs) for f in database.values())} configurations "
          f"across {len(database)} reactor families")
