"""
Extract and scale STL files from MixIT .mdata/.milib archives.

Impeller STLs in MixIT .milib are normalized to unit diameter (span ≈ 1.0).
They need to be scaled by the actual impeller diameter in meters.

Baffle STLs (in BaffleSTLs/) are already in meters and need no scaling.

This module handles extraction, scaling, and optional translation
for impeller positioning (off-center, clearance height).
"""

import zipfile
import io
import struct
import re
from pathlib import Path
from typing import Optional, Tuple


def _is_ascii_stl(data: bytes) -> bool:
    """Check if STL data is ASCII format (starts with 'solid' and contains 'facet')."""
    return data[:5] == b"solid" and b"facet" in data[:500]


def _scale_binary_stl(data: bytes, scale: float = 0.001,
                      translate: Tuple[float, float, float] = (0, 0, 0)) -> bytes:
    """
    Scale and translate a binary STL file.

    Binary STL format:
      header: 80 bytes
      num_triangles: uint32
      For each triangle:
        normal: 3 x float32
        vertex1: 3 x float32
        vertex2: 3 x float32
        vertex3: 3 x float32
        attribute: uint16
    """
    header = data[:80]
    num_triangles = struct.unpack("<I", data[80:84])[0]

    result = bytearray(header)
    result += struct.pack("<I", num_triangles)

    tx, ty, tz = translate
    offset = 84
    for _ in range(num_triangles):
        # Normal (3 floats) — scale but don't translate
        nx, ny, nz = struct.unpack_from("<3f", data, offset)
        result += struct.pack("<3f", nx, ny, nz)
        offset += 12

        # 3 vertices — scale and translate
        for _ in range(3):
            vx, vy, vz = struct.unpack_from("<3f", data, offset)
            result += struct.pack("<3f",
                                  vx * scale + tx,
                                  vy * scale + ty,
                                  vz * scale + tz)
            offset += 12

        # Attribute byte count
        attr = struct.unpack_from("<H", data, offset)[0]
        result += struct.pack("<H", attr)
        offset += 2

    return bytes(result)


def _scale_ascii_stl(data: bytes, scale: float = 0.001,
                     translate: Tuple[float, float, float] = (0, 0, 0)) -> bytes:
    """Scale and translate an ASCII STL file."""
    text = data.decode("utf-8", errors="replace")
    tx, ty, tz = translate

    def replace_vertex(match):
        x = float(match.group(1)) * scale + tx
        y = float(match.group(2)) * scale + ty
        z = float(match.group(3)) * scale + tz
        return f"vertex {x:.10g} {y:.10g} {z:.10g}"

    def replace_normal(match):
        # Normals are unit vectors — don't scale or translate
        return match.group(0)

    # Replace vertex coordinates
    text = re.sub(
        r"vertex\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        replace_vertex, text
    )

    return text.encode("utf-8")


def scale_stl(data: bytes, scale: float = 0.001,
              translate: Tuple[float, float, float] = (0, 0, 0)) -> bytes:
    """Scale and optionally translate STL data (auto-detects ASCII vs binary)."""
    if _is_ascii_stl(data):
        return _scale_ascii_stl(data, scale, translate)
    else:
        return _scale_binary_stl(data, scale, translate)


def extract_impeller_stl(mdata_path: str, reactor_name: str,
                         stl_name: str, output_path: str,
                         impeller_diameter: float = 1.0,
                         translate: Tuple[float, float, float] = (0, 0, 0)):
    """
    Extract an impeller STL from the nested archive, scale to physical size, and write.

    Impeller STLs in MixIT are normalized to unit diameter (span ≈ 1.0).
    They need to be scaled by the actual impeller diameter in meters.

    Args:
        mdata_path: Path to "MixIT Reactors.mdata"
        reactor_name: e.g. "Mobius_MIX"
        stl_name: e.g. "Mobius-MIX100L.stl"
        output_path: Where to write the scaled STL
        impeller_diameter: Actual impeller diameter in meters (used as scale factor)
        translate: (x, y, z) translation in meters to apply after scaling
    """
    outer = zipfile.ZipFile(mdata_path, "r")
    inner_data = outer.read(f"{reactor_name}.mdata")
    inner = zipfile.ZipFile(io.BytesIO(inner_data), "r")

    # Find the .milib file
    milib_files = [f for f in inner.namelist() if f.endswith(".milib")]
    stl_data = None

    for mf in milib_files:
        milib_data = inner.read(mf)
        milib_zip = zipfile.ZipFile(io.BytesIO(milib_data), "r")
        if stl_name in milib_zip.namelist():
            stl_data = milib_zip.read(stl_name)
            break

    if stl_data is None:
        raise FileNotFoundError(
            f"STL '{stl_name}' not found in {reactor_name}.milib"
        )

    # Scale normalized STL by impeller diameter and translate
    scaled = scale_stl(stl_data, scale=impeller_diameter, translate=translate)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(scaled)


def extract_baffle_stl(mdata_path: str, reactor_name: str,
                       stl_name: str, output_path: str,
                       translate: Tuple[float, float, float] = (0, 0, 0)):
    """
    Extract a baffle STL from the nested archive and write to disk.

    Baffle STLs are stored in the inner .mdata archive under BaffleSTLs/
    and are already in meters (no scaling needed).
    """
    outer = zipfile.ZipFile(mdata_path, "r")
    inner_data = outer.read(f"{reactor_name}.mdata")
    inner = zipfile.ZipFile(io.BytesIO(inner_data), "r")

    if stl_name not in inner.namelist():
        raise FileNotFoundError(
            f"Baffle STL '{stl_name}' not found in {reactor_name}.mdata"
        )

    stl_data = inner.read(stl_name)
    # Baffles are already in meters — only apply translation if needed
    if translate != (0, 0, 0):
        scaled = scale_stl(stl_data, scale=1.0, translate=translate)
    else:
        scaled = stl_data

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(scaled)


def extract_all_geometry(mdata_path: str, reactor_name: str,
                         impeller_stl_name: str,
                         baffle_stl_names: list,
                         output_dir: str,
                         impeller_diameter: float = 1.0,
                         impeller_translate: Tuple[float, float, float] = (0, 0, 0)):
    """
    Extract all geometry STLs for a case setup.

    Impeller STLs are normalized to unit diameter and scaled by impeller_diameter.
    Baffle STLs are already in meters.

    Writes to:
        output_dir/constant/geometry/impeller.stl
        output_dir/constant/geometry/baffle_N.stl  (if applicable)
    """
    geom_dir = Path(output_dir) / "constant" / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)

    # Extract impeller — scale from normalized to physical size
    extract_impeller_stl(
        mdata_path, reactor_name, impeller_stl_name,
        str(geom_dir / "impeller.stl"),
        impeller_diameter=impeller_diameter,
        translate=impeller_translate,
    )

    # Extract baffles — already in meters
    for i, bstl in enumerate(baffle_stl_names):
        extract_baffle_stl(
            mdata_path, reactor_name, bstl,
            str(geom_dir / f"baffle_{i+1}.stl"),
        )

    return geom_dir
