from __future__ import annotations

from io import StringIO

import numpy as np
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS
from ase.units import fs

from quansino.io import Logger


def test_logger():
    """Test the `Logger` class."""
    string_io = StringIO()

    logger = Logger(string_io, interval=1)
    logger.add_field("Class", lambda: "GrandCanonicalMC", str_format="{:<24s}")
    logger.add_field("Epot[eV]", lambda: 123141.0)
    logger.add_field("Step", lambda: 1, str_format="{:>12d}")
    logger.write_header()

    assert not hasattr(logger, "__dict__")

    logger()

    header = string_io.getvalue().split("\n")[0]
    values = string_io.getvalue().split("\n")[1]

    assert header == "Class" + " " * 22 + "Epot[eV]" + " " * 9 + "Step"
    assert values == "GrandCanonicalMC" + " " * 9 + "123141.000" + " " * 12 + "1"

    logger.remove_fields("Epot[eV]")

    assert "Epot[eV]" not in logger.fields

    logger.close()

    assert string_io.closed


def test_opt_custom_logger(bulk_small):
    """Test the `Logger` class with custom fields."""
    bulk_small.rattle(0.01)

    string_io = StringIO()
    logger = Logger(string_io, interval=1)
    opt = BFGS(bulk_small, logfile=None)

    def negative_omega():
        if opt.nsteps > 0:
            return str(np.any(np.linalg.eigh(opt.H)[0] < 0))  # type: ignore
        else:
            return "N/A"

    logger.add_opt_fields(opt)
    logger.add_field("NegativeEigenvalues", negative_omega, str_format="{:>22s}")
    opt.attach(logger)

    logger.remove_fields("Time")

    logger.add_field(
        "Symbols",
        bulk_small.get_chemical_symbols,
        str_format="{:>4s}" * len(bulk_small),
        header_format=f"{{:>{4 * len(bulk_small)}s}}",
        is_array=True,
    )

    logger.add_field(
        tuple(f"Symbol[{i}]" for i in range(len(bulk_small))),
        bulk_small.get_chemical_symbols,
        str_format="{:>12s}" * len(bulk_small),
        is_array=True,
    )

    logger.write_header()

    opt.run(fmax=0.01)

    text = string_io.getvalue()

    assert "NegativeEigenvalues" in text
    assert "False" in text
    assert "Class" in text

    assert (
        text.strip()
        == """Class                    Step     Epot[eV]   Fmax[eV/A]    NegativeEigenvalues          Symbols    Symbol[0]   Symbol[1]   Symbol[2]   Symbol[3]
BFGS                        0     -14.1584       0.1053                    N/A   Cu  Cu  Cu  Cu           Cu          Cu          Cu          Cu
BFGS                        1     -14.1587       0.0867                  False   Cu  Cu  Cu  Cu           Cu          Cu          Cu          Cu
BFGS                        2     -14.1595       0.0299                  False   Cu  Cu  Cu  Cu           Cu          Cu          Cu          Cu
BFGS                        3     -14.1595       0.0265                  False   Cu  Cu  Cu  Cu           Cu          Cu          Cu          Cu
BFGS                        4     -14.1596       0.0022                  False   Cu  Cu  Cu  Cu           Cu          Cu          Cu          Cu"""
    )

    string_io.close()


def test_ase_md_logger(bulk_small):
    """Test the `Logger` class with ASE MD simulation."""
    string_io = StringIO()
    logger = Logger(string_io, interval=1)

    MaxwellBoltzmannDistribution(bulk_small, temperature_K=300)
    dyn = VelocityVerlet(bulk_small, 1.0 * fs)

    dyn.attach(logger)

    logger.add_md_fields(dyn)
    logger.write_header()

    dyn.run(10)

    header = string_io.getvalue().split("\n")[0]

    assert (
        header
        == "Time[ps]" + " " * 9 + "Epot[eV]" + " " * 5 + "Ekin[eV]" + " " * 7 + "T[K]"
    )

    string_io.close()


def test_stress_logger(bulk_small):
    """Test the `Logger` class with stress fields."""
    bulk_small.rattle(0.1)

    bulk_small.calc = EMT()  # Stress

    string_io = StringIO()
    logger = Logger(string_io, interval=1)

    MaxwellBoltzmannDistribution(bulk_small, temperature_K=300)
    dyn = VelocityVerlet(bulk_small, 1.0 * fs)

    logger.add_md_fields(dyn)
    logger.add_stress_fields(bulk_small)

    dyn.attach(logger)

    logger.write_header()

    dyn.run(10)

    string_io.seek(0)
    logger_lines = string_io.read()

    pos = string_io.tell()

    assert "Stress[xx][GPa]" in logger_lines
    assert "Stress[yy][GPa]" in logger_lines
    assert "Stress[zz][GPa]" in logger_lines
    assert "Stress[yz][GPa]" in logger_lines
    assert "Stress[xz][GPa]" in logger_lines
    assert "Stress[xy][GPa]" in logger_lines

    logger.remove_fields("Stress[yy][GPa]")
    logger.add_stress_fields(bulk_small, mask=[False, False, False, False, False, True])

    logger.write_header()

    dyn.run(10)

    string_io.seek(pos)
    new_logger_lines = string_io.read()

    assert "Stress[xx][GPa]" not in new_logger_lines
    assert "Stress[yy][GPa]" not in new_logger_lines
    assert "Stress[zz][GPa]" not in new_logger_lines
    assert "Stress[yz][GPa]" not in new_logger_lines
    assert "Stress[xz][GPa]" not in new_logger_lines
    assert "Stress[xy][GPa]" in new_logger_lines

    string_io.close()

    assert string_io.closed
