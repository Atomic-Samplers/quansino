from __future__ import annotations

from io import StringIO

import numpy as np
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS
from ase.units import fs

from quansino.io import Logger


def test_logger():
    string_io = StringIO()

    logger = Logger(string_io)
    logger.add_field("Class", lambda: "GrandCanonicalMC", str_format="<24s")
    logger.add_field("Epot[eV]", lambda: 123141.0)
    logger.add_field("Step", lambda: 1, str_format=">12d")
    logger.write_header()

    logger()

    header = string_io.getvalue().split("\n")[0]
    values = string_io.getvalue().split("\n")[1]

    assert header == "Class" + " " * 24 + "Epot[eV]" + " " * 9 + "Step"
    assert values == "GrandCanonicalMC" + " " * 10 + "123141.0000" + " " * 12 + "1"


def test_opt_custom_logger(atoms):
    atoms.rattle(0.1)

    string_io = StringIO()
    logger = Logger(string_io)
    opt = BFGS(atoms, logfile=None)

    def negative_omega():
        if opt.nsteps > 0:
            return str(np.any(np.linalg.eigh(opt.H)[0] < 0))
        else:
            return "N/A"

    logger.add_opt_fields(opt)
    logger.add_field("NegativeEigenvalues", negative_omega, str_format=">22s")
    opt.attach(logger)

    logger.write_header()

    opt.run(fmax=0.01)

    text = string_io.getvalue()

    assert "NegativeEigenvalues" in text
    assert "False" in text
    assert "Optimizer" in text

    string_io.close()


def test_md_logger(atoms):
    string_io = StringIO()
    logger = Logger(string_io)

    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    dyn = VelocityVerlet(atoms, 1.0 * fs)

    dyn.attach(logger)

    logger.add_md_fields(dyn)
    logger.write_header()

    dyn.run(10)

    header = string_io.getvalue().split("\n")[0]

    assert (
        header
        == "Time[ps]"
        + " " * 9
        + "Etot[eV]"
        + " " * 5
        + "Epot[eV]"
        + " " * 5
        + "Ekin[eV]"
        + " " * 7
        + "T[K]"
    )

    string_io.close()


def test_opt_stress_logger(atoms):
    atoms.rattle(0.1)

    string_io = StringIO()
    logger = Logger(string_io)

    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    dyn = VelocityVerlet(atoms, 1.0 * fs)

    logger.add_md_fields(dyn)
    logger.add_stress_fields(atoms, mask=[False, True, False, True, True, False])

    dyn.attach(logger)

    logger.write_header()

    dyn.run(10)

    string_io.seek(0)
    logger_lines = string_io.read()

    pos = string_io.tell()

    assert "yyStress[GPa]" in logger_lines
    assert "xxStress[GPa]" not in logger_lines
    assert "zzStress[GPa]" not in logger_lines
    assert "yzStress[GPa]" in logger_lines
    assert "xzStress[GPa]" in logger_lines
    assert "xyStress[GPa]" not in logger_lines

    logger.remove_fields("yyStress[GPa]")
    logger.add_stress_fields(atoms, mask=[False, False, False, False, False, True])

    logger.write_header()

    dyn.run(10)

    string_io.seek(pos)
    new_logger_lines = string_io.read()

    assert "yyStress[GPa]" not in new_logger_lines
    assert "xzStress[GPa]" not in new_logger_lines
    assert "xyStress[GPa]" in new_logger_lines

    string_io.close()
