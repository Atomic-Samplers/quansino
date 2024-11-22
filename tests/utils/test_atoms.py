from __future__ import annotations

from ase.constraints import FixAtoms, FixCom, FixedPlane

from quansino.utils.atoms import has_constraint


def test_has_constraint(bulk_large):

    bulk_large.set_constraint(FixAtoms(indices=[0]))

    assert has_constraint(bulk_large, FixAtoms)
    assert has_constraint(bulk_large, "FixAtoms")

    bulk_large.set_constraint(FixCom())

    assert has_constraint(bulk_large, FixCom)
    assert has_constraint(bulk_large, "FixCom")

    bulk_large.set_constraint(FixedPlane([0, 1], (0, 0, 1)))

    assert has_constraint(bulk_large, FixedPlane)
    assert has_constraint(bulk_large, "FixedPlane")
    assert not has_constraint(bulk_large, "FixConstraint")
    assert not has_constraint(bulk_large, "FixCartesian")
    assert not has_constraint(bulk_large, "FixBondLength")
