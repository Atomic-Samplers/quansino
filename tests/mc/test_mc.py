from __future__ import annotations

from pathlib import Path

import pytest
from ase.calculators.calculator import Calculator
from tests.conftest import DummyCriteria

from quansino.io.logger import Logger
from quansino.mc.contexts import DisplacementContext
from quansino.mc.core import MonteCarlo
from quansino.mc.criteria import BaseCriteria
from quansino.moves import moves_registry
from quansino.moves.core import BaseMove, CompositeMove
from quansino.moves.displacement import DisplacementMove
from quansino.moves.exchange import ExchangeMove
from quansino.operations import operations_registry
from quansino.operations.cell import DeformationOperation
from quansino.operations.displacement import DisplacementOperation
from quansino.registry import register


def test_mc_class(bulk_small):
    """Test the `MonteCarlo` class."""
    mc = MonteCarlo(bulk_small, seed=42)

    assert mc._MonteCarlo__seed == 42  # type: ignore
    assert mc._rng is not None
    assert mc.default_trajectory is None
    assert mc.default_logger is None
    assert mc._default_restart is None
    assert mc.atoms == bulk_small
    assert mc.converged()

    mc.validate_simulation()

    assert isinstance(mc.atoms.calc, Calculator)

    del mc.atoms.calc.results

    with pytest.raises(AttributeError):
        mc.validate_simulation()

    with pytest.warns(
        UserWarning, match="Atoms object does not have calculator attached"
    ):
        mc.save_state()

    mc.atoms.calc = None
    with pytest.raises(AttributeError):
        mc.validate_simulation()

    with pytest.warns(
        UserWarning, match="Atoms object does not have calculator attached"
    ):
        mc.revert_state()

    mc.step_count = -1
    assert not mc.converged()

    move = DisplacementMove([])

    with pytest.raises(ValueError):
        mc.add_move(move, DummyCriteria(), probability=1, minimum_count=1000)

    assert (
        mc.__repr__()
        == "MonteCarlo(atoms=Atoms(symbols='Cu4', pbc=True, cell=[3.61, 3.61, 3.61]), max_cycles=1, seed=42, moves={}, step_count=-1, default_logger=None, default_trajectory=None, default_restart=None)"
    )


def test_mc_yield_moves(bulk_small):
    """Test the `yield_moves` method of the `MonteCarlo` class."""
    mc = MonteCarlo(bulk_small, seed=42)

    assert list(mc.yield_moves()) == []

    mc.add_move(
        DisplacementMove([]), criteria=DummyCriteria(), name="my_move", probability=1.0
    )

    assert list(mc.yield_moves()) == ["my_move"]

    mc.add_move(
        DisplacementMove([]),
        criteria=DummyCriteria(),
        name="my_move_2",
        probability=1.0,
        minimum_count=1,
    )

    assert list(mc.yield_moves()) == ["my_move_2"]

    with pytest.raises(ValueError):
        mc.add_move(
            DisplacementMove([]),
            criteria=DummyCriteria(),
            name="my_move",
            minimum_count=1,
        )

    mc.step_count = 1

    mc.moves["my_move"].interval = 2
    del mc.moves["my_move_2"]

    assert list(mc.yield_moves()) == []

    mc.context = DisplacementContext(bulk_small, mc._rng)

    move = DisplacementMove([])

    mc.add_move(
        move,
        criteria=DummyCriteria(),
        name="my_move_3",
        probability=1.0,
        minimum_count=1,
    )

    move.check_move = lambda: False

    for move_name in mc.step():
        assert move_name == "my_move_3"

    assert mc.move_history == [("my_move_3", None)]

    mc.max_cycles = 100

    for i in range(20):
        mc.add_move(
            DisplacementMove([]),
            criteria=DummyCriteria(),
            name=f"move_{i}",
            minimum_count=2,
        )

    for i in range(20, 50):
        mc.add_move(
            DisplacementMove([]),
            criteria=DummyCriteria(),
            name=f"move_{i}",
            probability=0.5,
        )

    move_list = list(mc.yield_moves())
    assert len(move_list) == 100

    for i in range(20):
        assert move_list.count(f"move_{i}") >= 2


def test_mc_logger(bulk_small, tmp_path):
    """Test the `Logger` class and its integration with `MonteCarlo`."""
    import sys  # noqa: PLC0415

    mc = MonteCarlo(bulk_small, seed=42, logfile=sys.stdout, logging_interval=1)

    assert not Path("sys.stdout").exists()
    assert not Path("<sys.stdout>").exists()
    assert mc.default_logger is not None
    assert len(mc.observers) == 1
    assert mc.logging_interval == 1
    assert mc.default_logger.file == sys.stdout

    assert not sys.stdout.closed
    mc.close()
    assert not sys.stdout.closed
    mc.default_logger.close()
    assert not sys.stdout.closed

    mc = MonteCarlo(
        bulk_small,
        seed=42,
        logfile=tmp_path / "mc.log",
        logging_mode="a",
        logging_interval=10,
    )

    assert mc.default_logger is not None
    assert Path(tmp_path, "mc.log").exists()
    assert mc.default_logger.file.name == str(tmp_path / "mc.log")
    assert str(mc.default_logger) == f"Path:{tmp_path / 'mc.log'!s}"
    assert mc.default_logger.file.mode == "a"
    assert mc.logging_interval == 10

    mc.close()

    assert mc.default_logger.file.closed

    logfile_path_str = str(Path(tmp_path, "mc_str.log"))
    mc = MonteCarlo(
        bulk_small, seed=42, trajectory=tmp_path / "mc.traj", logfile=logfile_path_str
    )

    assert Path(logfile_path_str).exists()
    assert Path(tmp_path, "mc.traj").exists()
    assert mc.default_trajectory is not None
    assert len(mc.observers) == 2

    logger = Logger(logfile=logfile_path_str, interval=1, mode="w")

    assert logger.file.name == logfile_path_str
    assert logger.file.mode == "w"
    assert logger.interval == 1

    logger.close()

    assert logger.file.closed

    with pytest.raises(ValueError):
        Logger(logfile=logger.file, interval=1, mode="w")

    with open(tmp_path / "mc_new.log", "w") as new_file:
        logger = Logger(logfile=new_file, interval=1, mode="w")
        assert logger.file.name == str(tmp_path / "mc_new.log")
        assert logger.file.mode == "w"
        logger()

    with pytest.raises(ValueError):
        logger()

    logger = Logger(logfile=Path(tmp_path / "mc_new.log"), interval=1, mode="w")

    mc = MonteCarlo(
        bulk_small, seed=42, logfile=logger, logging_mode="a", logging_interval=10
    )

    assert mc.default_logger is not None
    assert mc.default_logger.file.name == str(tmp_path / "mc_new.log")
    assert mc.default_logger.file.mode == "w"
    assert mc.logging_interval == 10
    assert mc.default_logger.interval == 1

    for i in range(10):
        logger = Logger(
            logfile=Path(tmp_path / f"mc_new_{i}.log"), interval=i, mode="w"
        )
        mc.attach_observer(f"logger_{i}", logger)

        observer = mc.observers[f"logger_{i}"]
        assert isinstance(observer, Logger)

        assert observer.file.name == str(tmp_path / f"mc_new_{i}.log")
        assert observer.file.mode == "w"
        assert observer.interval == i
        assert mc.logging_interval == 10
        assert not observer.file.closed

    mc.close()

    assert mc.default_logger.file.closed

    for i in range(10):
        observer = mc.observers[f"logger_{i}"]
        assert isinstance(observer, Logger)

        assert observer.file.name == str(tmp_path / f"mc_new_{i}.log")
        assert observer.file.mode == "w"
        assert observer.interval == i
        assert observer.file.closed

        mc.detach_observer(f"logger_{i}")
        assert f"logger_{i}" not in mc.observers


def test_mc_serialization_deserialization(bulk_small):
    """Test the serialization and deserialization of the `MonteCarlo` class."""
    mc = MonteCarlo(bulk_small, seed=42)

    dict_atoms = mc.atoms.copy()

    mc.step_count = 1234
    dictionary = mc.to_dict()

    assert dictionary == {
        "atoms": dict_atoms,
        "name": "MonteCarlo",
        "kwargs": {
            "seed": 42,
            "logging_interval": 1,
            "max_cycles": 1,
            "logging_mode": "a",
        },
        "context": {"last_results": {}},
        "attributes": {"step_count": 1234},
        "rng_state": mc._rng.bit_generator.state,
        "moves": {},
    }

    reconstructed_mc = MonteCarlo.from_dict(mc.to_dict())

    new_dictionary = reconstructed_mc.to_dict()

    assert dictionary == new_dictionary

    @register()
    class DummyCriteria(BaseCriteria):
        def evaluate(self) -> bool: ...

    for move_name, move in moves_registry.items():
        for operation_name, operation in operations_registry.items():
            if issubclass(operation, DeformationOperation):
                move_operation = operation(max_value=0.1)
            elif issubclass(operation, DisplacementOperation):
                move_operation = operation()
            else:
                try:
                    move_operation = operation()
                except TypeError as e:
                    raise TypeError(
                        f"Operation {operation_name} is not compatible with move {move_name}"
                    ) from e

            if issubclass(move, CompositeMove):
                to_add = move(
                    moves=[
                        DisplacementMove(labels=[0, 1, 2, 3], operation=move_operation)
                    ]
                )
            elif issubclass(move, ExchangeMove):
                to_add = move(labels=[-1, -1, -1, -1], operation=move_operation)
            elif issubclass(move, DisplacementMove):
                to_add = move(labels=[0, 1, 2, 3], operation=move_operation)
            elif issubclass(move, BaseMove):
                continue
            else:
                raise ValueError(
                    f"This test is not implemented for the {move_name} type."
                )

            mc.add_move(
                to_add, name=f"{move_name}_{operation_name}", criteria=DummyCriteria()
            )

    data = mc.to_dict()

    reconstructed_mc = MonteCarlo.from_dict(data)
    new_data = reconstructed_mc.to_dict()

    assert str(data) == str(new_data)
