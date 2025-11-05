from __future__ import annotations

from typing import TYPE_CHECKING

from quansino.mc.contexts import Context, MultiContexts

if TYPE_CHECKING:
    from ase import Atoms
    from numpy.random import Generator as RNG


def test_multi_contexts(bulk_small: Atoms, rng: RNG):
    """Test the MultiContexts class."""
    context_1 = Context(bulk_small, rng=rng)
    context_2 = Context(bulk_small, rng=rng)

    multi_context = MultiContexts([context_1, context_2])

    assert multi_context.contexts == [context_1, context_2]

    bulk_small.get_potential_energy()

    multi_context.save_state()

    for context in multi_context.contexts:
        assert context.atoms == bulk_small
        assert context.rng == rng
        assert context.last_results == bulk_small.calc.results  # type: ignore

    for context in multi_context.contexts:
        context.last_results = {"energy": -1.0}  # type: ignore

    multi_context.revert_state()

    for context in multi_context.contexts:
        assert context.atoms == bulk_small
        assert context.rng == rng
        assert context.last_results == {"energy": -1.0}  # type: ignore

    dictionary = multi_context.to_dict()
    assert "contexts" in dictionary
    assert len(dictionary["contexts"]) == 2
    assert dictionary["contexts"][0] == context_1.to_dict()
    assert dictionary["contexts"][1] == context_2.to_dict()
