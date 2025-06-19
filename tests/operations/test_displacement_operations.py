from __future__ import annotations

import numpy as np
from ase.build import molecule
from numpy.testing import assert_allclose, assert_array_equal, assert_array_less
from scipy.stats import chisquare

from quansino.mc.contexts import DisplacementContext
from quansino.operations import (
    Ball,
    Box,
    Rotation,
    Sphere,
    Translation,
    TranslationRotation,
)


def test_displacement_operations(single_atom, rng):
    context = DisplacementContext(single_atom, rng)

    sphere = Sphere(0.1)
    assert sphere.calculate(context).shape == (1, 3)
    assert_allclose(np.linalg.norm(sphere.calculate(context)), 0.1)

    box = Box(0.1)
    assert box.calculate(context).shape == (1, 3)
    assert (box.calculate(context) > -0.1).all()
    assert (box.calculate(context) < 0.1).all()

    ball = Ball(0.1)
    assert ball.calculate(context).shape == (1, 3)
    assert (np.linalg.norm(ball.calculate(context)) < 0.1).all()
    assert (np.linalg.norm(ball.calculate(context)) > 0).all()

    single_atom.set_cell(np.eye(3) * 100)
    context._moving_indices = [0]

    translation = Translation()
    assert translation.calculate(context).shape == (1, 3)
    assert (translation.calculate(context) < 100).all()
    assert (translation.calculate(context) > 0).all()

    old_positions = single_atom.get_positions()

    rotation = Rotation()
    assert rotation.calculate(context).shape == (1, 3)
    single_atom.positions += rotation.calculate(context)
    assert_allclose(context.atoms.positions, old_positions)

    context.atoms = molecule("H2O", vacuum=10)

    old_distances = np.linalg.norm(
        context.atoms.positions[:, None] - context.atoms.positions, axis=-1
    )

    context._moving_indices = [0, 1, 2]

    old_positions = context.atoms.get_positions()
    assert not np.allclose(rotation.calculate(context), 0)
    context.atoms.positions += rotation.calculate(context)
    assert not np.allclose(context.atoms.get_positions(), old_positions)

    new_distances = np.linalg.norm(
        context.atoms.positions[:, None] - context.atoms.positions, axis=-1
    )

    assert_allclose(old_distances, new_distances)

    translation_rotation = TranslationRotation()
    assert translation_rotation.calculate(context).shape == (3, 3)
    assert not np.allclose(translation_rotation.calculate(context), 0)

    new_distances_2 = np.linalg.norm(
        context.atoms.positions[:, None] - context.atoms.positions, axis=-1
    )

    assert_allclose(old_distances, new_distances_2)


def test_box_operation(bulk_small, rng):
    context = DisplacementContext(bulk_small, rng)
    operation = Box(0.1)

    displacement = operation.calculate(context)

    assert displacement.shape == (1, 3)
    assert_array_less(displacement, 0.1)


def test_sphere_operation(bulk_small, rng):
    context = DisplacementContext(bulk_small, rng)
    operation = Sphere(0.1)

    displacement = operation.calculate(context)

    assert displacement.shape == (1, 3)
    assert_allclose(np.linalg.norm(displacement), 0.1)


def test_ball_operation(single_atom, rng):
    operation = Ball(0.1)
    context = DisplacementContext(single_atom, rng)

    displacement = operation.calculate(context)

    assert displacement.shape == (1, 3)
    assert 0.1 > np.linalg.norm(displacement) > 0


def test_translation_operation(single_atom, rng):
    single_atom.center(vacuum=20)

    operation = Translation()
    context = DisplacementContext(single_atom, rng)

    context._moving_indices = [0]

    positions_recording = []

    for _ in range(10000):
        translation = operation.calculate(context)
        single_atom.set_positions(single_atom.positions + translation)
        assert np.all(single_atom.get_scaled_positions() < 1)
        positions_recording.append(single_atom.get_positions())

    positions_recording = np.array(positions_recording).flatten()

    histogram = np.histogram(positions_recording, bins=10)[0]

    assert chisquare(histogram, f_exp=np.ones_like(histogram) * 3000)[1] > 0.001


def test_rotation_operation(single_atom, rng):
    context = DisplacementContext(single_atom, rng)
    context._moving_indices = [0]

    operation = Rotation()

    random_cell = rng.uniform(-10, 10, (3, 3))
    single_atom.set_cell(random_cell)

    original_positions = single_atom.get_positions()

    for _ in range(50):
        translation = operation.calculate(context)
        single_atom.set_positions(single_atom.positions + translation)
        assert_allclose(single_atom.get_positions(), original_positions)

    water = molecule("H2O", vacuum=20)
    context.atoms = water
    context._moving_indices = [0, 1, 2]

    original_distances = water.positions[None, :] - water.positions
    original_distances = np.linalg.norm(original_distances, axis=-1)

    original_com = water.get_center_of_mass()
    original_cell = water.get_cell()

    for _ in range(50):
        translation = operation.calculate(context)
        water.set_positions(water.positions + translation)
        assert np.all(water.get_scaled_positions() < 1)
        new_distances = water.positions[None, :] - water.positions
        new_distances = np.linalg.norm(new_distances, axis=-1)
        assert_allclose(original_distances, new_distances)
        assert_allclose(original_com, water.get_center_of_mass())
        assert_allclose(original_cell, water.get_cell())


def test_translation_rotation_operation(single_atom, rng):
    context = DisplacementContext(single_atom, rng)
    context._moving_indices = [0]

    operation = TranslationRotation()

    random_cell = rng.uniform(-10, 10, (3, 3))
    single_atom.set_cell(random_cell)

    for _ in range(50):
        translation = operation.calculate(context)
        single_atom.set_positions(single_atom.positions + translation)
        assert_array_less(single_atom.get_scaled_positions(), 1)
        assert_array_less(-single_atom.get_scaled_positions(), 0)

    water = molecule("H2O", vacuum=20)
    context.atoms = water
    context._moving_indices = [0, 1, 2]

    original_distances = water.positions[None, :] - water.positions
    original_distances = np.linalg.norm(original_distances, axis=-1)

    for _ in range(50):
        translation = operation.calculate(context)
        water.set_positions(water.positions + translation)
        assert_array_less(water.get_scaled_positions(), 1.05)
        assert_array_less(-water.get_scaled_positions(), 0.05)
        new_distances = water.positions[None, :] - water.positions
        new_distances = np.linalg.norm(new_distances, axis=-1)
        assert_array_equal(original_distances, new_distances)
