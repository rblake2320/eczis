# UNFILED-IP
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.recon.regularizers import laplacian
from eczis.recon.shift_add import shift_add_reconstruct
from eczis.schemas import CaptureSet, FloatArray


@dataclass(frozen=True, slots=True)
class SolverReport:
    info: int
    iterations: int
    final_relative_change: float
    frame_weights: tuple[float, ...]


def _robust_frame_weights(
    capture: CaptureSet, model: EstimatedCameraModel, initial: FloatArray
) -> np.ndarray:
    errors: list[float] = []
    for frame in capture.frames:
        predicted = model.project(initial, frame.params)
        valid = frame.valid_mask
        error = float(np.median(np.abs(predicted[valid] - frame.data[valid]))) if valid.any() else 1.0
        errors.append(error)
    med = max(float(np.median(errors)), 1e-6)
    weights = 1.0 / (1.0 + (np.asarray(errors) / (2.5 * med)) ** 2)
    return weights.astype(np.float32)


def map_reconstruct(
    capture: CaptureSet,
    model: EstimatedCameraModel,
    *,
    regularization: float,
    max_iterations: int,
) -> tuple[FloatArray, SolverReport]:
    initial = shift_add_reconstruct(capture)
    shape = initial.shape
    size = initial.size
    frame_weights = _robust_frame_weights(capture, model, initial)

    rhs = np.zeros(shape, dtype=np.float32)
    for weight, frame in zip(frame_weights, capture.frames, strict=True):
        observed = np.where(frame.valid_mask, frame.data, 0.0).astype(np.float32)
        rhs += np.float32(weight) * model.adjoint(observed, frame.params)

    def matvec(vector: np.ndarray) -> np.ndarray:
        image = vector.reshape(shape).astype(np.float32)
        result = np.zeros(shape, dtype=np.float32)
        for weight, frame in zip(frame_weights, capture.frames, strict=True):
            projected = model.project(image, frame.params)
            projected = np.where(frame.valid_mask, projected, 0.0).astype(np.float32)
            result += np.float32(weight) * model.adjoint(projected, frame.params)
        result += np.float32(regularization) * (image + 0.15 * laplacian(image))
        return result.ravel().astype(np.float64)

    operator = LinearOperator((size, size), matvec=matvec, dtype=np.float64)
    history: list[float] = []
    previous = initial.astype(np.float64).ravel().copy()

    def callback(xk: np.ndarray) -> None:
        nonlocal previous
        denom = max(float(np.linalg.norm(previous)), 1e-12)
        history.append(float(np.linalg.norm(xk - previous) / denom))
        previous = xk.copy()

    solved, info = cg(
        operator,
        rhs.astype(np.float64).ravel(),
        x0=initial.astype(np.float64).ravel(),
        rtol=2e-4,
        atol=0.0,
        maxiter=max_iterations,
        callback=callback,
    )
    reconstruction = np.clip(solved.reshape(shape), 0.0, 1.0).astype(np.float32)
    report = SolverReport(
        info=int(info),
        iterations=len(history),
        final_relative_change=history[-1] if history else 0.0,
        frame_weights=tuple(float(x) for x in frame_weights),
    )
    return reconstruction, report
