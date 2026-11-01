import numpy as np
from scipy.spatial.transform import Rotation

from posecalib.geometry import eight_point, pnp_dlt, pose_from_essential, triangulate_dlt, umeyama


def _two_cameras(rng):
    R = Rotation.from_rotvec([0.05, 0.6, -0.1]).as_matrix()
    t = np.array([-1.0, 0.1, 0.3])
    t /= np.linalg.norm(t)
    X = rng.uniform([-1, -1, 4], [1, 1, 6], (60, 3))
    x1 = X[:, :2] / X[:, 2:]
    Xc2 = X @ R.T + t
    x2 = Xc2[:, :2] / Xc2[:, 2:]
    return R, t, X, x1, x2


def test_essential_and_decomposition_recover_relative_pose():
    rng = np.random.default_rng(0)
    R, t, X, x1, x2 = _two_cameras(rng)
    E = eight_point(x1[None], x2[None])[0]
    n_front, R_e, t_e, X_e, _ = pose_from_essential(E, x1, x2)
    assert n_front == len(X)
    assert np.allclose(R_e, R, atol=1e-8)
    assert np.allclose(t_e, t, atol=1e-8)
    s = np.linalg.norm(X[0]) / np.linalg.norm(X_e[0])          # unit-baseline reconstruction
    assert np.allclose(s * X_e, X, atol=1e-6)


def test_triangulation_is_exact_without_noise():
    rng = np.random.default_rng(1)
    Rs = Rotation.from_rotvec(rng.normal(0, 0.3, (3, 3))).as_matrix()
    ts = rng.normal(0, 0.5, (3, 3)) + [0, 0, 5]
    X = rng.uniform(-1, 1, (20, 3))
    Xc = np.einsum('mij,tj->tmi', Rs, X) + ts[None]
    xn = Xc[..., :2] / Xc[..., 2:]
    X_e = triangulate_dlt(Rs, ts, xn, np.ones((20, 3)))
    assert np.allclose(X_e, X, atol=1e-8)


def test_pnp_dlt_is_exact_without_noise():
    rng = np.random.default_rng(2)
    R = Rotation.from_rotvec([0.2, -0.4, 0.1]).as_matrix()
    t = np.array([0.1, -0.2, 4.0])
    X = rng.uniform(-1, 1, (12, 3))
    Xc = X @ R.T + t
    R_e, t_e = pnp_dlt(X[None], (Xc[:, :2] / Xc[:, 2:])[None])
    assert np.allclose(R_e[0], R, atol=1e-8)
    assert np.allclose(t_e[0], t, atol=1e-8)


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(3)
    src = rng.normal(size=(10, 3))
    R = Rotation.from_rotvec([0.3, 0.2, -0.5]).as_matrix()
    dst = 2.5 * src @ R.T + [1, 2, 3]
    s, R_e, t_e = umeyama(src, dst)
    assert np.isclose(s, 2.5)
    assert np.allclose(R_e, R)
    assert np.allclose(t_e, [1, 2, 3])
