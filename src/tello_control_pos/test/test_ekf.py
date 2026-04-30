"""
Contract tests for the EKF odometry integrator.

These tests verify properties any correct EKF implementation must
satisfy: state vector and covariance keep the right shape, the predict
step propagates position with velocity, sensor updates reduce
uncertainty in the observed variable, and the covariance stays
positive semi-definite under repeated cycles.

Tests do not lock in specific gains, sensor models, or numerical
stability tactics (e.g. Joseph-form update, symmetrization, Mahalanobis
gating). Future work that adds those should keep these tests passing.
"""
from unittest.mock import Mock

import numpy as np
import pytest
from rclpy.duration import Duration

from tello_control_pos.odometry_integrator import EKFOdometryNode


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def make_node():
    """Create fresh EKF nodes with a mocked publisher; destroy on teardown."""
    nodes = []

    def factory():
        n = EKFOdometryNode()
        n.odom_pub.publish = Mock()
        nodes.append(n)
        return n

    yield factory
    for n in nodes:
        n.destroy_node()


def _prime_dt(node, seconds_ago=0.1):
    """Force ``last_time`` into the past so the next predict sees a real dt."""
    node.last_time = node.get_clock().now() - Duration(seconds=seconds_ago)


# ---------------------------------------------------------------------------
# Initial state and shape preservation
# ---------------------------------------------------------------------------

def test_initial_state_shape(make_node):
    """Verify ``x`` is shape (7,) and ``P`` is shape (7,7) at construction."""
    node = make_node()
    assert node.x.shape == (7,)
    assert node.P.shape == (7, 7)


def test_state_shape_preserved_under_cycles(make_node):
    """Preserve state and covariance dimensions after many predict + update cycles."""
    node = make_node()

    for _ in range(100):
        _prime_dt(node, 0.02)
        node.predict_loop()
        node._ekf_update(np.array([1.0]), node.H_tof, node.R_tof)
        node._ekf_update(np.array([0.1, 0.0, 0.0]), node.H_vel, node.R_vel)

    assert node.x.shape == (7,)
    assert node.P.shape == (7, 7)


# ---------------------------------------------------------------------------
# Predict step
# ---------------------------------------------------------------------------

def test_predict_advances_position_with_velocity(make_node):
    """Advance ``px`` when ``vx`` is non-zero in the predict step."""
    node = make_node()
    node.x[3] = 1.0
    px_before = node.x[0]
    _prime_dt(node, 0.1)

    node.predict_loop()

    assert node.x[0] > px_before


def test_predict_grows_position_uncertainty(make_node):
    """Increase position covariance in predict (Q is added every step)."""
    node = make_node()
    pz_var_before = node.P[2, 2]
    _prime_dt(node, 0.1)

    node.predict_loop()

    assert node.P[2, 2] > pz_var_before


# ---------------------------------------------------------------------------
# Update step — uncertainty reduction per sensor
# ---------------------------------------------------------------------------

def test_tof_update_reduces_height_uncertainty(make_node):
    """Reduce ``P[2,2]`` (height variance) after a TOF measurement."""
    node = make_node()
    pz_var_before = node.P[2, 2]

    node._ekf_update(np.array([1.0]), node.H_tof, node.R_tof)

    assert node.P[2, 2] < pz_var_before


def test_velocity_update_reduces_velocity_uncertainty(make_node):
    """Reduce velocity-component variances after a flow-velocity measurement."""
    node = make_node()
    vx_var_before = node.P[3, 3]
    vy_var_before = node.P[4, 4]
    vz_var_before = node.P[5, 5]

    node._ekf_update(np.array([0.1, 0.0, 0.0]), node.H_vel, node.R_vel)

    assert node.P[3, 3] < vx_var_before
    assert node.P[4, 4] < vy_var_before
    assert node.P[5, 5] < vz_var_before


def test_yaw_update_reduces_yaw_uncertainty(make_node):
    """Reduce ``P[6,6]`` (yaw variance) after an IMU yaw measurement."""
    node = make_node()
    yaw_var_before = node.P[6, 6]

    node._ekf_update(np.array([0.5]), node.H_yaw, node.R_yaw)

    assert node.P[6, 6] < yaw_var_before


# ---------------------------------------------------------------------------
# Covariance positive semi-definiteness
# ---------------------------------------------------------------------------

def test_covariance_stays_positive_semidefinite(make_node):
    """
    Keep the covariance positive semi-definite across cycles.

    Drift away from PSD indicates a numerical-stability bug, e.g. the
    simple ``(I - KH)P`` update losing symmetry under accumulated
    rounding. The matrix is symmetrized before checking eigenvalues so
    small numerical asymmetry is allowed; negative eigenvalues are not.
    """
    node = make_node()

    for _ in range(50):
        _prime_dt(node, 0.02)
        node.predict_loop()
        node._ekf_update(np.array([1.0]), node.H_tof, node.R_tof)
        node._ekf_update(np.array([0.1, 0.0, 0.0]), node.H_vel, node.R_vel)
        node._ekf_update(np.array([0.0]), node.H_yaw, node.R_yaw)

    P_sym = 0.5 * (node.P + node.P.T)
    eigenvalues = np.linalg.eigvalsh(P_sym)
    assert np.all(eigenvalues >= -1e-9), (
        f"Covariance lost positive semi-definiteness: "
        f"min eigenvalue = {eigenvalues.min():.3e}"
    )
