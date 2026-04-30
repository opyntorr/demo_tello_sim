"""Contract tests for the position controller.

These tests describe what *any* position controller for this drone must
do: produce zero velocity when at the target, drive the drone toward
the target, respect velocity and integral limits, scale output by
``vel_scale``, and ignore commands when prerequisites are missing.

They do *not* lock in implementation details — there are no asserts
that integrals reset on a particular branch, that the derivative is
filtered, that a specific gain is used, or that the tolerance is
exactly 10 cm. PR #2 (and any future controller refactor) should
keep these tests passing.
"""
import math
from unittest.mock import Mock

import pytest
from geometry_msgs.msg import Point
from rclpy.duration import Duration

from tello_control_pos.controller import TelloPositionController


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def make_node():
    """Factory fixture. Each call returns a fresh controller node with
    its publish() method replaced by a Mock. All nodes created via the
    factory are destroyed on teardown."""
    nodes = []

    def factory():
        n = TelloPositionController()
        n.cmd_vel_pub.publish = Mock()
        nodes.append(n)
        return n

    yield factory
    for n in nodes:
        n.destroy_node()


def _set_pose(node, x, y, z):
    p = Point()
    p.x, p.y, p.z = float(x), float(y), float(z)
    node.current_pose = p


def _set_target(node, x, y, z):
    node.target_x = float(x)
    node.target_y = float(y)
    node.target_z = float(z)
    node.target_received = True


def _prime_dt(node, seconds_ago=0.1):
    """Force ``last_time`` into the past so the next control_loop sees a
    known positive dt. Uses ``Duration`` subtraction so this works
    regardless of whether the node is on system time or sim time."""
    node.last_time = node.get_clock().now() - Duration(seconds=seconds_ago)


def _last_twist(node):
    return node.cmd_vel_pub.publish.call_args.args[0]


# ---------------------------------------------------------------------------
# Safety gates
# ---------------------------------------------------------------------------

def test_does_not_publish_without_target(make_node):
    node = make_node()
    _set_pose(node, 0.0, 0.0, 1.0)
    node.control_loop()
    node.cmd_vel_pub.publish.assert_not_called()


def test_does_not_publish_without_pose(make_node):
    node = make_node()
    _set_target(node, 1.0, 0.0, 1.0)
    node.control_loop()
    node.cmd_vel_pub.publish.assert_not_called()


def test_does_not_publish_below_takeoff_height(make_node):
    """Pose below the takeoff threshold (drone hasn't lifted off yet)
    must short-circuit so we don't issue commands during ascent."""
    node = make_node()
    _set_pose(node, 0.0, 0.0, 0.2)
    _set_target(node, 1.0, 0.0, 1.0)
    node.control_loop()
    node.cmd_vel_pub.publish.assert_not_called()


# ---------------------------------------------------------------------------
# At-target behavior
# ---------------------------------------------------------------------------

def test_at_target_outputs_zero_velocity(make_node):
    """Contract: when essentially on top of the target, the controller
    publishes a Twist with zero linear velocity.

    This is a chosen contract — an alternative valid design would be to
    not publish at all when at the target. Both the current main
    implementation and PR #2 satisfy this stricter contract, so we pin
    it down here. If a future redesign opts for "publish nothing,"
    update this test deliberately.
    """
    node = make_node()
    _set_pose(node, 1.0, 1.0, 1.0)
    _set_target(node, 1.0, 1.0, 1.0)  # exactly at target
    _prime_dt(node)

    node.control_loop()

    twist = _last_twist(node)
    assert twist.linear.x == 0.0
    assert twist.linear.y == 0.0
    assert twist.linear.z == 0.0


# ---------------------------------------------------------------------------
# Direction correctness — sign of output must match sign of error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pose,target,expected_signs",
    [
        # Positive direction on each axis
        ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (+1, 0, 0)),
        ((0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0, +1, 0)),
        ((0.0, 0.0, 1.0), (0.0, 0.0, 2.0), (0, 0, +1)),
        # Negative direction on each axis
        ((1.0, 0.0, 1.0), (0.0, 0.0, 1.0), (-1, 0, 0)),
        ((0.0, 1.0, 1.0), (0.0, 0.0, 1.0), (0, -1, 0)),
        ((0.0, 0.0, 2.0), (0.0, 0.0, 1.0), (0, 0, -1)),
    ],
)
def test_velocity_points_toward_target(make_node, pose, target, expected_signs):
    """Output velocity sign on each axis must match the sign of the
    error to the target — no sign flips, no axis crosstalk.

    Note: the strict 1e-6 tolerance on "zero" axes assumes an
    axis-independent controller (no body-frame rotation, no yaw
    coupling). If the controller later adds those, this test should be
    updated deliberately rather than loosened silently.
    """
    node = make_node()
    _set_pose(node, *pose)
    _set_target(node, *target)
    _prime_dt(node)

    node.control_loop()
    twist = _last_twist(node)

    for axis, expected_sign in zip(("x", "y", "z"), expected_signs):
        actual = getattr(twist.linear, axis)
        if expected_sign > 0:
            assert actual > 0, f"axis {axis} expected positive, got {actual}"
        elif expected_sign < 0:
            assert actual < 0, f"axis {axis} expected negative, got {actual}"
        else:
            assert abs(actual) < 1e-6, f"axis {axis} expected ~0, got {actual}"


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------

def test_velocity_does_not_exceed_max(make_node):
    """An absurdly large error must not produce velocities larger than
    the configured maximum (after vel_scale)."""
    node = make_node()
    _set_pose(node, 0.0, 0.0, 1.0)
    _set_target(node, 100.0, 100.0, 100.0)
    _prime_dt(node)

    node.control_loop()
    twist = _last_twist(node)

    cap = node.max_vel * node.vel_scale + 1e-9
    assert abs(twist.linear.x) <= cap
    assert abs(twist.linear.y) <= cap
    assert abs(twist.linear.z) <= cap


# ---------------------------------------------------------------------------
# Anti-windup — observed externally via published velocity, not via
# internal integral state, so future controllers that handle windup
# differently (leak, conditional integration, etc.) still pass.
# ---------------------------------------------------------------------------

def test_no_runaway_after_sustained_error(make_node):
    """When the drone cannot reach the target (pose stuck), the
    published velocity must stay bounded by max_vel and must not keep
    growing. This is the externally-observable consequence of any
    correct anti-windup policy."""
    node = make_node()
    _set_pose(node, 0.0, 0.0, 1.0)
    _set_target(node, 5.0, 5.0, 5.0)

    velocities_x = []
    for _ in range(500):
        _prime_dt(node, seconds_ago=0.1)
        node.control_loop()
        velocities_x.append(_last_twist(node).linear.x)

    cap = node.max_vel * node.vel_scale + 1e-9

    # Every published command stays bounded.
    assert all(abs(v) <= cap for v in velocities_x)

    # In steady state (last 100 samples), the output is stable —
    # not oscillating wildly. This catches integrator runaway that
    # might be hidden by saturation but visible as oscillation.
    tail = velocities_x[-100:]
    assert max(tail) - min(tail) < 0.1


# ---------------------------------------------------------------------------
# velocity_scale parameter
# ---------------------------------------------------------------------------

def test_velocity_scale_is_proportional(make_node):
    """Scaling vel_scale by k must scale the published velocity by k.

    Note on saturation: this controller clamps to max_vel *before*
    applying vel_scale (controller.py: clamp at lines 119-121, scale
    at lines 140-142). So even when the internal command saturates,
    ``published = clip(internal) * scale`` remains linear in ``scale``.
    Two fresh nodes with identical inputs but different scales let us
    verify that linearity directly.
    """
    node1 = make_node()
    node1.vel_scale = 1.0
    _set_pose(node1, 0.0, 0.0, 1.0)
    _set_target(node1, 1.0, 0.0, 1.0)
    _prime_dt(node1)
    node1.control_loop()
    v1 = _last_twist(node1).linear.x

    node10 = make_node()
    node10.vel_scale = 10.0
    _set_pose(node10, 0.0, 0.0, 1.0)
    _set_target(node10, 1.0, 0.0, 1.0)
    _prime_dt(node10)
    node10.control_loop()
    v10 = _last_twist(node10).linear.x

    assert v1 > 0.0  # sanity: there should be a positive command in +X
    assert abs(v10 - 10.0 * v1) < 1e-6


# ---------------------------------------------------------------------------
# Closed-loop convergence — strongest "is this actually a controller"
# check we can do without a Gazebo simulation.
# ---------------------------------------------------------------------------

def test_closed_loop_converges_toward_target(make_node):
    """Iteratively feeding the controller's published velocity back
    into a simulated pose must drive the drone toward the target.

    The plant is a first-order integrator (``pose += twist * dt``),
    not a real drone or Gazebo simulation: there is no inertia, no
    drag, no sensor noise. That is intentional — this test is about
    the controller's *direction and magnitude* contract, not flight
    dynamics. A failure here means the controller doesn't even
    converge in the easiest possible plant.
    """
    node = make_node()
    target = (1.0, 0.5, 1.5)
    pose = [0.0, 0.0, 1.0]  # above takeoff threshold
    _set_target(node, *target)

    initial_distance = math.dist(pose, target)
    dt_step = 0.1

    for _ in range(200):  # 20 simulated seconds
        _set_pose(node, *pose)
        _prime_dt(node, seconds_ago=dt_step)
        node.control_loop()
        twist = _last_twist(node)
        pose[0] += twist.linear.x * dt_step
        pose[1] += twist.linear.y * dt_step
        pose[2] += twist.linear.z * dt_step

    final_distance = math.dist(pose, target)

    # Significant progress: at least halved the distance.
    assert final_distance < initial_distance / 2, (
        f"Controller did not converge: initial={initial_distance:.3f}m, "
        f"final={final_distance:.3f}m"
    )
    # And reasonably close — within 30 cm is generous for any
    # controller after 20 seconds on a ~1.9 m journey.
    assert final_distance < 0.3, (
        f"Controller did not converge to within 30 cm: "
        f"final={final_distance:.3f}m"
    )
