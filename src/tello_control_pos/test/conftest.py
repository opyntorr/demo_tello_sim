"""Shared fixtures for tello_control_pos unit tests.

rclpy must be initialized exactly once per test session. Each test
creates a fresh node (so state doesn't leak between tests) and that
node is destroyed during teardown.
"""
import pytest
import rclpy


@pytest.fixture(scope='session', autouse=True)
def _rclpy_session():
    rclpy.init()
    yield
    rclpy.shutdown()
