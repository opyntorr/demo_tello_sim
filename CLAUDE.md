# Project conventions for Claude

This file is read by Claude Code on every session in this repo. Apply
these rules to all changes.

## Docstring style (pep257-compliant)

The CI `qa` workflow runs `ament_pep257` against Python sources in
`tello_control_pos`. The convention enforces multi-line docstrings with
the summary on the second line (D213), among other rules. To stay
green, follow these patterns.

**Single-line — preferred when one sentence is enough:**

```python
def helper():
    """Reset the integrator state to zero."""
```

Rules: imperative mood (`Reset`, not `Resets`), end with a period.

**Multi-line — when a summary plus a description is warranted:**

```python
def factory():
    """
    Create a fresh controller node with a mocked publisher.

    Each call returns a new ``TelloPositionController`` whose
    ``publish()`` method is a ``Mock``. All nodes are destroyed
    on teardown.
    """
```

Rules:

- Opening `"""` on its own line, then a newline.
- Summary line (imperative, ends with period).
- One blank line between summary and description.
- Closing `"""` on its own line.

**Avoid** these patterns — they trigger ament_pep257 errors:

```python
"""Summary.

Description.
"""              # D213: summary should be on second line, not first.

"""
Summary
Description.
"""              # D205: missing blank line between summary and description.

"""Summary."""   # only OK as single-line; do not mix with longer text below.
```

## Lint policy

`tello_control_pos/setup.cfg` already disables a set of cosmetic flake8
rules the team chose not to enforce (W291, W293, E501, E302, E303,
E305, E702, E741, E221). Do not re-add violations of *other* flake8
rules — those still gate the build.

`E741` (`I` as variable name) is whitelisted because EKF code uses
`I` for the identity matrix by convention.

## Branch and commit conventions

- Branch names: `qa/<area>-<desc>`, `test/<area>-<desc>`,
  `fix/<bug>`, `ci/<desc>`. Use descriptive names, not internal
  phase/sprint IDs.
- Commit messages: Conventional Commits prefix (`test:`, `ci:`, `fix:`,
  `qa:`, `chore:`). Keep subject under ~72 chars.
- Squash-merge to `main`. One PR = one concern.
- Do not include `Co-Authored-By: Claude` lines in commits unless the
  user asks.

## Tests

- New Python tests live under `<package>/test/test_<area>.py`.
- Use `make_node` factory fixtures rather than module-level node
  fixtures — each test should start from a clean state.
- Prefer contract tests (verify externally-observable behavior) over
  tests that read or assert internal attributes. Internal state may
  legitimately change between implementations; contracts shouldn't.
- Mock `publish()` with `unittest.mock.Mock()` rather than
  monkey-patching with a list.
