from concurrent.futures import Future

from zenml.execution.pipeline.dynamic.outputs import (
    _InlineStepFuture,
    _IsolatedStepFuture,
    MapResultsFuture,
    StepFuture,
)


def test_inline_step_future_running_checks() -> None:
    wrapped: Future[object] = Future()
    inline_future = _InlineStepFuture(invocation_id="inline", wrapped=wrapped)
    assert inline_future.running() is True
    wrapped.set_result(object())
    assert inline_future.running() is False


def test_inline_step_future_waits_for_scheduler_dispatch() -> None:
    inline_future = _InlineStepFuture(invocation_id="inline")

    assert inline_future.running() is True

    wrapped: Future[object] = Future()
    inline_future._bind_execution_future(wrapped)

    assert inline_future.running() is True

    wrapped.set_result(object())

    assert inline_future.result() is not None
    assert inline_future.running() is False


def test_isolated_step_future_running_checks_before_dispatch() -> None:
    isolated_future = _IsolatedStepFuture(
        pipeline_run_id="00000000-0000-0000-0000-000000000000",
        invocation_id="isolated",
    )

    assert isolated_future.running() is True

    isolated_future._set_terminal_result(object())

    assert isolated_future.running() is False


def test_map_results_future_waits_for_expansion() -> None:
    expansion: Future[list[StepFuture]] = Future()
    map_future = MapResultsFuture(wrapped=expansion)

    assert map_future.running() is True

    wrapped: Future[object] = Future()
    child_future = StepFuture(
        wrapped=_InlineStepFuture(invocation_id="child", wrapped=wrapped),
        output_keys=[],
    )
    expansion.set_result([child_future])

    assert len(map_future) == 1
    assert map_future.running() is True

    wrapped.set_result(object())

    assert map_future.running() is False
