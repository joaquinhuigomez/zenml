"""Startup controller for dynamic pipeline step execution."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from zenml.config.step_configurations import GroupInfo, StepConfigurationUpdate
from zenml.enums import StepRuntime
from zenml.execution.pipeline.dynamic.outputs import (
    AnyStepFuture,
    BaseFuture,
    MapResultsFuture,
    StepFuture,
    _InlineStepFuture,
    _IsolatedStepFuture,
)

if TYPE_CHECKING:
    from zenml.models import StepRunResponse
    from zenml.steps import BaseStep
    from zenml.execution.pipeline.dynamic.runner import DynamicPipelineRunner


class _StartupStatus(Enum):
    """Lifecycle states of pending step and map startup work."""

    WAITING_ON_DEPENDENCIES = auto()
    EXPANDING = auto()
    DISPATCHING = auto()
    ACTIVE_INLINE = auto()
    ACTIVE_ISOLATED = auto()
    FINISHED_SUCCESS = auto()
    FINISHED_FAILED = auto()
    FINISHED = auto()


@dataclass
class PendingStepStartup:
    """Internal controller record for a submitted step startup."""

    invocation_id: str
    step: "BaseStep"
    inputs: Dict[str, Any]
    after: Union["AnyStepFuture", Sequence["AnyStepFuture"], None]
    step_config: Optional["StepConfigurationUpdate"]
    output_keys: List[str]
    runtime: StepRuntime
    wrapped_future: Union["_InlineStepFuture", "_IsolatedStepFuture"]
    future: StepFuture
    dependencies: List[BaseFuture]
    status: _StartupStatus = _StartupStatus.WAITING_ON_DEPENDENCIES


@dataclass
class PendingMapExpansion:
    """Internal controller record for a pending map expansion."""

    step: "BaseStep"
    kwargs: Dict[str, Any]
    after: Union["AnyStepFuture", Sequence["AnyStepFuture"], None]
    product: bool
    group_info: GroupInfo
    expansion_future: Future[List[StepFuture]]
    dependencies: List[BaseFuture]
    status: _StartupStatus = _StartupStatus.WAITING_ON_DEPENDENCIES


@dataclass
class _InlineCompletionEvent:
    """Event emitted when an inline executor future completes."""

    invocation_id: str
    future: Future["StepRunResponse"]


@dataclass
class _IsolatedCompletionEvent:
    """Event emitted when an isolated step reaches a terminal state."""

    invocation_id: str
    step_run: "StepRunResponse"


class StartupController:
    """Controller that releases step startups once dependencies are ready."""

    def __init__(self, runner: "DynamicPipelineRunner") -> None:
        """Initialize the startup controller.

        Args:
            runner: The owning dynamic pipeline runner.
        """
        self._runner = runner
        self._condition = threading.Condition()
        self._pending_steps: Dict[str, PendingStepStartup] = {}
        self._pending_maps: List[PendingMapExpansion] = []
        self._events: List[
            Union[
                _InlineCompletionEvent,
                _IsolatedCompletionEvent,
            ]
        ] = []
        self._shutdown = False
        self._thread = threading.Thread(
            name="DynamicPipelineRunner-Startup-Controller",
            target=self._run_loop,
            daemon=True,
        )
        self._thread.start()

    def submit_step(
        self,
        *,
        step: "BaseStep",
        invocation_id: str,
        inputs: Dict[str, Any],
        after: Union["AnyStepFuture", Sequence["AnyStepFuture"], None],
        step_config: Optional["StepConfigurationUpdate"],
        output_keys: List[str],
        runtime: StepRuntime,
        waiting_for_steps: List[str],
    ) -> StepFuture:
        """Register a submitted step with the startup controller."""
        if waiting_for_steps:
            self._runner._log_waiting_for_steps(
                waiting_for_steps=waiting_for_steps,
                invocation_id=invocation_id,
            )

        if runtime == StepRuntime.INLINE:
            wrapped_future: Union[_InlineStepFuture, _IsolatedStepFuture] = (
                _InlineStepFuture(invocation_id=invocation_id)
            )
        else:
            wrapped_future = _IsolatedStepFuture(
                pipeline_run_id=self._runner.run.id,
                invocation_id=invocation_id,
            )

        future = StepFuture(wrapped=wrapped_future, output_keys=output_keys)
        self._runner._track_future(invocation_id=invocation_id, future=future)

        record = PendingStepStartup(
            invocation_id=invocation_id,
            step=step,
            inputs=inputs,
            after=after,
            step_config=step_config,
            output_keys=output_keys,
            runtime=runtime,
            wrapped_future=wrapped_future,
            future=future,
            dependencies=_collect_dependencies(inputs=inputs, after=after),
        )

        with self._condition:
            self._pending_steps[invocation_id] = record
            self._condition.notify_all()

        return future

    def submit_map(
        self,
        *,
        step: "BaseStep",
        kwargs: Dict[str, Any],
        after: Union["AnyStepFuture", Sequence["AnyStepFuture"], None],
        product: bool,
        group_info: GroupInfo,
    ) -> MapResultsFuture:
        """Register a pending map expansion with the startup controller."""
        expansion_future: Future[List[StepFuture]] = Future()
        map_future = MapResultsFuture(wrapped=expansion_future)
        record = PendingMapExpansion(
            step=step,
            kwargs=kwargs,
            after=after,
            product=product,
            group_info=group_info,
            expansion_future=expansion_future,
            dependencies=_collect_dependencies(inputs=kwargs, after=None),
        )

        with self._condition:
            self._pending_maps.append(record)
            self._condition.notify_all()

        return map_future

    def notify_inline_step_finished(
        self, invocation_id: str, future: Future["StepRunResponse"]
    ) -> None:
        """Notify the controller about an inline step completion."""
        with self._condition:
            self._events.append(
                _InlineCompletionEvent(
                    invocation_id=invocation_id,
                    future=future,
                )
            )
            self._condition.notify_all()

    def notify_isolated_step_finished(
        self, invocation_id: str, step_run: "StepRunResponse"
    ) -> None:
        """Notify the controller about an isolated terminal step state."""
        with self._condition:
            self._events.append(
                _IsolatedCompletionEvent(
                    invocation_id=invocation_id,
                    step_run=step_run,
                )
            )
            self._condition.notify_all()

    def await_all(self) -> None:
        """Wait until the controller has no in-progress work left."""
        with self._condition:
            while self._has_in_progress_work_locked():
                self._condition.wait()

    def has_in_progress_steps(self) -> bool:
        """Check whether the controller still tracks in-progress work."""
        with self._condition:
            return self._has_in_progress_work_locked()

    def shutdown(self) -> None:
        """Stop the startup controller loop."""
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()

        self._thread.join()

    def _run_loop(self) -> None:
        """Run the startup controller loop."""
        while True:
            with self._condition:
                while not self._shutdown and not self._should_process_locked():
                    self._condition.wait()

                if self._shutdown:
                    return

                events = list(self._events)
                self._events.clear()

            for event in events:
                self._handle_event(event)

            map_actions = self._collect_map_actions()
            for action in map_actions:
                self._expand_map(action)

            startup_actions = self._collect_step_startup_actions()
            for record in startup_actions:
                self._start_step(record)

            with self._condition:
                self._condition.notify_all()

    def _should_process_locked(self) -> bool:
        """Check whether the loop should process work."""
        return bool(
            self._events
            or any(
                record.status == _StartupStatus.WAITING_ON_DEPENDENCIES
                for record in self._pending_steps.values()
            )
            or any(
                record.status == _StartupStatus.WAITING_ON_DEPENDENCIES
                for record in self._pending_maps
            )
        )

    def _has_in_progress_work_locked(self) -> bool:
        """Check whether the controller still tracks unfinished work."""
        if any(
            record.status
            not in {
                _StartupStatus.FINISHED_SUCCESS,
                _StartupStatus.FINISHED_FAILED,
            }
            for record in self._pending_steps.values()
        ):
            return True

        return any(
            record.status != _StartupStatus.FINISHED
            for record in self._pending_maps
        )

    def _handle_event(
        self,
        event: Union[
            _InlineCompletionEvent,
            _IsolatedCompletionEvent,
        ],
    ) -> None:
        """Handle a startup controller event."""
        with self._condition:
            record = self._pending_steps.get(event.invocation_id)
            if record is None:
                return

            if isinstance(event, _InlineCompletionEvent):
                try:
                    step_run = event.future.result()
                except BaseException as exception:
                    record.wrapped_future._set_terminal_exception(exception)
                    record.status = _StartupStatus.FINISHED_FAILED
                else:
                    record.wrapped_future._set_terminal_result(step_run)
                    record.status = _StartupStatus.FINISHED_SUCCESS
            elif isinstance(event, _IsolatedCompletionEvent):
                if event.step_run.status.is_successful:
                    record.wrapped_future._set_terminal_result(event.step_run)
                    record.status = _StartupStatus.FINISHED_SUCCESS
                else:
                    exception = self._runner._reconstruct_step_run_exception(
                        invocation_id=event.invocation_id,
                        step_run=event.step_run,
                    )
                    record.wrapped_future._set_terminal_exception(exception)
                    record.status = _StartupStatus.FINISHED_FAILED

            self._condition.notify_all()

    def _collect_map_actions(self) -> List[PendingMapExpansion]:
        """Collect map expansions that are ready to execute."""
        ready_actions = []
        with self._condition:
            for record in self._pending_maps:
                if record.status != _StartupStatus.WAITING_ON_DEPENDENCIES:
                    continue

                ready, exception = _dependencies_ready_or_exception(
                    record.dependencies
                )
                if not ready:
                    continue

                if exception:
                    record.expansion_future.set_exception(exception)
                    record.status = _StartupStatus.FINISHED
                    continue

                record.status = _StartupStatus.EXPANDING
                ready_actions.append(record)

        return ready_actions

    def _expand_map(self, record: PendingMapExpansion) -> None:
        """Expand a pending map operation."""
        try:
            resolved_kwargs = self._runner._resolve_step_inputs(record.kwargs)
            step_inputs = self._runner._expand_map_inputs(
                inputs=resolved_kwargs, product=record.product
            )
            child_futures = [
                self.submit_step(
                    step=record.step.copy(),
                    invocation_id=self._runner._reserve_invocation_id(
                        step=record.step, custom_id=None
                    ),
                    inputs=inputs,
                    after=record.after,
                    step_config=self._runner._create_step_config(
                        group=record.group_info
                    ),
                    output_keys=list(record.step.entrypoint_definition.outputs),
                    runtime=self._runner._predict_step_runtime(
                        step=record.step,
                        step_config=self._runner._create_step_config(
                            group=record.group_info
                        ),
                    ),
                    waiting_for_steps=self._runner._get_running_upstream_steps_for_startup_controller(
                        inputs=inputs,
                        after=record.after,
                    ),
                )
                for inputs in step_inputs
            ]
        except BaseException as exception:
            record.expansion_future.set_exception(exception)
        else:
            record.expansion_future.set_result(child_futures)
        finally:
            with self._condition:
                record.status = _StartupStatus.FINISHED
                self._condition.notify_all()

    def _collect_step_startup_actions(self) -> List[PendingStepStartup]:
        """Collect step startups that are ready to dispatch."""
        ready_actions = []
        with self._condition:
            for record in self._pending_steps.values():
                if record.status != _StartupStatus.WAITING_ON_DEPENDENCIES:
                    continue

                ready, exception = _dependencies_ready_or_exception(
                    record.dependencies
                )
                if not ready:
                    continue

                if exception:
                    record.wrapped_future._set_terminal_exception(exception)
                    record.status = _StartupStatus.FINISHED_FAILED
                    continue

                record.status = _StartupStatus.DISPATCHING
                ready_actions.append(record)

        return ready_actions

    def _start_step(self, record: PendingStepStartup) -> None:
        """Start a ready step."""
        try:
            resolved_inputs = self._runner._resolve_step_inputs(record.inputs)
            prepared = self._runner._prepare_step_launch_for_startup_controller(
                step=record.step,
                invocation_id=record.invocation_id,
                inputs=resolved_inputs,
                after=record.after,
                step_config=record.step_config,
            )

            existing_status = (
                self._runner._maybe_handle_existing_step_run_for_startup_controller(
                    invocation_id=record.invocation_id,
                    prepared=prepared,
                    wrapped_future=record.wrapped_future,
                )
            )
            if existing_status is not None:
                status = existing_status
            elif record.runtime == StepRuntime.INLINE:
                executor_future = self._runner._launch_scheduled_inline_step(
                    step=prepared.compiled_step,
                    remaining_retries=prepared.remaining_retries,
                )
                record.wrapped_future._bind_execution_future(executor_future)
                executor_future.add_done_callback(
                    lambda future, invocation_id=record.invocation_id: (
                        self.notify_inline_step_finished(
                            invocation_id=invocation_id,
                            future=future,
                        )
                    )
                )
                status = _StartupStatus.ACTIVE_INLINE
            else:
                self._runner._launch_scheduled_isolated_step(
                    step=prepared.compiled_step,
                    wrapped_future=record.wrapped_future,
                )
                status = _StartupStatus.ACTIVE_ISOLATED
        except BaseException as exception:
            record.wrapped_future._set_terminal_exception(exception)
            status = _StartupStatus.FINISHED_FAILED

        with self._condition:
            record.status = status
            self._condition.notify_all()


def _collect_dependencies(
    inputs: Dict[str, Any],
    after: Union["AnyStepFuture", Sequence["AnyStepFuture"], None],
) -> List[BaseFuture]:
    """Collect controller dependencies from step inputs and `after`."""
    dependencies: List[BaseFuture] = []
    for value in inputs.values():
        if isinstance(value, BaseFuture):
            dependencies.append(value)
        elif isinstance(value, Sequence) and all(
            isinstance(item, BaseFuture) for item in value
        ):
            dependencies.extend(value)

    if isinstance(after, BaseFuture):
        dependencies.append(after)
    elif isinstance(after, Sequence):
        dependencies.extend(after)

    return dependencies


def _dependencies_ready_or_exception(
    dependencies: List[BaseFuture],
) -> tuple[bool, Optional[BaseException]]:
    """Check whether dependencies are ready or failed."""
    for dependency in dependencies:
        if dependency.running():
            return False, None

    for dependency in dependencies:
        try:
            dependency.result()
        except BaseException as exception:
            return True, exception

    return True, None
