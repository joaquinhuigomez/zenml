#  Copyright (c) ZenML GmbH 2025. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
"""Dynamic pipeline execution outputs."""

from abc import ABC, abstractmethod
from concurrent.futures import Future
import threading
from typing import Any, Iterator, List, Optional, Tuple, Union, overload
from uuid import UUID

from zenml.logger import get_logger
from zenml.models import ArtifactVersionResponse, StepRunResponse
from zenml.utils import exception_utils

logger = get_logger(__name__)


class OutputArtifact(ArtifactVersionResponse):
    """Dynamic step run output artifact."""

    output_name: str
    step_name: str
    chunk_index: Optional[int] = None
    chunk_size: Optional[int] = None

    def chunk(self, index: int) -> "OutputArtifact":
        """Get a chunk of the output artifact.

        Args:
            index: The index of the chunk.

        Raises:
            ValueError: If the output artifact can not be chunked or the index
                is out of range.

        Returns:
            The artifact chunk.
        """
        if not self.item_count:
            raise ValueError(
                f"Output artifact `{self.output_name}` of step "
                f"`{self.step_name}` can not be chunked."
            )

        if index < 0 or index >= self.item_count:
            raise ValueError(
                f"Chunk index `{index}` out of range for output artifact "
                f"`{self.output_name}` of step `{self.step_name}`."
            )

        if self.chunk_index is not None and self.chunk_index != index:
            raise ValueError(
                f"Output artifact `{self.output_name}` of step "
                f"`{self.step_name}` is already referring to a "
                "different chunk."
            )

        return self.model_copy(update={"chunk_index": index, "chunk_size": 1})


StepRunOutputs = Union[None, OutputArtifact, Tuple[OutputArtifact, ...]]


class BaseFuture(ABC):
    """Base future."""

    @abstractmethod
    def running(self) -> bool:
        """Check if the future is running.

        Returns:
            True if the future is running, False otherwise.
        """

    @abstractmethod
    def result(self) -> Any:
        """Get the result of the future.

        Returns:
            The result of the future.
        """


class _StepFutureState:
    """Shared lifecycle state for a startup-controlled step invocation."""

    def __init__(self) -> None:
        """Initialize the future state."""
        self._condition = threading.Condition()
        self._dispatched = False
        self._inline_future: Optional[Future["StepRunResponse"]] = None
        self._terminal_step_run: Optional["StepRunResponse"] = None
        self._terminal_exception: Optional[BaseException] = None

    def bind_inline_future(self, future: Future["StepRunResponse"]) -> None:
        """Bind the executor future of an inline step.

        Args:
            future: The executor future of the inline step execution.
        """
        with self._condition:
            self._inline_future = future
            self._dispatched = True
            self._condition.notify_all()

    def mark_dispatched(self) -> None:
        """Mark the invocation as dispatched."""
        with self._condition:
            self._dispatched = True
            self._condition.notify_all()

    def set_terminal_result(self, step_run: "StepRunResponse") -> None:
        """Mark the invocation as finished successfully.

        Args:
            step_run: The terminal step run.
        """
        with self._condition:
            self._terminal_step_run = step_run
            self._condition.notify_all()

    def set_terminal_exception(self, exception: BaseException) -> None:
        """Mark the invocation as failed before or after dispatch.

        Args:
            exception: The exception to raise for waiters.
        """
        with self._condition:
            self._terminal_exception = exception
            self._condition.notify_all()

    def inline_future(self) -> Optional[Future["StepRunResponse"]]:
        """Get the executor future of an inline step if available.

        Returns:
            The executor future if it was bound.
        """
        with self._condition:
            return self._inline_future

    def is_dispatched(self) -> bool:
        """Check whether the invocation was dispatched.

        Returns:
            Whether the invocation was dispatched.
        """
        with self._condition:
            return self._dispatched

    def is_terminal(self) -> bool:
        """Check whether the invocation reached a terminal state.

        Returns:
            Whether the invocation reached a terminal state.
        """
        with self._condition:
            return (
                self._terminal_step_run is not None
                or self._terminal_exception is not None
            )

    def wait_for_inline_future_or_terminal(
        self,
    ) -> Optional[Future["StepRunResponse"]]:
        """Wait for the inline executor future or terminal pre-dispatch state.

        Returns:
            The inline executor future if one was bound, otherwise `None` if the
            invocation reached a terminal state before dispatch.
        """
        with self._condition:
            while (
                self._inline_future is None
                and self._terminal_step_run is None
                and self._terminal_exception is None
            ):
                self._condition.wait()

            return self._inline_future

    def wait_for_dispatch_or_terminal(self) -> bool:
        """Wait for dispatch or terminal pre-dispatch state.

        Returns:
            Whether the invocation was dispatched.
        """
        with self._condition:
            while (
                not self._dispatched
                and self._terminal_step_run is None
                and self._terminal_exception is None
            ):
                self._condition.wait()

            return self._dispatched

    def wait_for_terminal_result(self) -> "StepRunResponse":
        """Wait for a terminal result and raise stored exceptions.

        # noqa: DAR401
        Raises:
            BaseException: The stored terminal exception.

        Returns:
            The successful terminal step run.
        """
        with self._condition:
            while (
                self._terminal_step_run is None
                and self._terminal_exception is None
            ):
                self._condition.wait()

            if self._terminal_exception is not None:
                raise self._terminal_exception

            assert self._terminal_step_run is not None
            return self._terminal_step_run


class _SchedulerBackedStepFuture(BaseFuture):
    """Base class for startup-controlled step futures."""

    def __init__(
        self, invocation_id: str, state: Optional[_StepFutureState] = None
    ) -> None:
        """Initialize the startup-controlled step future.

        Args:
            invocation_id: The invocation ID of the step run.
            state: Optional shared startup-controller state.
        """
        self.invocation_id = invocation_id
        self._state = state or _StepFutureState()

    def _set_terminal_result(self, step_run: "StepRunResponse") -> None:
        """Mark this future as successfully finished.

        Args:
            step_run: The successful terminal step run.
        """
        self._state.set_terminal_result(step_run)

    def _set_terminal_exception(self, exception: BaseException) -> None:
        """Mark this future as failed.

        Args:
            exception: The exception to raise for waiters.
        """
        self._state.set_terminal_exception(exception)


class _InlineStepFuture(BaseFuture):
    """Future for an inline step run."""

    def __init__(
        self,
        invocation_id: str,
        wrapped: Optional[Future["StepRunResponse"]] = None,
        state: Optional[_StepFutureState] = None,
    ) -> None:
        """Initialize the inline step run future.

        Args:
            invocation_id: The invocation ID of the step run.
            wrapped: Optional wrapped executor future.
            state: Optional shared startup-controller state.
        """
        self.invocation_id = invocation_id
        self._state = state or _StepFutureState()
        if wrapped is not None:
            self._state.bind_inline_future(wrapped)

    def _bind_execution_future(
        self, wrapped: Future["StepRunResponse"]
    ) -> None:
        """Bind the executor future once the controller dispatches the step.

        Args:
            wrapped: The wrapped executor future.
        """
        self._state.bind_inline_future(wrapped)

    def _set_terminal_result(self, step_run: "StepRunResponse") -> None:
        """Mark this future as successfully finished.

        Args:
            step_run: The successful terminal step run.
        """
        self._state.set_terminal_result(step_run)

    def _set_terminal_exception(self, exception: BaseException) -> None:
        """Mark this future as failed.

        Args:
            exception: The exception to raise for waiters.
        """
        self._state.set_terminal_exception(exception)

    def running(self) -> bool:
        """Check if the step run future is running.

        Returns:
            True if the step run future is running, False otherwise.
        """
        wrapped = self._state.inline_future()
        if wrapped is not None:
            return not wrapped.done()

        return not self._state.is_terminal()

    def result(self) -> "StepRunResponse":
        """Get the result of the step run future.

        # noqa: DAR401
        Raises:
            BaseException: Any exception that happened while waiting for the
                step to finish or before it was dispatched.

        Returns:
            The result of the step run future.
        """
        wrapped = self._state.wait_for_inline_future_or_terminal()
        if wrapped is None:
            return self._state.wait_for_terminal_result()

        return wrapped.result()


class _IsolatedStepFuture(_SchedulerBackedStepFuture):
    """Future for an isolated step run."""

    def __init__(
        self,
        pipeline_run_id: UUID,
        invocation_id: str,
        wrapped: Optional[Future["StepRunResponse"]] = None,
        state: Optional[_StepFutureState] = None,
    ) -> None:
        """Initialize the step run future.

        Args:
            pipeline_run_id: The ID of the pipeline run.
            invocation_id: The invocation ID of the step run.
            wrapped: Optional future to wait for that submits the step run.
            state: Optional shared startup-controller state.
        """
        super().__init__(invocation_id=invocation_id, state=state)
        self._wrapped = wrapped
        self.pipeline_run_id = pipeline_run_id

    def _mark_dispatched(self) -> None:
        """Mark the step as dispatched."""
        self._state.mark_dispatched()

    def running(self) -> bool:
        """Check if the isolated step future is running.

        Returns:
            True if the isolated step future is running, False otherwise.
        """
        from zenml.execution.pipeline.dynamic.utils import get_latest_step_run

        if self._wrapped and not self._wrapped.done():
            return True

        if not self._state.is_dispatched():
            return not self._state.is_terminal()

        step_run = get_latest_step_run(
            self.pipeline_run_id, self.invocation_id, hydrate=False
        )

        return not step_run.status.is_finished

    def result(self) -> "StepRunResponse":
        """Get the result of the step future.

        # noqa: DAR401
        Raises:
            BaseException: Any exception that happened while waiting for the
                step to finish.

        Returns:
            The result of the step future.
        """
        from zenml.execution.pipeline.dynamic.utils import (
            wait_for_step_to_finish,
        )

        if self._wrapped:
            # We first wait until the step run is submitted and only then
            # start monitoring the actual step.
            self._wrapped.result()
            self._state.mark_dispatched()

        if not self._state.wait_for_dispatch_or_terminal():
            return self._state.wait_for_terminal_result()

        step_run = wait_for_step_to_finish(
            pipeline_run_id=self.pipeline_run_id, step_name=self.invocation_id
        )

        if step_run.status.is_failed:
            raise exception_utils.reconstruct_exception(
                exception_info=step_run.exception_info,
                fallback_message=(
                    f"Step `{self.invocation_id}` failed with "
                    f"status `{step_run.status}`."
                ),
            )

        return step_run


class BaseStepFuture(BaseFuture):
    """Base step future."""

    def __init__(
        self,
        wrapped: Union[_InlineStepFuture, _IsolatedStepFuture],
        **kwargs: Any,
    ) -> None:
        """Initialize the dynamic step run future.

        Args:
            wrapped: The wrapped future object.
            **kwargs: Additional keyword arguments.
        """
        self._wrapped = wrapped

    @property
    def invocation_id(self) -> str:
        """The step run invocation ID.

        Returns:
            The step run invocation ID.
        """
        return self._wrapped.invocation_id

    def running(self) -> bool:
        """Check if the step run future is running.

        Returns:
            True if the step run future is running, False otherwise.
        """
        return self._wrapped.running()


class ArtifactFuture(BaseStepFuture):
    """Future for a step run output artifact."""

    def __init__(
        self,
        wrapped: Union[_InlineStepFuture, _IsolatedStepFuture],
        index: int,
    ) -> None:
        """Initialize the future.

        Args:
            wrapped: The wrapped future object.
            index: The index of the output artifact.
        """
        super().__init__(wrapped=wrapped)
        self._index = index

    def result(self) -> OutputArtifact:
        """Get the output artifact this future represents.

        Raises:
            RuntimeError: If the future returned an invalid output.

        Returns:
            The output artifact.
        """
        step_run = self._wrapped.result()
        from zenml.execution.pipeline.dynamic.utils import (
            load_step_run_outputs,
        )

        result = load_step_run_outputs(step_run.id)

        if isinstance(result, OutputArtifact):
            return result
        elif isinstance(result, tuple):
            return result[self._index]
        else:
            raise RuntimeError(
                f"Step {self.invocation_id} returned an invalid output: "
                f"{result}."
            )

    def load(self, disable_cache: bool = False) -> Any:
        """Load the step run output artifact data.

        Args:
            disable_cache: Whether to disable the artifact cache.

        Returns:
            The step run output artifact data.
        """
        return self.result().load(disable_cache=disable_cache)

    def chunk(self, index: int) -> "OutputArtifact":
        """Get a chunk of the output artifact.

        This method will wait for the future to complete and then return the
        artifact chunk.

        Args:
            index: The index of the chunk.

        Returns:
            The artifact chunk.
        """
        return self.result().chunk(index=index)


class StepFuture(BaseStepFuture):
    """Future for a step run output."""

    def __init__(
        self,
        wrapped: Union[_InlineStepFuture, _IsolatedStepFuture],
        output_keys: List[str],
    ) -> None:
        """Initialize the future.

        Args:
            wrapped: The wrapped future object.
            output_keys: The output keys of the step run.
        """
        super().__init__(wrapped=wrapped)
        self._output_keys = output_keys

    def get_artifact(self, key: str) -> ArtifactFuture:
        """Get an artifact future by key.

        Args:
            key: The key of the artifact future.

        Raises:
            KeyError: If no artifact for the given name exists.

        Returns:
            The artifact future.
        """
        if key not in self._output_keys:
            raise KeyError(
                f"Step run {self.invocation_id} does not have an output with "
                f"the name: {key}."
            )

        return ArtifactFuture(
            wrapped=self._wrapped,
            index=self._output_keys.index(key),
        )

    def wait(self) -> None:
        """Wait for the step to finish."""
        self._wrapped.result()

    def artifacts(self) -> StepRunOutputs:
        """Get the step run output artifacts.

        Returns:
            The step run output artifacts.
        """
        return self.result()

    def result(self) -> StepRunOutputs:
        """Get the step run outputs this future represents.

        Returns:
            The step run outputs.
        """
        from zenml.execution.pipeline.dynamic.utils import (
            load_step_run_outputs,
        )

        step_run = self._wrapped.result()
        return load_step_run_outputs(step_run.id)

    def load(self, disable_cache: bool = False) -> Any:
        """Get the step run output artifact data.

        Args:
            disable_cache: Whether to disable the artifact cache.

        Raises:
            ValueError: If the step run output is invalid.

        Returns:
            The step run output artifact data.
        """
        result = self.artifacts()

        if result is None:
            return None
        elif isinstance(result, ArtifactVersionResponse):
            return result.load(disable_cache=disable_cache)
        elif isinstance(result, tuple):
            return tuple(
                item.load(disable_cache=disable_cache) for item in result
            )
        else:
            raise ValueError(f"Invalid step run output: {result}")

    @overload
    def __getitem__(self, key: int) -> ArtifactFuture: ...

    @overload
    def __getitem__(self, key: slice) -> Tuple[ArtifactFuture, ...]: ...

    def __getitem__(
        self, key: Union[int, slice]
    ) -> Union[ArtifactFuture, Tuple[ArtifactFuture, ...]]:
        """Get an artifact future.

        Args:
            key: The index or slice of the artifact futures.

        Raises:
            TypeError: If the key is not an integer or slice.

        Returns:
            The artifact futures.
        """
        if isinstance(key, int):
            output_key = self._output_keys[key]

            return ArtifactFuture(
                wrapped=self._wrapped,
                index=self._output_keys.index(output_key),
            )
        elif isinstance(key, slice):
            output_keys = self._output_keys[key]
            return tuple(
                ArtifactFuture(
                    wrapped=self._wrapped,
                    index=self._output_keys.index(output_key),
                )
                for output_key in output_keys
            )
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def __iter__(self) -> Any:
        """Iterate over the artifact futures.

        Raises:
            ValueError: If the step does not return any outputs.

        Yields:
            The artifact futures.
        """
        if not self._output_keys:
            raise ValueError(
                f"Step {self.invocation_id} does not return any outputs."
            )

        for index in range(len(self._output_keys)):
            yield ArtifactFuture(
                wrapped=self._wrapped,
                index=index,
            )

    def __len__(self) -> int:
        """Get the number of artifact futures.

        Returns:
            The number of artifact futures.
        """
        return len(self._output_keys)


class MapResultsFuture(BaseFuture):
    """Future that represents the results of a `step.map/product(...)` call."""

    def __init__(
        self,
        futures: Optional[List[StepFuture]] = None,
        wrapped: Optional[Future[List[StepFuture]]] = None,
    ) -> None:
        """Initialize the map results future.

        Args:
            futures: Optional already expanded step run futures.
            wrapped: Optional future that resolves to the expanded step run
                futures once the controller finishes map expansion.
        """
        self._futures = futures
        self._wrapped = wrapped

    @property
    def futures(self) -> List[StepFuture]:
        """Get the expanded step futures.

        Returns:
            The expanded step futures.
        """
        if self._futures is None:
            assert self._wrapped is not None
            self._futures = self._wrapped.result()

        return self._futures

    def expanded_futures(self) -> Optional[List[StepFuture]]:
        """Get expanded step futures without blocking for expansion.

        Returns:
            The expanded step futures if expansion completed, otherwise `None`.
        """
        if self._futures is not None:
            return self._futures

        if self._wrapped and self._wrapped.done():
            self._futures = self._wrapped.result()
            return self._futures

        return None

    def running(self) -> bool:
        """Check if the map results future is running.

        Returns:
            True if the map results future is running, False otherwise.
        """
        if self._wrapped and not self._wrapped.done():
            return True

        return any(future.running() for future in self.futures)

    def result(self) -> List[StepRunOutputs]:
        """Get the step run outputs this future represents.

        Returns:
            The step run outputs.
        """
        return [future.result() for future in self.futures]

    def load(self, disable_cache: bool = False) -> List[Any]:
        """Load the step run output artifacts.

        Args:
            disable_cache: Whether to disable the artifact cache.

        Returns:
            The step run output artifacts.
        """
        return [
            future.load(disable_cache=disable_cache) for future in self.futures
        ]

    def unpack(self) -> Tuple[List[ArtifactFuture], ...]:
        """Unpack the map results future.

        This method can be used to get lists of artifact futures that represent
        the outputs of all the step runs that are part of this map result.

        Example:
        ```python
        from zenml import pipeline, step

        @step
        def create_int_list() -> list[int]:
            return [1, 2]

        @step
        def do_something(a: int) -> Tuple[int, int]:
            return a * 2, a * 3

        @pipeline
        def map_pipeline():
            int_list = create_int_list()
            results = do_something.map(a=int_list)
            double, triple = results.unpack()

            # [future.load() for future in double] will return [2, 4]
            # [future.load() for future in triple] will return [3, 6]
        ```

        Returns:
            The unpacked map results.
        """
        return tuple(map(list, zip(*self.futures)))

    @overload
    def __getitem__(self, key: int) -> StepFuture: ...

    @overload
    def __getitem__(self, key: slice) -> List[StepFuture]: ...

    def __getitem__(
        self, key: Union[int, slice]
    ) -> Union[StepFuture, List[StepFuture]]:
        """Get a step run future.

        Args:
            key: The index or slice of the step run futures.

        Returns:
            The step run futures.
        """
        return self.futures[key]

    def __iter__(self) -> Iterator[StepFuture]:
        """Iterate over the step run futures.

        Yields:
            The step run futures.
        """
        yield from self.futures

    def __len__(self) -> int:
        """Get the number of step run futures.

        Returns:
            The number of step run futures.
        """
        return len(self.futures)


AnyStepFuture = Union[ArtifactFuture, StepFuture, MapResultsFuture]
