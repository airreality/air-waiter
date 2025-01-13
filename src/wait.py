from __future__ import annotations

import operator
import time
from contextlib import suppress
from functools import partial
from time import sleep
from typing import Any, Literal, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


class Wait[T]:
    def __init__(  # noqa: PLR0913
        self,
        action: Callable[..., T],
        *args: Any,  # noqa: ANN401
        timeout: float,
        max_attempts: int = 0,
        exceptions_to_ignore: tuple[type[Exception]] = (),  # type: ignore[assignment]
        interval: float = 0.1,
        is_exponential: bool = False,
        max_interval: float = 0,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Waiter logic class to call a callable until expected result.

        Waiter can be limited by timeout or/and by maximal calls count.
        If the remaining time till timeout is less then interval to sleep,
        waiter will sleep remaining time only and do the last call.

        :param action: Callable to call
        :param args: Positional args for the action
        :param timeout: Maximal time in seconds to wait. 0 to wait without limit by time
        :param max_attempts: Maximal calls count. 0 to call without limit by count
        :param exceptions_to_ignore: Exceptions which will be ignored if happen during the action call
        :param interval: Polling interval in seconds between calls of an action
        :param is_exponential: Exponential waiter doubles interval after every call
        :param max_interval: Limit in seconds for the exponential waiter. Is used only with is_exponential = True.
            0 to ignore and increase interval endlessly
        :param kwargs: Keyword args for the action
        """
        self._action = partial(action, *args, **kwargs)
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._interval = interval
        self._exceptions_to_ignore = exceptions_to_ignore
        self._is_exponential = is_exponential
        self._max_interval = max_interval

        self._calls_count = 0
        self._results: list[Any] | None = None

    def _poll(self, predicate: Callable[[T], bool]) -> T:
        self._calls_count = 0
        self._results = []
        assert self._results is not None

        end_time = time.time() + self._timeout

        while True:
            remaining_time = end_time - time.time()
            if (self._timeout != 0 and remaining_time < 0) or self._calls_count >= self._max_attempts:
                break

            delay = (
                min(self._interval * (2**self._calls_count), self._max_interval)
                if self._is_exponential
                else self._interval
            )
            sleep(min(delay, remaining_time))
            self._calls_count += 1

            with suppress(*self._exceptions_to_ignore):
                result = self._action()
                self._results.append(result)
                if predicate(result):
                    return result

        msg = f"Waiter timeout after {self._calls_count} action calls with results {result}"
        raise TimeoutError(msg)

    def until(self) -> T:
        return self._poll(predicate=operator.truth)

    def until_not(self) -> T:
        return self._poll(predicate=operator.not_)

    def until_equal_to(self, value: T) -> T:
        return self._poll(predicate=partial(operator.eq, value))

    def until_not_equal_to(self, value: T) -> T:
        return self._poll(predicate=partial(operator.ne, value))

    def until_true(self) -> Literal[True]:
        return self._poll(predicate=partial(operator.is_, True))  # type: ignore[return-value]  # noqa: FBT003

    def until_false(self) -> Literal[False]:
        return self._poll(predicate=partial(operator.is_, False))  # type: ignore[return-value]  # noqa: FBT003

    def until_none(self) -> None:
        return self._poll(predicate=partial(operator.is_, None))  # type: ignore[return-value]

    def until_not_none(self) -> T:
        # TODO: is it possible to constraint this method annotation that it never returns none?
        # without passing action as method argument?
        # self.action: Callable[..., T | None]
        return self._poll(predicate=partial(operator.is_not, None))
