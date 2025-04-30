from __future__ import annotations

import operator
import time
from contextlib import suppress
from functools import partial
from time import sleep
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


class UnlimitedWaiterError(Exception):
    def __init__(self) -> None:
        super().__init__("Wrong waiter configuration: endless timeout is not allowed with not limited attempts")


class UnusedMaxIntervalError(Exception):
    def __init__(self) -> None:
        super().__init__("Wrong waiter configuration: max_interval should be used with is_exponential `True`")


class WaiterTimeoutError(Exception):
    pass


class Wait[T]:
    def __init__(  # noqa: PLR0913
        self,
        action: Callable[..., T],
        *args: Any,
        timeout: float,
        max_attempts: int = 0,
        exceptions_to_ignore: tuple[type[Exception], ...] = (),
        interval: float = 0.1,
        is_exponential: bool = False,
        max_interval: float = 0,
        timeout_message: str = "",
        debug: bool = False,
        **kwargs: Any,
    ) -> None:
        """Waiter logic class to call a callable until the expected result.

        Waiter can be limited by timeout or/and by maximal calls count.
        If the remaining time till timeout is less then interval to sleep,
        waiter will sleep remaining time only and do the last call.

        :param action: Callable to call
        :param args: Positional args for the action
        :param timeout: Maximal time in seconds to wait. 0 to wait without limit by time.
            0 is allowed only with max_attempts != 0.
        :param max_attempts: Maximal calls count. 0 to call without limit by count
        :param exceptions_to_ignore: Exceptions which will be ignored if happen during the action call
        :param interval: Polling interval in seconds between calls of an action
        :param is_exponential: Exponential waiter doubles interval after every call
        :param max_interval: Limit in seconds for the exponential waiter. Is used only with is_exponential = True.
            0 to ignore and increase interval endlessly
        :param timeout_message: message in case of waiter is failed
        :param debug: save all results of action while polling and show them in timeout exception if happens
        :param kwargs: Keyword args for the action
        """
        self._action = partial(action, *args, **kwargs)

        if timeout <= 0 and max_attempts <= 0:
            raise UnlimitedWaiterError

        self._timeout = timeout
        self._max_attempts = max_attempts
        self._interval = interval
        self._exceptions_to_ignore = exceptions_to_ignore

        if not is_exponential and max_interval != 0:
            raise UnusedMaxIntervalError

        self._is_exponential = is_exponential
        self._max_interval = max_interval
        self._timeout_message = timeout_message
        self._debug = debug

        self._calls_count = 0
        self._results: list[Any] | None = None

    def _poll(self, predicate: Callable[[T], bool]) -> T:
        self._calls_count = 0
        if self._debug:
            self._results = []

        end_time = time.time() + self._timeout

        while True:
            remaining_time = end_time - time.time()
            if (self._timeout > 0 and remaining_time < 0) or (
                self._max_attempts != 0 and self._calls_count >= self._max_attempts
            ):
                break

            if self._is_exponential:
                interval = self._interval * (2**self._calls_count)
                delay = min(interval, self._max_interval) if self._max_interval != 0 else interval
            else:
                delay = self._interval

            sleep(min(delay, remaining_time) if self._timeout > 0 else delay)
            self._calls_count += 1

            with suppress(*self._exceptions_to_ignore):
                result = self._action()
                if self._debug and self._results is not None:
                    self._results.append(result)
                if predicate(result):
                    return result

        msg_parts = [
            msg_part
            for msg_part in (
                self._timeout_message,
                f"Waiter timeout after {self._calls_count} action calls",
                f"Results: {self._results}" if self._debug else "",
            )
            if msg_part
        ]
        msg = "\n".join(msg_parts)
        raise WaiterTimeoutError(msg)

    def until(self, predicate: Callable[[T], bool] | None = None) -> T:
        return self._poll(predicate=predicate or operator.truth)

    def until_not(self) -> T:
        return self._poll(predicate=operator.not_)

    def until_equal_to(self, value: T) -> T:
        return self._poll(predicate=partial(operator.eq, value))

    def until_not_equal_to(self, value: T) -> T:
        return self._poll(predicate=partial(operator.ne, value))

    def until_is(self, value: T) -> T:
        return self._poll(predicate=partial(operator.is_, value))

    def until_is_not(self, value: T) -> T:
        return self._poll(predicate=partial(operator.is_not, value))

    def until_is_true(self) -> Literal[True]:
        return self._poll(predicate=partial(operator.is_, True))  # type: ignore[return-value]  # noqa: FBT003

    def until_is_false(self) -> Literal[False]:
        return self._poll(predicate=partial(operator.is_, False))  # type: ignore[return-value]  # noqa: FBT003

    def until_is_none(self) -> None:
        return self._poll(predicate=partial(operator.is_, None))  # type: ignore[return-value]

    def until_is_not_none(self) -> T:
        # TODO: is it possible to constraint this method annotation that it never returns none?
        # without passing action as method argument?
        # action: Callable[..., T | None]
        # https://github.com/airreality/air-waiter/issues/4
        return self._poll(predicate=partial(operator.is_not, None))
