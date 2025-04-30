from __future__ import annotations

import re
import time
from typing import Any, TYPE_CHECKING

import pytest

from air_waiter.wait import UnlimitedWaiterError, UnusedMaxIntervalError, Wait, WaiterTimeoutError

if TYPE_CHECKING:
    from pytest_mock import MockFixture


MAX_THRESHOLD = 0.02


class TestWait:
    @staticmethod
    @pytest.mark.parametrize(
        ("args", "kwargs"),
        (
            ((), {}),
            ((1, "2"), {}),
            ((), {"k1": "v1", "k2": "v2"}),
            (("1", None), {"k": "v"}),
        ),
    )
    def test_action(mocker: MockFixture, args: tuple[Any], kwargs: dict[str, Any]) -> None:
        action_mock = mocker.Mock(return_value=True)
        Wait(action_mock, *args, timeout=1, interval=0, **kwargs).until()
        action_mock.assert_called_once_with(*args, **kwargs)

    @staticmethod
    @pytest.mark.parametrize("timeout", (0.1, 0.05))
    def test_timeout(mocker: MockFixture, timeout: float) -> None:
        action_mock = mocker.Mock(return_value=None)

        start = time.perf_counter()
        with pytest.raises(WaiterTimeoutError):
            Wait(action_mock, timeout=timeout, interval=0).until()

        assert 0 <= time.perf_counter() - start - timeout <= MAX_THRESHOLD

    @staticmethod
    @pytest.mark.parametrize("max_attempts", (2, 3))
    def test_max_attempts(mocker: MockFixture, max_attempts: int) -> None:
        action_mock = mocker.Mock(return_value=None)
        with pytest.raises(WaiterTimeoutError):
            Wait(action_mock, timeout=1, interval=0, max_attempts=max_attempts).until()

        assert action_mock.call_count == max_attempts

    @staticmethod
    @pytest.mark.parametrize(
        ("max_attempts", "timeout", "is_timeout_reached"),
        (
            (2, 10, False),
            (2, 0, False),
            (100000, 0.03, True),
            (0, 0.01, True),
        ),
    )
    def test_max_attempts_with_timeout(
        mocker: MockFixture, max_attempts: int, timeout: float, is_timeout_reached: bool
    ) -> None:
        action_mock = mocker.Mock(return_value=None)

        start = time.perf_counter()
        with pytest.raises(WaiterTimeoutError):
            Wait(action_mock, timeout=timeout, interval=0, max_attempts=max_attempts).until()

        if is_timeout_reached:
            assert 0 <= time.perf_counter() - start - timeout <= MAX_THRESHOLD
            if max_attempts != 0:
                assert action_mock.call_count < max_attempts
        else:
            assert action_mock.call_count == max_attempts
            if timeout != 0:
                assert 0 <= time.perf_counter() - start < timeout

    @staticmethod
    def test_unlimited_max_attempts_with_unlimited_timeout(mocker: MockFixture) -> None:
        action_mock = mocker.Mock(return_value=None)
        with pytest.raises(UnlimitedWaiterError):
            Wait(action_mock, timeout=0, interval=0, max_attempts=0)

    @staticmethod
    @pytest.mark.parametrize("exception", (RuntimeError, Exception))
    def test_exceptions_to_ignore(mocker: MockFixture, exception: type[Exception]) -> None:
        action_mock = mocker.Mock(side_effect=(exception, True))
        Wait(action_mock, timeout=1, interval=0, exceptions_to_ignore=(exception,)).until()

    @staticmethod
    def test_exceptions_to_ignore_negative(mocker: MockFixture) -> None:
        action_mock = mocker.Mock(side_effect=(RuntimeError, True))

        with pytest.raises(RuntimeError):
            Wait(action_mock, timeout=1, interval=0, exceptions_to_ignore=(ValueError,)).until()

    @staticmethod
    def test_interval(mocker: MockFixture) -> None:
        interval, attempts = 0.04, 3
        action_mock = mocker.Mock(return_value=False)
        start_time = time.perf_counter()
        with pytest.raises(WaiterTimeoutError):
            Wait(action_mock, timeout=0, max_attempts=attempts, interval=interval).until()

        assert 0 <= time.perf_counter() - start_time - interval * attempts < MAX_THRESHOLD

    @staticmethod
    @pytest.mark.parametrize(
        ("interval", "attempts", "total_interval"),
        (
            (0.03, 3, 0.21),
            (0.04, 2, 0.12),
        ),
    )
    def test_exponential_interval(mocker: MockFixture, interval: float, attempts: int, total_interval: float) -> None:
        action_mock = mocker.Mock(return_value=False)
        start_time = time.perf_counter()
        with pytest.raises(WaiterTimeoutError):
            Wait(action_mock, timeout=0, max_attempts=attempts, interval=interval, is_exponential=True).until()

        assert 0 <= time.perf_counter() - start_time - total_interval < MAX_THRESHOLD

    @staticmethod
    @pytest.mark.parametrize(
        ("interval", "attempts", "max_interval", "total_interval"),
        (
            (0.03, 3, 0.05, 0.13),
            (0.04, 2, 0.06, 0.1),
        ),
    )
    def test_exponential_max_interval(
        mocker: MockFixture, interval: float, attempts: int, max_interval: float, total_interval: float
    ) -> None:
        action_mock = mocker.Mock(return_value=False)
        start_time = time.perf_counter()
        with pytest.raises(WaiterTimeoutError):
            Wait(
                action_mock,
                timeout=0,
                max_attempts=attempts,
                interval=interval,
                is_exponential=True,
                max_interval=max_interval,
            ).until()

        assert 0 <= time.perf_counter() - start_time - total_interval < MAX_THRESHOLD

    @staticmethod
    def test_unused_max_interval(mocker: MockFixture) -> None:
        action_mock = mocker.Mock(return_value=False)
        with pytest.raises(UnusedMaxIntervalError):
            Wait(action_mock, timeout=0.01, is_exponential=False, max_interval=0.01)

    @staticmethod
    @pytest.mark.parametrize("timeout_message", ("", "message"))
    @pytest.mark.parametrize("debug", (True, False))
    def test_timeout_message(mocker: MockFixture, timeout_message: str, debug: bool) -> None:
        results = [1, "2", True]
        results_count = len(results)
        expected_message = f"{timeout_message}\n" if timeout_message else ""
        expected_message = f"{expected_message}Waiter timeout after {results_count} action calls"
        if debug:
            expected_message = f"{expected_message}\nResults: {results}"
        expected_message = rf"^{re.escape(expected_message)}$"
        action_mock = mocker.Mock(side_effect=results)
        waiter = Wait(
            action_mock, timeout=0, interval=0, max_attempts=results_count, timeout_message=timeout_message, debug=debug
        )
        with pytest.raises(WaiterTimeoutError, match=expected_message):
            waiter.until_is_false()

    @staticmethod
    def test_calls_with_no_debug(mocker: MockFixture) -> None:
        results = [0, 1, "", "1", (), (1,), False, None]
        action_mock = mocker.Mock(side_effect=results)
        waiter = Wait(action_mock, timeout=1, interval=0, debug=False)
        waiter.until_is_none()
        assert waiter._results is None
        assert waiter._calls_count == len(results)

    @staticmethod
    def test_calls_with_debug(mocker: MockFixture) -> None:
        expected_results = [0, 1, "", "1", (), (1,), False, None]
        action_mock = mocker.Mock(side_effect=expected_results)
        waiter = Wait(action_mock, timeout=1, interval=0, debug=True)
        waiter.until_is_none()
        assert waiter._results == expected_results
        assert waiter._calls_count == len(expected_results)

        expected_results = [True, None]
        action_mock.side_effect = expected_results
        waiter.until_is_none()
        assert waiter._results == expected_results
        assert waiter._calls_count == len(expected_results)

    @staticmethod
    @pytest.mark.parametrize("value", (1, "1", (1,), True))
    def test_until(mocker: MockFixture, value: Any) -> None:
        expected_call_count = 2
        action_mock = mocker.Mock(side_effect=(None, value))
        result = Wait(action_mock, timeout=1, interval=0).until()
        assert result == value
        assert action_mock.call_count == expected_call_count

    @staticmethod
    def test_until_predicate(mocker: MockFixture) -> None:
        expected_call_count = 2
        expected_result = 4
        action_mock = mocker.Mock(side_effect=(2, 4))
        result = Wait(action_mock, timeout=1, interval=0).until(predicate=lambda x: x > 3)  # noqa: PLR2004
        assert result == expected_result
        assert action_mock.call_count == expected_call_count

    @staticmethod
    @pytest.mark.parametrize("value", (0, "", (), False, None))
    def test_until_negative(mocker: MockFixture, value: Any) -> None:
        expected_call_count = 2
        action_mock = mocker.Mock(side_effect=(value, True))
        result = Wait(action_mock, timeout=1, interval=0).until()
        assert result is True
        assert action_mock.call_count == expected_call_count

    @staticmethod
    @pytest.mark.parametrize("value", (0, "", (), False, None))
    def test_until_not(mocker: MockFixture, value: Any) -> None:
        expected_call_count = 2
        action_mock = mocker.Mock(side_effect=(True, value))
        result = Wait(action_mock, timeout=1, interval=0).until_not()
        assert result == value
        assert action_mock.call_count == expected_call_count

    @staticmethod
    @pytest.mark.parametrize("value", (1, "1", (1,), True))
    def test_until_not_negative(mocker: MockFixture, value: Any) -> None:
        expected_call_count = 2
        action_mock = mocker.Mock(side_effect=(value, None))
        result = Wait(action_mock, timeout=1, interval=0).until_not()
        assert result is None
        assert action_mock.call_count == expected_call_count

    @staticmethod
    @pytest.mark.parametrize("value", (0, 1, "", "1", (), (1,), False, None))
    def test_until_equal_to(mocker: MockFixture, value: Any) -> None:
        expected_value = 10
        expected_call_count = 2
        action_mock = mocker.Mock(side_effect=(value, expected_value))
        result = Wait(action_mock, timeout=1, interval=0).until_equal_to(expected_value)
        assert result == expected_value
        assert action_mock.call_count == expected_call_count

    @staticmethod
    @pytest.mark.parametrize("value", (0, 1, "", "1", (), (1,), False, None))
    def test_until_not_equal_to(mocker: MockFixture, value: Any) -> None:
        not_expected_value = 10
        action_mock = mocker.Mock(side_effect=(not_expected_value, value))
        expected_call_count = 2
        result = Wait(action_mock, timeout=1, interval=0).until_not_equal_to(not_expected_value)
        assert result == value
        assert action_mock.call_count == expected_call_count

    @staticmethod
    @pytest.mark.parametrize("value", (True, False, None))
    def test_until_is(mocker: MockFixture, value: Any) -> None:
        not_expected_value = not value
        action_mock = mocker.Mock(side_effect=(not_expected_value, value))
        expected_call_count = 2
        result = Wait(action_mock, timeout=1, interval=0).until_is(value)
        assert result is value
        assert action_mock.call_count == expected_call_count

    @staticmethod
    @pytest.mark.parametrize("value", (1, False, None))
    def test_until_is_negative(mocker: MockFixture, value: Any) -> None:
        expected_call_count = 2
        action_mock = mocker.Mock(side_effect=(value, True))
        result = Wait(action_mock, timeout=1, interval=0).until_is(True)  # noqa: FBT003
        assert result is True
        assert action_mock.call_count == expected_call_count

    @staticmethod
    @pytest.mark.parametrize("value", (True, False, None))
    def test_until_is_not(mocker: MockFixture, value: Any) -> None:
        expected_value = not value
        action_mock = mocker.Mock(side_effect=(value, expected_value))
        expected_call_count = 2
        result = Wait(action_mock, timeout=1, interval=0).until_is_not(value)
        assert result is expected_value
        assert action_mock.call_count == expected_call_count

    @staticmethod
    @pytest.mark.parametrize("value", (1, True, None))
    def test_until_is_not_negative(mocker: MockFixture, value: Any) -> None:
        expected_call_count = 2
        action_mock = mocker.Mock(side_effect=(False, value))
        result = Wait(action_mock, timeout=1, interval=0).until_is_not(False)  # noqa: FBT003
        assert result == value
        assert action_mock.call_count == expected_call_count

    @staticmethod
    @pytest.mark.parametrize("value", (0, 1, "", "1", (), (1,), False, None))
    def test_until_true(mocker: MockFixture, value: Any) -> None:
        action_mock = mocker.Mock(side_effect=(value, True))
        expected_call_count = 2
        result = Wait(action_mock, timeout=1, interval=0).until_is_true()
        assert result is True
        assert action_mock.call_count == expected_call_count

    @staticmethod
    @pytest.mark.parametrize("value", (0, 1, "", "1", (), (1,), None, True))
    def test_until_false(mocker: MockFixture, value: Any) -> None:
        expected_call_count = 2
        action_mock = mocker.Mock(side_effect=(value, False))
        result = Wait(action_mock, timeout=1, interval=0).until_is_false()
        assert result is False
        assert action_mock.call_count == expected_call_count

    @staticmethod
    @pytest.mark.parametrize("value", (0, 1, "", "1", (), (1,), False, True))
    def test_until_none(mocker: MockFixture, value: Any) -> None:
        expected_call_count = 2
        action_mock = mocker.Mock(side_effect=(value, None))
        result = Wait(action_mock, timeout=1, interval=0).until_is_none()  # type: ignore[func-returns-value]
        assert result is None
        assert action_mock.call_count == expected_call_count

    @staticmethod
    @pytest.mark.parametrize("value", (0, 1, "", "1", (), (1,), False, True))
    def test_until_not_none(mocker: MockFixture, value: Any) -> None:
        expected_call_count = 2
        action_mock = mocker.Mock(side_effect=(None, value))
        result = Wait(action_mock, timeout=1, interval=0).until_is_not_none()
        assert result == value
        assert action_mock.call_count == expected_call_count
