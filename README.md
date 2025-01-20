# air-waiter

Call a callable until the expected result. Raises TimeoutError if limit is reached. Waiter can be limited by timeout or/and by maximal calls count.

## Install

```sh
# With uv
uv add air-waiter

# With poetry
poetry add air-waiter

# With pip
pip install air-waiter
```

## Usage

### Wait untill the action returns the expected value in timeout

```sh
Wait(action, timeout=10, interval=0.1).until(lambda x: check_action(x))
```

### Wait untill the action returns the expected value with limited attempts

```sh
Wait(action, timeout=0, max_attempts=5, interval=0.1).until(lambda x: check_action(x))
```

### Wait untill the action returns the expected value, doubling the interval after each attempt

```sh
Wait(action, timeout=10, is_exponential=True, interval=0.01).until(lambda x: check_action(x))
```
