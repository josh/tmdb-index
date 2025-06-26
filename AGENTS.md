# Agents Guide

This project uses Python 3.12 and manages dependencies with `uv`.

## Setup

Install Python 3.12 or newer.

Install `uv` with:

```sh
$ curl -LsSf https://astral.sh/uv/install.sh | sh
# or
$ pipx install uv
```

Then install dependencies with:

```sh
$ uv sync
```

Note that when running in an offline sandbox, you may need to run uv with the `--offline` flag when calling `uv run`.

## Testing

Check code style with ruff:

```sh
$ uv tool run ruff format --diff .
$ uv tool run ruff check .
```

Check type correctness with mypy:

```sh
$ uv tool run mypy .
```

Run the test suite with:

```sh
$ uv run pytest
```

## Formatting

You can automatically fix most formatting issues with:

```sh
$ uv tool run ruff format .
```

Functions should be sorted in dependency order with:

```sh
$ uv tool run ssort .
```

## Code Quality

You can generate a code coverage report while running the test suite with:

```sh
$ uv run pytest --cov=tmdb_index --cov-report=term-missing .
```

Optionally scan for dead code with vulture:

```sh
$ uv tool run vulture --exclude .venv/ .
```
