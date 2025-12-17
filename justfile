set dotenv-load := true

TEST_PYPI_TOKEN := env('TEST_PYPI_TOKEN', '')
PYPI_TOKEN := env('PYPI_TOKEN', '')
python := justfile_directory() / ".venv" / "bin" / "python"

# List all available recipes
default:
  just --list
  @echo "To execute a recipe: just [recipe-name]"

# Format with Black
black: check-uv
  uv run black {{justfile_directory()}}/src
  # uv run black {{justfile_directory()}}/tests

# Run `uv build`
build *BUILD_ARGS: check-uv
  rm -rf {{justfile_directory()}}/dist/*
  uv build --project {{justfile_directory()}} {{BUILD_ARGS}}

# # Build the website docs with quartodoc
# [working-directory: "docs"]
# build-docs: check-uv
#   uv run quartodoc build
#   uv run quartodoc interlinks
#   uv run quarto render

# # Check dependencies are installed
# check-dependencies: check-uv check-readme-dependencies
#   which pytest
#   which black

# Check if git and GH cli are installed
check-github:
  which git
  which gh

# Check quarto-cli and jupyter is installed
check-readme-dependencies:
  which quarto
  which jupyter

# Check uv is installed
check-uv:
  @which uv

# Publish a package release on GitHub
publish-github: check-uv check-github
  #!/usr/bin/env zsh
  VERSION=$(uv run python -c "import fulgur; print(fulgur.__version__)")
  if git ls-remote --tags origin | grep -q "refs/tags/v$VERSION"; then
    echo "Release v$VERSION already exists."
    exit 1
  fi
  git tag v$VERSION
  git push origin v$VERSION
  gh release create v$VERSION dist/* \
    --title "Release v$VERSION" \
    --notes "New release of version $VERSION."

# Publish the package to PyPI
publish-pypi: check-uv
  uv publish --project {{justfile_directory()}} --token {{PYPI_TOKEN}}

# Publish the package to TestPyPI
publish-testpypi: check-uv
  uv publish --publish-url https://test.pypi.org/legacy/ --project {{justfile_directory()}} --token {{TEST_PYPI_TOKEN}}

# Run tests with PyTest
# test *TEST_ARGS: check-uv
#   uv run pytest {{TEST_ARGS}}