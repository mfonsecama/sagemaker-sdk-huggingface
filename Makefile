.PHONY: quality style test test-examples

# Check that source code meets quality standards
check_dirs := src


quality:
	black --check --line-length 119 --target-version py36  $(check_dirs) 
	isort --check-only  $(check_dirs) 
	flake8  $(check_dirs) 

# Format source code automatically

style:
	black --line-length 119 --target-version py36  $(check_dirs)
	isort  $(check_dirs)

# Run tests for the library

test:
	pytest -n auto --dist=loadfile -s -v ./tests/
