# Makefile
SHELL := /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "uncertainty-estimates : list of main operations."
	@echo "venv    : creates and activates development environment."
	@echo "install : Installs all the required packages"
	@echo "venv-install : Creates venv and installs all the packages inside it"
	@echo "format   : formats code using black."
	@echo "style : styles the code."
	@echo "clean   : cleans all unnecessary files."
	@echo "test    : run non-training tests."


# Environment
.ONESHELL:
venv:
	conda create -n uncertainty-estimates python=3.9 &&\
	conda activate uncertainty-estimates


.PHONY: install
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

.PHONY: venv-install
venv-install: venv install


# Styling
.PHONY: style
style:
	black .
	flake8
	isort .


# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage


.PHONY: test
test:
	python -m pytest -vv --cov=cli --cov=mlib --cov=utilscli test_mlib.py

.PHONY: format
format:
	black *.py

.PHONY: lint
lint:
	#pylint --disable=R,C,W1203,E1101 mlib cli utilscli
	#lint Dockerfile
	#docker run --rm -i hadolint/hadolint < Dockerfile

.PHONY: all
all: install lint test