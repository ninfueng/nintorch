.PHONY: clean
clean:
	rm -rf wandb/
	rm -rf dist/
	rm -rf build/
	rm -rf nintorch.egg-info/
	find . -iname .vscode | xargs rm -rf
	find . -iname __pycache__ | xargs rm -rf
	find . -iname .pytest_cache | xargs rm -rf
	find . -iname .mypy_cache | xargs rm -rf
	find . -iname .ipynb_checkpoints | xargs rm -rf

.PHONY: cleanall
cleanall: clean
	rm -f *.csv
	rm -f *.pkl
	rm -rf experiments/
	rm -rf downloads/
	rm -rf typings/
	rm -f subset_idx.pt

.PHONY: fmt
fmt:
	isort --verbose . --skip __init__.py
	black . --exclude ./experiments

.PHONY: pip
pip:
	python setup.py sdist
	twine upload dist/*

.PHONY: install
install:
	python setup.py develop

.PHONY: test
test:
	pytest -vls -m "not slow" ./tests

.PHONY: testall
testall:
	pytest ./tests
