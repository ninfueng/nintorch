.PHONY: traincifar10
traincifar10:
	python train.py --fp16

.PHONY: clean
clean:
	rm -rf ./wandb/
	rm -rf ./dist/
	rm -rf ./build/
	rm -rf ./nintorch.egg-info/
	find . -iname .vscode | xargs rm -rf
	find . -iname __pycache__ | xargs rm -rf
	find . -iname .pytest_cache | xargs rm -rf
	find . -iname .mypy_cache | xargs rm -rf
	find . -iname .ipynb_checkpoints | xargs rm -rf

.PHONY: cleanall
cleanall: clean
	rm -f *.csv
	rm -f *.pkl
	rm -rf exps/
	rm -rf downloads/
	rm -rf typings/
	rm -f subset_idx.pt

.PHONY: fmt
fmt:
	# https://github.com/PyCQA/isort/issues/1632
	isort . \
		--skip __init__.py \
		--profile black \
		--verbose
	black . \
		--line-length 120 \
		--exclude ./exps


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
