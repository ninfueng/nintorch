.PHONY: train-cifar10
train-cifar10:
	python train.py --half --chl-last --compile --ema

.PHONY: train-imagenet
train-imagenet:
	python train.py --half --chl-last --compile --dataset imagenet --model-name resnet18 --ema

.PHONY: clean
clean:
	rm -rf ./dist/
	rm -rf ./build/
	rm -rf ./nintorch.egg-info/
	find . -iname wandb | xargs rm -rf
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
	rm -rf datasets/
	rm -rf typings/
	rm -f subset_idx.pt

.PHONY: fmt
fmt:
	# https://github.com/PyCQA/isort/issues/1632
	# https://black.readthedocs.io/en/stable/guides/using_black_with_other_tools.html
	find -iname "*.py" | xargs pyupgrade
	isort . \
		--skip __init__.py \
		--line-length 88 \
		--profile black \
		--multi-line 3 \
		--trailing-comma \
		--force-grid-wrap 0 \
		--use-parentheses \
		--ensure-newline-before-comments \
		--filter-files
	black . \
		--line-length 88 \
		--exclude ./exps \
		--target-version py312 \
		--skip-string-normalization

.PHONY: fmtstr
fmtstr:
	find -iname "*.py" | xargs sed -i s/\"/\'/g
	find -iname "*.py" | xargs sed -i s/\'\'\'/\"\"\"/g

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

.PHONY: mypy
mypy:
	mypy . --disable-error-code=import-untyped
