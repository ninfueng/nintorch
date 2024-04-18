.PHONY: docbuild
docbuild:
	docker build -t nintorch -f Dockerfile .

.PHONY: docrun
docrun: docrun
	docker run -it --rm --gpus all nintorch

.PHONY: sinbuild
sinbuild: torch.sif

torch.sif: torch.def
	# https://github.com/apptainer/singularity/issues/5791
	# `-E` for using user environment variables
	mkdir -p ~/tmp && \
	export SINGULARITY_TMPDIR=~/tmp && \
	sudo -E singularity build $@ $< && \
	sudo rm -rf ~/tmp

.PHONY: clean
clean:
	rm -f torch.sif
