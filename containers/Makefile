.PHONY: docbuild
docbuild:
	docker build -t nintorch -f Dockerfile .

.PHONY: docrun
docrun:
	docker run -it --rm --gpus all nintorch

.PHONY: sinbuild
sinbuild: torch.sif

torch.sif: torch.def
	sudo singularity build $@ $<

.PHONY: clean
clean:
	rm -f torch.sif
