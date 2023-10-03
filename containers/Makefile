.PHONY: docker_build
docker_build:
	docker build -t nintorch -f Dockerfile .

.PHONY: docker_run
docker_run:
	sudo docker run -it --rm --gpus all nintorch

.PHONY: sing_build
sing_build: torch.sif

torch.sif: torch.def
	sudo singularity build $@ $<

.PHONY: clean
clean:
	rm -f torch.sif
