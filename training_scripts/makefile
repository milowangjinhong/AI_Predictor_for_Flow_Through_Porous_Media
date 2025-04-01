all:
	qsub -q hx submit_all.pbs

ae:
	qsub -q hx submit_ae.pbs

fno:
	qsub -q hx submit_fno.pbs

unet:
	qsub -q hx submit_unet.pbs

ae_sweep:
	qsub -q hx submit_ae_sweep.pbs

fno_sweep:
	qsub -q hx submit_fno_sweep.pbs

unet_sweep:
	qsub -q hx submit_unet_sweep.pbs

run_ae:
	python train_ae.py

run_fno:
	python train_fno.py

run_unet:
	python train_unet.py

run_ae_sweep:
	python train_ae_sweep.py

run_fno_sweep:
	python train_fno_sweep.py

run_unet_sweep:
	python train_unet_sweep.py

run_vae:
	python train_vae.py

run_all:
	python train_ae.py
	python train_unet.py
	# python train_fno.py
	# python train_vae.py

wandb_clean:
	wandb sync --clean