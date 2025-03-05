#!/bin/sh
#BSUB -J train_VAE[1,2,3,4,5] # could also be doe as train_VAE[1-5]
#BSUB -o VAE/batch_scripts/logs/Train%J_%I.out
#BSUB -e VAE/batch_scripts/logs/Train%J_%I.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 16 
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2G]"
#BSUB -W 0:20
#BSUB -N 
# end of BSUB options

module load cuda/11.8

source ../venv/bin/activate # change to your virtual environment if needed

# adjust the arguments as needed
python train_VAE.py --config gaus --seed $LSB_JOBINDEX