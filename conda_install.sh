#!/bin/bash
#SBATCH --job-name=sdfstudio                   # Job name
#SBATCH --mail-type=END,FAIL                   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=james.gardner@york.ac.uk   # Where to send mail
#SBATCH --ntasks=1                             # Run a single task...
#SBATCH --cpus-per-task=1                      # ...with a single CPU
#SBATCH --mem=16gb                             # Job memory request
#SBATCH --time=01:00:00                        # Time limit hrs:min:sec
#SBATCH --output=cuda_job_%j.log               # Standard output and error log
#SBATCH --partition=gpu_big                    # Select the GPU nodes...
#SBATCH --gres=gpu:1    