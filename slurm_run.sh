#!/bin/bash
#SBATCH --job-name=sdfstudio                   # Job name
#SBATCH --mail-type=END,FAIL                   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=james.gardner@york.ac.uk   # Where to send mail
#SBATCH --ntasks=1                             # Run a single task...
#SBATCH --cpus-per-task=16                     # ...with a single CPU
#SBATCH --mem=24gb                             # Job memory request
#SBATCH --time=06:00:00                        # Time limit hrs:min:sec
#SBATCH --output=outputs/SLURM/cuda_job_%j.log # Standard output and error log
#SBATCH --partition=gpu                        # Select the GPU nodes...
#SBATCH --gres=gpu:1                           # ...and the Number of GPUs

module purge # clear any inherited modules

export CUDA_VISIBLE_DEVICES=1
 
echo `date`: executing sdfstudio/run.sh on host $HOSTNAME with $SLURM_CPUS_ON_NODE cpu cores
echo
cudaDevs=$(echo $CUDA_VISIBLE_DEVICES | sed -e 's/,/ /g')
echo I can see GPU devices $CUDA_VISIBLE_DEVICES
echo
 
source ~/.bashrc

# Arguments
CONTAINER=".devcontainer/sdfstudio.sif"
BIND_USER_SCRATCH="/mnt/scratch/users/$USER:/users/$USER/scratch"
BIND_USER_CONFIG="/users/$USER/scratch/.config:/users/$USER/.config"
BIND_USER_LOCAL="/users/$USER/scratch/.local:/users/$USER/.local"
BIND_VSCODE_SERVER="/users/$USER/.vscode-server:/users/$USER/.vscode-server"
MPLCONFIGDIR="/users/$USER/scratch/.config/matplotlib"
WANDB__SERVICE_WAIT=2000
COMMAND="ns-train neus-facto --vis wandb nerfosr-data --scene lk2 --use-session-data False"

# Command
singularity exec \
  --nv \
  --no-home \
  -B "$BIND_USER_SCRATCH" \
  -B "$BIND_USER_CONFIG" \
  -B "$BIND_USER_LOCAL" \
  -B "$BIND_VSCODE_SERVER" \
  --env MPLCONFIGDIR="$MPLCONFIGDIR" \
  --env WANDB__SERVICE_WAIT=$WANDB__SERVICE_WAIT \
  --env WANDB_MODE=offline \
  "$CONTAINER" \
  $COMMAND