#!/bin/bash
#SBATCH --job-name=sdfstudio                   # Job name
#SBATCH --mail-type=END,FAIL                   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=james.gardner@york.ac.uk   # Where to send mail
#SBATCH --ntasks=1                             # Run a single task...
#SBATCH --cpus-per-task=1                      # ...with a single CPU
#SBATCH --mem=16gb                             # Job memory request
#SBATCH --time=01:00:00                        # Time limit hrs:min:sec
#SBATCH --output=cuda_job_%j.log               # Standard output and error log
#SBATCH --account=cs-dclabs-2019               # Project account
#SBATCH --partition=gpu                        # Select the GPU nodes...
#SBATCH --gres=gpu:1                           # ...and the Number of GPUs

module purge
module load lang/Miniconda3
module load system/CUDA/11.3.1
module load compiler/GCC/9.3.0 # for CUDA 11.3 - 5.0.0 < GCC < 10.0.0
 
echo `date`: executing gpu_test on host $HOSTNAME with $SLURM_CPUS_ON_NODE cpu cores
echo
cudaDevs=$(echo $CUDA_VISIBLE_DEVICES | sed -e 's/,/ /g')
echo I can see GPU devices $CUDA_VISIBLE_DEVICES
echo
 
source ~/.bashrc

# singularity exec .devcontainer/sdfstudio_singularity.sif ns-download-data sdfstudio

singularity exec --nv .devcontainer/sdfstudio_singularity.sif python3 scripts/train.py neus-facto --pipeline.model.sdf-field.inside-outside False --vis viewer --experiment-name neus-facto-dtu65 sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65 --auto-orient True

# find_in_conda_env(){
#     conda env list | grep "${@}" >/dev/null 2>/dev/null
# }

# if find_in_conda_env ".*sdfstudio.*" ; then
#   conda activate sdfstudio
# else 
#   conda create --name sdfstudio -y python=3.8
#   conda activate sdfstudio

#   python -m pip install --upgrade pip

#   conda install -y -c conda-forge cudatoolkit-dev==11.3.1

#   pip install open3d==0.16.1
#   pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
#   pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

#   pip install --upgrade pip setuptools
#   pip install -e .
#   # install tab completion
#   ns-install-cli
# fi

# # Download some test data: you might need to install curl if your system don't have that
# ns-download-data sdfstudio

# # Train model on the dtu dataset scan65
# ns-train neus-facto --pipeline.model.sdf-field.inside-outside False --vis viewer --experiment-name neus-facto-dtu65 sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65
