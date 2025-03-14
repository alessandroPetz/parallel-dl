#!/bin/bash
#SBATCH --nodes=2                 # 4 nodi
#SBATCH --ntasks-per-node=1       # 1 task per nodo
#SBATCH --gres=gpu:2              # 2 GPU per nodo
#SBATCH --cpus-per-task=1         # 1 CPU per task
#SBATCH --job-name=check_gpus
#SBATCH --output=gpu_check.log
#SBATCH --error=gpu_check.error

module load slurm  # Assicurati che SLURM sia caricato

# Comando per vedere tutte le GPU disponibili su ogni nodo
echo "Running on nodes: $SLURM_JOB_NODELIST"

srun --nodes=2 --ntasks=2 bash -c 'echo "Node $(hostname): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"'
srun --nodes=2 --ntasks=2 nvidia-smi