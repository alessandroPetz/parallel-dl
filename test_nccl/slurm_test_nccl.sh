#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=test_nccl
#SBATCH --output=test_nccl.log
#SBATCH --error=test_nccl.err

module load slurm
eval "$(conda shell.bash hook)"
conda activate Parallel-DL  # Cambia con il tuo env

# Trova il nodo master e il suo IP su InfiniBand
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
master_node=${nodes[0]}
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$master_node" ip -o -4 addr show ib0 | awk '{print $4}' | cut -d'/' -f1)
export MASTER_PORT=29500
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO

# Imposta variabili per torch.distributed
#export WORLD_SIZE=$SLURM_NTASKS
#export RANK=$SLURM_PROCID
#export LOCAL_RANK=$SLURM_LOCALID

# Avvia il test
srun torchrun /home/apetrella/Workspace/parallel-dl/test_nccl/test_nccl.py