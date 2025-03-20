#!/bin/bash
#SBATCH --nodelist=gn01
#SBATCH --nodes=1  
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=mnist_fsdp
#SBATCH --error=mnist_fsdp.err
#SBATCH --output=mnist_fsdp.log

module load slurm

eval "$(conda shell.bash hook)"
conda activate llm

# ottieni il master node e il suo ip
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)    
head_node=${nodes_array[0]}

#Log NCCL
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export LOGLEVEL=INFO

#INSERT YOUR SCRIPT HERE
export MASTER_ADDR=$head_node
export MASTER_PORT=29500
export WORLD_SIZE=4

# Configura NCCL per InfiniBand (forse non servono)
export NCCL_IB_HCA=mlx5_0,1
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0


# Avvia l'allenamento distribuito
srun torchrun --nnodes=1 --nproc_per_node=2 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    fsdp_mnist.py