#!/bin/bash
#SBATCH --nodelist=gn01,gn02
#SBATCH --nodes=2  
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --job-name=ddp-multigpu
#SBATCH --error=ddp-multigpu.error
#SBATCH --output=ddp-multigpu.log
#---------------------------------------------------------------------------------------

# 4 nodi, 4 task, ogni task 2 gpu. il task e il numero di nodi deve essere lo stesso....

module load slurm

eval "$(conda shell.bash hook)"
conda activate Parallel-DL

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

echo "MASTER_ADDR=$MASTER_ADDR" > ddp-multigpu.output
echo "MASTER_PORT=$MASTER_PORT" >> ddp-multigpu.output
echo "SLURM_JOB_ID=$SLURM_JOB_ID" >> ddp-multigpu.output
echo "Node ID: $SLURM_NODEID, Process ID: $SLURM_PROCID" >> ddp-multigpu.output

srun torchrun  \
    --nnodes=2 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    multinode-torchrun.py 5000 100