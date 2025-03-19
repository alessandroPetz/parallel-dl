#!/bin/bash
### #SBATCH --nodelist=gn01,gn02
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --job-name=fsdp-mnist
#SBATCH --error=fsdp-mnist.error
#SBATCH --output=fsdp-mnist.log
#---------------------------------------------------------------------------------------

module load slurm

eval "$(conda shell.bash hook)"
conda activate Parallel-DL

# ottieni il master node e il suo ip
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)    
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_node_ip_infiniband=$(srun --nodes=1 --ntasks=1 -w "$head_node" ip -o -4 addr show ib0 | awk '{print $4}' | cut -d'/' -f1)  # con infiniband

# Log NCCL
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export LOGLEVEL=INFO

#INSERT YOUR SCRIPT HERE
# senza infiniband
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1   # Disabilita InfiniBand
export NCCL_P2P_DISABLE=0  # Mantiene il supporto per P2P over PCIe
export NCCL_SHM_DISABLE=0  # Abilita l'uso della shared memory 


# Configura NCCL per InfiniBand
#export MASTER_ADDR=$head_node_ip_infiniband
#export MASTER_PORT=29500
#export NCCL_SOCKET_IFNAME=ib0
#export NCCL_IB_DISABLE=0
#export NCCL_P2P_DISABLE=0
#export NCCL_SHM_DISABLE=0


echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MASTER_ADDR=$MASTER_ADDR" > ddp-multigpu.output
echo "SLURM_JOB_ID=$SLURM_JOB_ID" >> ddp-multigpu.output
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> ddp-multigpu.output
echo "Node ID: $SLURM_NODEID, Process ID: $SLURM_PROCID" >> ddp-multigpu.output

python main_FSDP_mnist.py