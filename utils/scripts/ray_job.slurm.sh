#!/bin/bash

# SBATCH --job-name=ray-rllib-job
# SBATCH --cpus-per-task=4
# SBATCH --nodes=2
# SBATCH --tasks-per-node=1
# SBATCH --time=01:00:00
# SBATCH --partition=your_partition
# SBATCH --output=ray-rllib-%j.out

# 加载 Python 环境和 Ray
module load python/3.7  # 假设你使用 module 系统
source activate your_virtualenv  # 激活你的虚拟环境

# 获取节点列表
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

head_node=${nodes_array[0]}
port=6379
ip_prefix=$(srun --nodes=1 --ntasks=1 -w $head_node hostname --ip-address)
if [[ $ip_prefix == *"."* ]]; then
    # IPV4
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w $head_node hostname --ip-address)
else
    # IPV6
    head_node_ip="[$(srun --nodes=1 --ntasks=1 -w $head_node hostname --ip-address)]"
fi

export head_node
export port
export head_node_ip

# 启动 Ray head 节点
srun --nodes=1 --ntasks=1 -w $head_node ray start --head --node-ip-address=$head_node_ip --port=$port --block &

# 其余节点作为 worker 节点加入
for ((  i=1; i<${#nodes_array[@]}; i++ ))
do
    node_i=${nodes_array[$i]}
    srun --nodes=1 --ntasks=1 -w $node_i ray start --address=$head_node_ip:$port --block &
done

# 等待节点启动
sleep 30

# 执行 RLLib 训练任务
python your_training_script.py

# 停止 Ray 集群
ray stop
