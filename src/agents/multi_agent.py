# multi_agent.py

import random
import time
import threading
import redis
from stable_baselines3 import PPO

# 初始化Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# 智能体A：环境感知和初步决策
def agent_a():
    while True:
        # 模拟环境感知
        state = random.randint(0, 100)
        redis_client.set('state', state)
        print(f"Agent A sensed state: {state}")
        time.sleep(1)

# 智能体B：用户交互
def agent_b():
    while True:
        state = int(redis_client.get('state'))
        # 模拟用户交互和反馈
        user_action = f"User action based on state {state}"
        redis_client.set('user_action', user_action)
        print(f"Agent B processed user action: {user_action}")
        time.sleep(2)

# 智能体C：策略规划和任务分配
def agent_c():
    model = PPO('MlpPolicy', 'CartPole-v1', verbose=1)
    model.learn(total_timesteps=10000)
    
    while True:
        state = int(redis_client.get('state'))
        user_action = redis_client.get('user_action').decode('utf-8')
        
        # 模拟任务规划和分配
        task = f"Task for state {state} and action {user_action}"
        redis_client.set('task', task)
        print(f"Agent C assigned task: {task}")
        time.sleep(3)

# 启动多智能体系统
threading.Thread(target=agent_a).start()
threading.Thread(target=agent_b).start()
threading.Thread(target=agent_c).start()
