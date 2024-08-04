# multi_agent.py

import random
import time
import threading
import redis
import os
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import ray
from stable_baselines3 import PPO
from textblob import TextBlob
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import SGDClassifier
import numpy as np

# Redis客户端类
class RedisClient:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db)

    def set_value(self, key, value):
        self.client.set(key, value)

    def get_value(self, key):
        return self.client.get(key)

# 智能体基类
class BaseAgent:
    def __init__(self, redis_client):
        self.redis_client = redis_client

# 智能体A：环境感知和初步决策
class AgentA(BaseAgent):
    def run(self):
        while True:
            state = random.randint(0, 100)
            self.redis_client.set_value('state', state)
            print(f"Agent A sensed state: {state}")
            time.sleep(1)

# 智能体B：用户交互
class AgentB(BaseAgent):
    def run(self):
        while True:
            state = int(self.redis_client.get_value('state'))
            user_action = f"User action based on state {state}"
            self.redis_client.set_value('user_action', user_action)
            print(f"Agent B processed user action: {user_action}")
            time.sleep(2)

# 智能体C：策略规划和任务分配
class AgentC(BaseAgent):
    def __init__(self, redis_client):
        super().__init__(redis_client)
        self.model = PPO('MlpPolicy', 'CartPole-v1', verbose=1)
        self.model.learn(total_timesteps=10000)

    def run(self):
        while True:
            state = int(self.redis_client.get_value('state'))
            user_action = self.redis_client.get_value('user_action').decode('utf-8')
            task = f"Task for state {state} and action {user_action}"
            self.redis_client.set_value('task', task)
            print(f"Agent C assigned task: {task}")
            time.sleep(3)

# NLP智能体：情感分析与反馈循环
class NlpAgent:
    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def feedback_loop(self, user_feedback, text):
        print(f"Received user feedback: {user_feedback} for text: {text}")

# 预训练模型智能体类
class PretrainedModelAgent:
    def __init__(self, model_path, tokenizer_path, device):
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def run(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        outputs = self.model.generate(inputs['input_ids'], max_length=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# SGD模型处理类
class SgdModelHandler:
    def __init__(self, X_train, y_train):
        self.model = SGDClassifier()
        self.classes = np.unique(y_train)
        self.model.partial_fit(X_train, y_train, classes=self.classes)

    def partial_fit(self, X_partial, y_partial):
        self.model.partial_fit(X_partial, y_partial, classes=self.classes)

# 分布式智能体类（基于Ray）
@ray.remote
class CharacterAgent:
    def __init__(self, character_name, model_path='path_to_your_rasa_model'):
        try:
            self.agent = Agent.load(model_path)
            self.character_name = character_name
            logging.info(f"Initialized agent for {character_name}")
        except Exception as e:
            logging.error(f"Error initializing agent for {character_name}: {e}")
            raise

    def handle_message(self, message):
        try:
            response = self.agent.handle_text(message)
            return response
        except Exception as e:
            logging.error(f"Error handling message '{message}' for {self.character_name}: {e}")
            return {"error": str(e)}

    def perform_task(self, task):
        try:
            logging.info(f"Python Agent {self.character_name} is performing task: {task}")
            processor = TaskProcessor(task)
            result = processor.process()
            return result
        except Exception as e:
            logging.error(f"Error performing task '{task}' for {self.character_name}: {e}")
            return {"error": str(e)}

class TaskProcessor:
    def __init__(self, task):
        self.task = task

    def process(self):
        return f"Processed: {self.task}"

# KamenRiderBlade多功能智能体类
class KamenRiderBladeAgent:
    def __init__(self, base_path, config_path='config.json'):
        self.base_path = base_path
        self.config = self._load_config(config_path)
        self.model_config = self.config['model_config']
        self.project_config = self.config['project_config']
        self.character_agents = {}
        self.project_structure = self.project_config.get('project_structure', {})
        self._init_logging()
        self._setup_routes()
        ray.init()

    def _load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config

    def _init_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _setup_routes(self):
        @self.app.route('/perform_task', methods=['POST'])
        def perform_task():
            try:
                data = request.json
                self.logger.info(f"Received task request: {data}")
                if 'task' not in data or 'agent_id' not in data:
                    raise BadRequest("Missing 'task' or 'agent_id' in request")
                
                task = data['task']
                agent_id = data['agent_id']
                
                if agent_id not in self.character_agents:
                    self.logger.error(f"Agent {agent_id} not found")
                    return jsonify({'error': 'Agent not found'}), 404
                
                agent = self.character_agents[agent_id]
                result = ray.get(agent.perform_task.remote(task))
                return jsonify({'result': result})
            except BadRequest as e:
                self.logger.error(f"Bad request: {e}")
                return jsonify({'error': str(e)}), 400
            except Exception as e:
                self.logger.error(f"Error performing task: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/interact', methods=['POST'])
        def interact():
            try:
                data = request.json
                self.logger.info(f"Received interact request: {data}")
                if 'character_name' not in data or 'message' not in data:
                    raise BadRequest("Missing 'character_name' or 'message' in request")
                
                character_name = data['character_name']
                message = data['message']
                
                if character_name not in self.character_agents:
                    self.logger.error(f"Character {character_name} not found")
                    return jsonify({'error': 'Character not found'}), 404
                
                agent = self.character_agents[character_name]
                response = ray.get(agent.handle_message.remote(message))
                return jsonify({'response': response})
            except BadRequest as e:
                self.logger.error(f"Bad request: {e}")
                return jsonify({'error': str(e)}), 400
            except Exception as e:
                self.logger.error(f"Error interacting with character: {e}")
                return jsonify({'error': str(e)}), 500

    def _initialize_agents(self):
        agent_ids = self.project_config.get('agent_ids', [])
        character_names = self.project_config.get('character_names', [])

        for agent_id in agent_ids:
            self.character_agents[agent_id] = CharacterAgent.remote(agent_id)
            self.logger.info(f"Initialized agent: {agent_id}")

        for character_name in character_names:
            self.character_agents[character_name] = CharacterAgent.remote(character_name)
            self.logger.info(f"Initialized character agent: {character_name}")

    def create_standard_directories(self):
        for name, content in self.project_structure.items():
            path = os.path.join(self.base_path, name)
            if isinstance(content, dict):
                os.makedirs(path, exist_ok=True)
                self._create_subdirectories(path, content)
            else:
                with open(path, 'w') as f:
                    f.write(content)
        self.logger.info(f"Project structure created at {self.base_path}")

    def _create_subdirectories(self, base_path, structure):
        for name, content in structure.items():
            path = os.path.join(base_path, name)
            if isinstance(content, dict):
                os.makedirs(path, exist_ok=True)
                self._create_subdirectories(path, content)
            else:
                with open(path, 'w') as f:
                    f.write(content)

    def search_google_scholar(self, query):
        base_url = "https://scholar.google.com/scholar"
        params = {"q": query}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(base_url, params=params, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            results = soup.find_all("div", class_="gs_r gs_or gs_scl")
            for result in results:
                title = result.find("h3", class_="gs_rt").text
                pdf_link = result.find("a", class_="gs_or_mor").get("href")
                print("Title:", title)
                print("PDF Link:", pdf_link)
                print()
        else:
            self.logger.error(f"Error: {response.status_code}")

    def rename_photos(self, folder_path, prefix='photo', start_index=1):
        files = os.listdir(folder_path)
        index = start_index
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                new_name = f"{prefix}_{index}.jpg"
                old_path = os.path.join(folder_path, file)
                new_path = os.path.join(folder_path, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed: {file} -> {new_name}")
                index += 1

    def xlsx_to_csv(self, file_path, output_path):
        df = pd.read_excel(file_path)
        df.to_csv(output_path, index=False, encoding='utf-8')

    def parse_title(self, url, soup):
        if "jmir.org" in url:
            return soup.find('h1').get_text().strip()
        else:
            return soup.find('title').get_text().strip() if soup.find('title') else "标题未找到"

    def get_user_agent(self, device_type):
        user_agents = {
            'desktop': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'mobile': 'Mozilla/5.0 (iPhone; CPU iPhone OS 10_3_1 like Mac OS X) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.0 Mobile/14E304 Safari/602.1'
        }
        return user_agents.get(device_type, user_agents['desktop'])

    def scrape_titles(self, urls, device_type='desktop'):
        headers = {'User-Agent': self.get_user_agent(device_type)}
        for url in urls:
            try:
                print(f"url: {url}")
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.content, 'html.parser')
                title = self.parse_title(url, soup)
                print(title)
            except Exception as e:
                print(f"无法从 {url} 爬取数据: {e}")

    def interact_with_character(self, character_name, message):
        if character_name not in self.character_agents:
            self.logger.error(f"Character {character_name} not found")
            return {"error": "Character not found"}
        agent = self.character_agents[character_name]
        return ray.get(agent.handle_message.remote(message))

    def run_app(self):
        try:
            self.app.run(port=5000)
        except Exception as e:
            self.logger.error(f"Error running the app: {e}")
            raise
        finally:
            ray.shutdown()

# 多智能体系统类
class MultiAgentSystem:
    def __init__(self):
        self.redis_client = RedisClient()
        self.kamen_rider_agent = KamenRiderBladeAgent('/path/to/base', 'config.json')

    def start_agents(self):
        agent_a = AgentA(self.redis_client)
        agent_b = AgentB(self.redis_client)
        agent_c = AgentC(self.redis_client)

        threading.Thread(target=agent_a.run).start()
        threading.Thread(target=agent_b.run).start()
        threading.Thread(target=agent_c.run).start()

    def nlp_processing(self, text):
        nlp_agent = NlpAgent()
        sentiment = nlp_agent.analyze_sentiment(text)
        print(f"Sentiment polarity: {sentiment}")
        user_feedback = "positive" if sentiment > 0 else "negative"
        nlp_agent.feedback_loop(user_feedback, text)

    def run_pretrained_agents(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            agents_config = json.load(f)['characters']
        
        test_text = "这是一个测试对话。"
        
        for character, config in agents_config.items():
            agent = PretrainedModelAgent(config['model_path'], config['tokenizer_path'], config['device'])
            response = agent.run(test_text)
            print(f"Response from {character}: {response}")

    def handle_sgd_training(self, X_train, y_train):
        sgd_handler = SgdModelHandler(X_train, y_train)
        for X_partial, y_partial in generate_partial_data():
            sgd_handler.partial_fit(X_partial, y_partial)

    def run_kamen_rider_tasks(self):
        # 示例：运行KamenRiderBladeAgent中定义的功能
        query = 'your search keywords'
        self.kamen_rider_agent.search_google_scholar(query)

        folder_path = '/default/path/to/photos'
        self.kamen_rider_agent.rename_photos(folder_path)

        file_path = '/default/path/to/excel.xlsx'
        output_path = '/default/path/to/output.csv'
        self.kamen_rider_agent.xlsx_to_csv(file_path, output_path)

        urls = [
            "https://formative.jmir.org/2024/1/e50056",
            "https://www.proquest.com/openview/1dccdeab218b1ac51ca7ef049c3c6636/1?pq-origsite=gscholar&cbl=18750&diss=y",
            "https://link.springer.com/article/10.1007/s10994-023-06460-4"
        ]
        self.kamen_rider_agent.scrape_titles(urls)

    def run_all(self):
        self.start_agents()
        self.run_kamen_rider_tasks()

def main():
    system = MultiAgentSystem()

    # 启动多智能体系统
    system.run_all()

    # NLP处理示例
    text = "I love this movie. It's amazing!"
    system.nlp_processing(text)
    
    # 运行预训练模型智能体
    config_file = 'agents_config.json'
    system.run_pretrained_agents(config_file)
    
    # SGD模型处理示例
    X_train, y_train = get_training_data()  # 请确保此函数已定义
    system.handle_sgd_training(X_train, y_train)

if __name__ == "__main__":
    main()
