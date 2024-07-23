# agent.py

import json
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import ray

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
        # 这里可以实现具体的任务处理逻辑
        return f"Processed: {self.task}"

def main():
    base_path = os.getenv('BASE_PATH', '/default/path/to/base')  # 使用环境变量或默认路径
    config_path = os.getenv('CONFIG_PATH', 'config.json')  # 使用环境变量或默认配置文件路径
    agent = KamenRiderBladeAgent(base_path, config_path)
    agent.create_standard_directories()

    query = os.getenv('SEARCH_QUERY', 'your search keywords')  # 使用环境变量或默认搜索关键字
    agent.search_google_scholar(query)

    folder_path = os.getenv('PHOTO_FOLDER_PATH', '/default/path/to/photos')  # 使用环境变量或默认路径
    agent.rename_photos(folder_path)

    file_path = os.getenv('EXCEL_FILE_PATH', '/default/path/to/excel.xlsx')  # 使用环境变量或默认路径
    output_path = os.getenv('CSV_OUTPUT_PATH', '/default/path/to/output.csv')  # 使用环境变量或默认路径
    agent.xlsx_to_csv(file_path, output_path)

    urls = [
        "https://formative.jmir.org/2024/1/e50056",
        "https://www.proquest.com/openview/1dccdeab218b1ac51ca7ef049c3c6636/1?pq-origsite=gscholar&cbl=18750&diss=y",
        "https://link.springer.com/article/10.1007/s10994-023-06460-4"
    ]
    agent.scrape_titles(urls)
    agent.run_app()

if __name__ == "__main__":
    main()
