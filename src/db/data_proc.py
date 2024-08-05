# data_proc.py

import re
import os
import csv
import json
import boto3

class DataProcessor:
    def __init__(self, directory_path, config_file):
        self.directory_path = directory_path
        self.data = []
        self.dialog = []
        self.current_time = None
        self.current_episode = {'episode': 'Unknown', 'dialogs': []}
        self.current_speaker = None
        self.config = self.load_config(config_file)

    def get_directory_path(self):
        return self.directory_path

    def get_data(self):
        return self.data

    def get_current_episode(self):
        return self.current_episode

    def load_config(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def parse_train_data(self, data):
        input_key = self.config['train_data']['input_key']
        output_key = self.config['train_data']['output_key']
        parsed_data = [{'input': entry[input_key], 'ideal_response': entry[output_key]} for entry in data]
        return parsed_data

    def parse_asr_data(self, data):
        text_key = self.config['asr_data']['text_key']
        parsed_data = [{'text': entry[text_key]} for entry in data]
        return parsed_data

    def lambda_handler(self, event, context):
        s3 = boto3.client('s3')

        for record in event['Records']:
            bucket_name = record['s3']['bucket']['name']
            object_key = record['s3']['object']['key']
            print(f"A new file {object_key} was uploaded in bucket {bucket_name}")

        return {
            'statusCode': 200,
            'body': json.dumps('Process completed successfully!')
        }

    @staticmethod
    def sort_files(filename):
        part = filename.split('.')[0]
        try:
            return int(part)
        except ValueError:
            return float('inf')

    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return lines

    def finalize_episode(self):
        if self.current_episode:
            if self.dialog:
                self.current_episode['dialogs'].append({
                    'speaker': self.current_speaker,
                    'time': self.current_time,
                    'text': ' '.join(self.dialog).strip()
                })
                self.dialog = []
            self.data.append(self.current_episode)
            print(f"Finalized episode: {self.current_episode}")
            self.current_episode = {'episode': 'Unknown', 'dialogs': []}

    def process_line(self, line):
        speaker_match = re.match(r'^話者(\d+)\s+(\d{2}:\d{2})\s+(.*)$', line)  # 日语版本的正则表达式
        if speaker_match:
            if self.dialog:  # 如果有未完成的对话，先完成它
                self.current_episode['dialogs'].append({
                    'speaker': self.current_speaker,
                    'time': self.current_time,
                    'text': ' '.join(self.dialog).strip()
                })
                self.dialog = []
            self.current_speaker, self.current_time, text = speaker_match.groups()
            self.dialog = [text]
        else:
            self.dialog.append(line)

    def process_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    self.process_line(line)
        self.finalize_episode()
        print(f"Processed file: {file_path} with data: {self.data[-1] if self.data else 'No Data'}")

    def process_all_files(self):
        files = [f for f in os.listdir(self.directory_path) if f.endswith('.txt')]
        files = sorted(files, key=self.sort_files)
        for filename in files:
            file_path = os.path.join(self.directory_path, filename)
            self.process_file(file_path)

    def export_to_txt(self, output_file):
        with open(output_file, 'w', encoding='utf-8') as file:
            for content in self.data:
                file.write(json.dumps(content, ensure_ascii=False) + '\n')

    def handle_dialog(self, lines):
        for line in lines:
            speaker_match = re.match(r'^話者(\d+) (\d{2}:\d{2})', line)  # 日语版本的正则表达式
            if speaker_match:
                if self.current_speaker is not None:
                    self.data.append({
                        'speaker': self.current_speaker,
                        'time': self.current_time,
                        'text': ' '.join(self.dialog).strip()
                    })
                    self.dialog = []
                self.current_speaker, self.current_time = speaker_match.groups()
            else:
                self.dialog.append(line.strip())

        if self.current_speaker and self.dialog:
            self.data.append({
                'speaker': self.current_speaker,
                'time': self.current_time,
                'text': ' '.join(self.dialog).strip()
            })

        return self.data

    def save_as_json(self, output_file):
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(self.data, file, ensure_ascii=False, indent=4)

    def save_as_csv(self, output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            fieldnames = ['episode', 'time', 'speaker', 'text']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for episode in self.data:
                if 'dialogs' not in episode:
                    continue
                for dialog in episode['dialogs']:
                    writer.writerow({
                        'episode': episode['episode'],
                        'time': dialog['time'],
                        'speaker': dialog['speaker'],
                        'text': dialog['text']
                    })

    def generate_new_entry(self, last_dialog):
        new_prompt = last_dialog['text']
        new_response = "新しい回答"  # 日语版本
        return {
            'prompt': new_prompt,
            'response': new_response,
            'chosen': new_response,
            'rejected': "他の選択肢"
        }

    def decide_chosen_and_rejected(self, responses):
        if responses:
            chosen = responses[0]
            rejected = responses[1:]
        else:
            chosen = None
            rejected = []
        return chosen, rejected

def main():
    directory_path = '/path/to/your/data/episodes_txt'  # 更新为你的实际路径
    config_file = 'config.json'
    output_json_path = os.path.join(directory_path, 'data.json')
    output_csv_path = os.path.join(directory_path, 'data.csv')
    output_txt_path = os.path.join(directory_path, 'combined.txt')

    proc = DataProcessor(directory_path, config_file)

    proc.process_all_files()
    print("Data ready for export:", proc.get_data()[:1])
    proc.export_to_txt(output_txt_path)

    all_lines = proc.read_file(os.path.join(directory_path, '1.txt'))
    dialog_data = proc.handle_dialog(all_lines)

    proc.data = dialog_data
    proc.save_as_json(output_json_path)
    proc.save_as_csv(output_csv_path)

if __name__ == "__main__":
    main()
