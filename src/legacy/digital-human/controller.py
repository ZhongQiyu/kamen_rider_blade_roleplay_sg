# controller.py

import openai
import spacy
import pyautogui
from emotion_recognition import EmotionRecognizer

# 初始化spacy模型
nlp = spacy.load('en_core_web_sm')

# 设置OpenAI API密钥
openai.api_key = 'your-api-key-here'

# 自然语言指令解析
def parse_command_with_llm(input_text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"解析以下指令并生成游戏操作命令：{input_text}",
        max_tokens=50
    )
    return response.choices[0].text.strip()

# 情感分析
emotion_recognizer = EmotionRecognizer()

def analyze_emotion(input_text):
    emotion = emotion_recognizer.recognize(input_text)
    return emotion

class GameController:
    def build_structure(self, building_type, location):
        pyautogui.press('b')
        pyautogui.press(building_type[0])
        pyautogui.click(location)

    def train_units(self, unit_type, quantity):
        pyautogui.press('t')
        pyautogui.press(unit_type[0])
        for _ in range(quantity):
            pyautogui.press('enter')

    def move_units(self, unit_type, destination):
        pyautogui.press('m')
        pyautogui.click(destination)

class YourKamenRiderBladeAgent:
    def generate_response(self, input_text):
        return f"假面骑士剑响应: {input_text}"

class GameDialogueManager:
    def __init__(self, game_controller, dialogue_agent):
        self.game_controller = game_controller
        self.dialogue_agent = dialogue_agent
    
    def generate_response(self, input_text):
        command = parse_command_with_llm(input_text)
        if not command:
            return "无法识别指令。请重试。"

        if "建造" in command:
            building_type = command.split()[1]
            location = command.split()[-1]
            self.game_controller.build_structure(building_type, location)
            return f"正在建造{building_type}在{location}。"

        elif "训练" in command:
            unit_type = command.split()[1]
            quantity = int(command.split()[-1])
            self.game_controller.train_units(unit_type, quantity)
            return f"正在训练{quantity}个{unit_type}。"

        elif "移动" in command:
            unit_type = command.split()[1]
            destination = command.split()[-1]
            self.game_controller.move_units(unit_type, destination)
            return f"正在将{unit_type}移动到{destination}。"

        return "无法识别指令。请重试。"
    
    def generate_dialogue_response(self, input_text):
        response = self.dialogue_agent.generate_response(input_text)
        return response

def generate_emotional_response(input_text, game_dialogue_manager):
    emotion = analyze_emotion(input_text)
    if emotion == "anger":
        return "我注意到你有些沮丧，需要帮助吗？"
    elif emotion == "happiness":
        return "很高兴你满意！需要更多帮助吗？"
    else:
        response = game_dialogue_manager.generate_response(input_text)
        if not response:
            response = game_dialogue_manager.generate_dialogue_response(input_text)
        return response
