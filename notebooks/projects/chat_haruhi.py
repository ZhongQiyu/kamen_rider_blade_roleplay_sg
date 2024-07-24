# chat_haruhi.py

import openai

class ChatHaruhi:
    def __init__(self, role_name, api_key, temperature=0.7, max_tokens=150):
        self.role_name = role_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        openai.api_key = self.api_key

    def chat(self, role, text):
        prompt = f"{role}：{text}\n{self.role_name}："
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=1,
            stop=None
        )
        message = response.choices[0].text.strip()
        return message

# 创建 ChatHaruhi 实例
chatbot = ChatHaruhi(
    role_name='haruhi',
    api_key='your_openai_api_key',  # 替换为你的 OpenAI API 密钥
    temperature=0.7,
    max_tokens=150
)

# 测试函数
def test_chatbot():
    test_cases = [
        {'role': '阿虚', 'text': '野球の新シーズンが始まりますね！参加する？', 'expected_substring': '野球'},
        {'role': 'ハルヒ', 'text': '次の映画は何を見たいですか？', 'expected_substring': '映画'},
        {'role': 'みくる', 'text': '今日は何をしましょうか？', 'expected_substring': '今日は'},
        {'role': '古泉', 'text': '天気が良いですね。散歩に行きますか？', 'expected_substring': '天気'},
    ]

    for case in test_cases:
        response = chatbot.chat(role=case['role'], text=case['text'])
        print(f"Role: {case['role']}, Text: {case['text']}, Response: {response}")
        assert case['expected_substring'] in response, f"Test failed for role: {case['role']}"

if __name__ == "__main__":
    test_chatbot()
