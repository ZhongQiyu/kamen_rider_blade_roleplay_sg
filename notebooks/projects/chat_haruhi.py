# chat_haruhi.py

from chatharuhi import ChatHaruhi

chatbot = ChatHaruhi(
    role_name = 'haruhi',
    llm = 'openai'
)

response = chatbot.chat(role='阿虚', text='野球の新シーズンが始まりますね！参加する？')

print(response)