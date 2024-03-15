from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import io

# 初始化客户端
client = speech.SpeechClient()

# 从本地文件加载音频
with io.open('audio_file.wav', 'rb') as audio_file:
    content = audio_file.read()
    audio = types.RecognitionAudio(content=content)

# 设置识别配置
config = types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code='en-US'
)

# 调用Google Speech-to-Text API进行语音识别
response = client.recognize(config=config, audio=audio)

# 打印识别结果
for result in response.results:
    print('Transcript: {}'.format(result.alternatives[0].transcript))
<<<<<<< HEAD

=======
>>>>>>> origin/main
