import json
import language_tool_python

# 初始化 LanguageTool，用于中文语法检查
tool = language_tool_python.LanguageTool('zh-CN')

# 从 data.json 文件中加载数据
with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 修正语法错误
for item in data:
    # 使用 LanguageTool 进行语法检查和修正
    matches = tool.check(item['text'])
    corrected_text = language_tool_python.utils.correct(item['text'], matches)
    item['text'] = corrected_text

# 将修正后的数据保存到新的 JSON 文件中
with open('corrected_data.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("语法修正完成，结果已保存到 corrected_data.json 文件中。")

import json
import language_tool_python

# 初始化 LanguageTool，用于中文语法检查
tool = language_tool_python.LanguageTool('zh-CN')

# 从 data.json 文件中加载数据
with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 修正语法错误
for item in data:
    # 使用 LanguageTool 进行语法检查和修正
    matches = tool.check(item['text'])
    corrected_text = language_tool_python.utils.correct(item['text'], matches)
    item['text'] = corrected_text

# 将修正后的数据保存到新的 JSON 文件中
with open('corrected_data.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("语法修正完成，结果已保存到 corrected_data.json 文件中。")
