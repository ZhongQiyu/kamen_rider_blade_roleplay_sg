import re

def handle_dialog_from_file(file_path):
    data = []
    current_speaker = None
    current_time = None
    dialog = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            speaker_match = re.match(r'^说话人(\d+) (\d{2}:\d{2})', line)
            if speaker_match:
                if current_speaker is not None:
                    # 当遇到新的说话人时，保存之前的对话
                    data.append({
                        'speaker': current_speaker,
                        'time': current_time,
                        'text': ' '.join(dialog).strip()
                    })
                    dialog = []  # 重置对话列表
                current_speaker, current_time = speaker_match.groups()
            else:
                # 收集对话行
                dialog.append(line.strip())

    # 确保最后一个对话被添加
    if current_speaker and dialog:
        data.append({
            'speaker': current_speaker,
            'time': current_time,
            'text': ' '.join(dialog).strip()
        })

    return data

# 路径到您的完整文件
file_path = '/mnt/data/1.txt'
all_dialogs = handle_dialog_from_file(file_path)

# 打印出处理结果的一部分或进行进一步的分析
print(all_dialogs[:5])  # 打印前5条对话以检查输出
