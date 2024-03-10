import json

json_path = "transcripts_output_transcript_68c650f0-0000-26af-b8d3-2405887b1c1c.json"

# 加载JSON数据
with open(json_path, 'r') as f:
    data = json.load(f)

# 从JSON数据中提取信息
for result in data['results']:
    for alternative in result['alternatives']:
        transcript = alternative['transcript']
        confidence = alternative['confidence']
        print(f"Transcript: {transcript}")
        print(f"Confidence: {confidence}")

        # 如果有词时间偏移（word timing offsets）
        if 'words' in alternative:
            for word_info in alternative['words']:
                word = word_info['word']
                start_time = word_info['startTime']
                end_time = word_info['endTime']
                print(f"Word: {word}, start time: {start_time}, end time: {end_time}")
