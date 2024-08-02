import random

# 模拟的验证函数，实际应用中需要替换为真正的验证逻辑
def verify_fact(question, answer):
    # 这里我们随机返回 True 或 False 来模拟验证过程
    # 实际应用中应该替换为基于事实核实的代码
    return random.choice([True, False])

# 示例问答对列表，这里应该是从您提供的数据中获取
base_questions_and_answers = [
    # ... (100个问答对，由于篇幅限制，省略了实际内容)
]

# 去除重复的问答对
unique_questions_and_answers = list(set(base_questions_and_answers))

# 验证问答对
verified_questions_and_answers = [(q, a) for q, a in unique_questions_and_answers if verify_fact(q, a)]

# 如果删除了不正确的问答对后数量减少了，我们需要从剩余的问答对中随机选择补充
while len(verified_questions_and_answers) < len(base_questions_and_answers):
    q, a = random.choice(base_questions_and_answers)
    if (q, a) not in verified_questions_and_answers:
        verified_questions_and_answers.append((q, a))

# 确保最终数量和原始数量一致
assert len(verified_questions_and_answers) == len(base_questions_and_answers)

# 打乱顺序以去除可能的偏差
random.shuffle(verified_questions_and_answers)

# 展示处理后的问答对数量及前5个问答对
verified_questions_and_answers[:5]
