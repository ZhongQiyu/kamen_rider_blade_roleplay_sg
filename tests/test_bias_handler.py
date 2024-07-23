# test_bias_handler.py

from bias_handler import BiasHandler  # 导入 BiasHandler 类

# 初始化偏见处理器
bias_handler = BiasHandler()

# 示例文本
text = "彼は悪い人です。彼はダメな人です。"

# 处理文本
processed_text = bias_handler.process_text(text)

# 打印处理后的文本
print(processed_text)
