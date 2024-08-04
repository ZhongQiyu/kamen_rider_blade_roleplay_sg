from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch

# 假设已经加载了角色对话数据集
dialogs = {
    "角色1": ["台词1角色1", "台词2角色1", ...],
    "角色2": ["台词1角色2", "台词2角色2", ...],
    ...
}

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

for role, lines in dialogs.items():
    # 为每个角色创建一个文本文件
    with open(f"{role}_lines.txt", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + tokenizer.eos_token)

    # 加载数据集和数据整理器
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=f"{role}_lines.txt",
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # 加载预训练模型
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=f'./{role}_finetuned_gpt2', # 输出目录
        overwrite_output_dir=True,
        num_train_epochs=5, # 训练轮数
        per_device_train_batch_size=4, # batch大小
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # 开始训练
    trainer.train()

    # 保存模型
    model.save_pretrained(f'./{role}_finetuned_gpt2')
