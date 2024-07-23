# 舞台剧多智能体语言模型项目

## 项目简介

本项目旨在开发一个能够针对特定舞台剧进行交互的多智能体大型语言模型（LLM）。项目包括四个主要智能体，每个智能体都能够理解并生成与其角色相关的对话和行为。

## 开始使用

### 环境搭建

1. 克隆仓库到本地：

```bash
git clone https://github.com/your-repository/stage-play-llm.git
```

数据准备：将舞台剧相关数据放置于 data/raw/ 目录下，并运行预处理脚本
```bash
python scripts/preprocessing.py
```

模型训练：启动模型训练过程
```bash
python scripts/train.py
```

模型微调：根据具体需求对模型进行微调
```bash
python scripts/finetune.py
```

模型评估：评估模型的性能
```bash
python scripts/evaluate.py
```

2. 安装必要的依赖
```bash
pip install -r requirements.txt
```
	
## 技术栈
- TensorFlow 或 PyTorch
- LangChain
- Python 3.x

## 贡献者指南

我们欢迎更多的开发者加入我们的项目。如果您对改进此项目有任何建议或想要贡献代码，请阅读 `CONTRIBUTING.md`。

## 许可证

此项目采用 MIT 许可证。详情请见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或需要支持，请联系 `xiaoyu991214@gmail.com`。



# API Documentation

## Overview

This document provides detailed information about the interfaces of the Stage Play LLM API. It is intended for developers who need to integrate the LLM models into stage play applications.

# Methodologies
<div align="center">
<img src="./assets/method.gif" alt="Method" title="method">
</div>

## Models

## Reproducing the Evaluation

### 1. Clone the Repo

### 2. Download the Model

### 3. Install Libraries

### 4. Run

## Authentication

- **Method**: API Key
- **Key Location**: Header

```http
GET /api/resource HTTP/1.1
Host: example.com
Authorization: Api-Key {your-api-key-here}
```

## Reference

## Acknowledgement

https://cs.union.edu/csc120/

# Introduction

All characters in the TV Series of Kamen Rider Blade (仮面ライダー剣 in 2004).

I inherit the work from the first training session in InternLM (书生·浦语) in 2024 to build this system.

This is a repository for building the agent system that enables chatting features with the copies the stage.

It mirrors my understanding towards a large part of my life so far, while working as an AI project that attempts to chain my learning path in computer science over the years too.

運命の切り札...

「やらなければいけないことがあります。私たちは手を組み、必勝の陣形で前に進んでいくでしょう。」

戦さ！

自分との戦いには、終わりがありません。