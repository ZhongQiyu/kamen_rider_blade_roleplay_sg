# Stage Play Multi-Agent Language Model Project

## Introduction

All characters in the TV Series of Kamen Rider Blade (仮面ライダー剣 in 2004).
I inherit the work from the first training session in InternLM (书生·浦语) in 2024 to build this system.
This is a repository for building the agent system that enables chatting features with the copies the stage.
It mirrors my understanding towards a large part of my life so far, while working as an AI project that attempts to chain my learning path in computer science over the years too.

## Overview

This project aims to develop a large language model (LLM) specifically for a stage play. The project will include four to twelve main agents, each capable of understanding and generating dialogues and actions related to their roles.

This project aims to develop a large language model (LLM) specifically for a stage play, containing four to twelve independent agents, each capable of understanding and generating dialogues and actions related to their roles.

This document provides detailed information about the interfaces of the Stage Play LLM API. It is intended for developers who need to integrate the LLM models into stage play applications.

## Project Description

This project aims to develop a large language model (LLM) for stage plays, containing four to twelve independent agents, each capable of understanding and generating dialogues and actions related to their roles.

## Milestones and Delivery Schedule

### Timeline
- **May 2024**: System framework and agent prototypes completed.
- **June 2024**: Complete agent training and preliminary testing.
- **July 2024**: System integration and comprehensive testing.
- **August 2024**: User acceptance testing and deployment preparation.
- **September 2024**: Project officially goes live.

### Goals
- Train a language model with four to twelve agents.
- Each agent should be able to accurately generate dialogues based on the plot and character backgrounds.
- Implement effective interactions between agents to support the unfolding of complex plots.

### Final Deliverables
- Autonomous performing agents.
- A complete deployment solution including backend model deployment and frontend interaction interface.

## Getting Started

### Setting Up the Environment

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/your-repository/stage-play-llm.git
    ```

2. Install the necessary dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Data preparation: Place the stage play related data in the `data/raw/` directory and run the preprocessing script:

    ```bash
    python scripts/preprocessing.py
    ```

4. Model training: Start the model training process:

    ```bash
    python scripts/train.py
    ```

5. Model fine-tuning: Fine-tune the model according to specific needs:

    ```bash
    python scripts/finetune.py
    ```

6. Model evaluation: Evaluate the model performance:

    ```bash
    python scripts/evaluate.py
    ```

### Methodologies

<div align="center">
<img src="./assets/method.gif" alt="Method" title="method">
</div>

## Models

### 1. Clone the Repo

```bash
git clone https://github.com/your-repository/stage-play-llm.git
```

### 2. Download the Model

Download the pre-trained models from the provided link.

### 3. Install Libraries

```bash
pip install -r requirements.txt
```

### 4. Run

```http
Authentication
Method: API Key
Key Location: Header
GET /api/resource HTTP/1.1
Host: example.com
Authorization: Api-Key {your-api-key-here}
```

## Reference

## Acknowledgement

## Technical Requirements

### Software Requirements

- Python 3.8+: Primary programming language.
- TensorFlow 2.x / PyTorch 1.8+: For model training and inference.
- Flask/Django: For API development and deployment.
- Docker: For application packaging and deployment.
- torch: Deep learning framework for dynamic neural networks.
- transformers: Library for state-of-the-art natural language processing.
- pysrt: Python library for parsing and modifying SubRip (.srt) subtitle files.
- Hardware Requirements
- GPU: NVIDIA RTX 2080 Ti or higher, at least 4 cards.
- CPU: Intel i7 or higher.
- RAM: At least 64GB.
- Storage: SSD at least 2TB.

### Dependency Libraries

See requirements.txt for a list of all required Python libraries.

```plaintext
requirements.txt
tensorflow>=2.x
torch>=1.8
transformers
flask
django
docker
pysrt
```

### Functional Requirements

- Agent Functions
Language Understanding: Understand complex language inputs and respond.
Emotion Expression: Express and understand emotions in dialogue.
Memory Ability: Remember previous dialogues and actions.

- System Functions
User Interaction Interface: A clean interface that allows users to interact and watch performances.
Performance Monitoring: Monitor the performance of agents and the system.

- Security and Compliance
Comply with GDPR and other data protection regulations.
The system has security measures to prevent data leakage.

- Testing Requirements
Unit Testing: Test key functions.
Integration Testing: Ensure system components work together.
Performance Testing: System performance under high load.
User Acceptance Testing: Ensure compliance with user expectations and requirements.

- Milestones and Delivery Schedule
May 2024: System framework and agent prototypes completed.
June 2024: Complete agent training and preliminary testing.
July 2024: System integration and comprehensive testing.
August 2024: User acceptance testing and deployment preparation.
September 2024: Project officially goes live.

## Technology Stack

- TensorFlow or PyTorch
- LangChain
- Python 3.x

## Technical Requirements

### Software Requirements

- **Python 3.8+**: Primary programming language.
- **TensorFlow 2.x / PyTorch 1.8+**: For model training and inference.
- **Flask/Django**: For API development and deployment.
- **Docker**: For application packaging and deployment.
- **torch**: Deep learning framework for dynamic neural networks.
- **transformers**: Library for state-of-the-art natural language processing.
- **pysrt**: Python library for parsing and modifying SubRip (.srt) subtitle files.

### Hardware Requirements

- **GPU**: NVIDIA RTX 2080 Ti or higher, at least 4 cards.
- **CPU**: Intel i7 or higher.
- **RAM**: At least 64GB.
- **Storage**: SSD at least 2TB.

### Dependency Libraries

See the `requirements.txt` file for a list of all required Python libraries.

```plaintext
# requirements.txt

tensorflow>=2.x
torch>=1.8
transformers
flask
django
docker
pysrt
```

## Functional Requirements

### Agent Functions

- **Language Understanding**: Understand complex language inputs and respond.
- **Emotion Expression**: Express and understand emotions in dialogue.
- **Memory Ability**: Remember previous dialogues and actions.

### System Functions

- **User Interaction Interface**: A clean interface that allows users to interact and watch performances.
- **Performance Monitoring**: Monitor the performance of agents and the system.

## Security and Compliance

- **Comply with GDPR and other data protection regulations.**
- **The system has security measures to prevent data leakage.**

## Testing Requirements

- **Unit Testing**: Test key functions.
- **Integration Testing**: Ensure system components work together.
- **Performance Testing**: System performance under high load.
- **User Acceptance Testing**: Ensure compliance with user expectations and requirements.

## Contributor Guide

We welcome more developers to join our project. If you have any suggestions for improving this project or want to contribute code, please read the `CONTRIBUTING.md`.

## License

This project is licensed under the MIT License. For details, please see the `LICENSE` file.

## Contact Information

If you have any questions or need support, please contact xiaoyu991214@gmail.com.