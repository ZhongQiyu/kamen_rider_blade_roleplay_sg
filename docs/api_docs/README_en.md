# Stage Play Multi-Agent Language Model Project

## Introduction

All characters in the TV Series of Kamen Rider Blade (仮面ライダー剣 in 2004).
I inherit the work from the first training session in InternLM (书生·浦语) in 2024 to build this system.
This is a repository for building the agent system that enables chatting features with the copies the stage.
It mirrors my understanding towards a large part of my life so far, while working as an AI project that attempts to chain my learning path in computer science over the years too.

## Overview

This project aims to develop a large language model (LLM) specifically for a stage play, containing four to twelve independent agents, each capable of understanding and generating dialogues and actions related to their roles.

This document provides detailed information about the interfaces of the Stage Play LLM API. It is intended for developers who need to integrate the LLM models into stage play applications.

# Methodologies
<div align="center">
<img src="./assets/method.gif" alt="Method" title="method">
</div>

## Models

## Reproducing the Evaluation

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

Reference
Acknowledgement

Technical Requirements
Software Requirements
Python 3.8+: Primary programming language.
TensorFlow 2.x / PyTorch 1.8+: For model training and inference.
Flask/Django: For API development and deployment.
Docker: For application packaging and deployment.
torch: Deep learning framework for dynamic neural networks.
transformers: Library for state-of-the-art natural language processing.
pysrt: Python library for parsing and modifying SubRip (.srt) subtitle files.
Hardware Requirements
GPU: NVIDIA RTX 2080 Ti or higher, at least 4 cards.
CPU: Intel i7 or higher.
RAM: At least 64GB.
Storage: SSD at least 2TB.
Dependency Libraries
See requirements.txt for a list of all required Python libraries.

plaintext
Copy code
# requirements.txt

tensorflow>=2.x
torch>=1.8
transformers
flask
django
docker
pysrt
Functional Requirements
Agent Functions
Language Understanding: Understand complex language inputs and respond.
Emotion Expression: Express and understand emotions in dialogue.
Memory Ability: Remember previous dialogues and actions.
System Functions
User Interaction Interface: A clean interface that allows users to interact and watch performances.
Performance Monitoring: Monitor the performance of agents and the system.
Security and Compliance
Comply with GDPR and other data protection regulations.
The system has security measures to prevent data leakage.
Testing Requirements
Unit Testing: Test key functions.
Integration Testing: Ensure system components work together.
Performance Testing: System performance under high load.
User Acceptance Testing: Ensure compliance with user expectations and requirements.
Milestones and Delivery Schedule
May 2024: System framework and agent prototypes completed.
June 2024: Complete agent training and preliminary testing.
July 2024: System integration and comprehensive testing.
August 2024: User acceptance testing and deployment preparation.
September 2024: Project officially goes live.







## Project Overview

This project aims to develop a large language model (LLM) specifically for a stage play. The project will include four to twelve main agents, each capable of understanding and generating dialogues and actions related to their roles.

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

## Technology Stack

- TensorFlow or PyTorch
- LangChain
- Python 3.x

## Contributor Guide

We welcome more developers to join our project. If you have any suggestions for improving this project or want to contribute code, please read the `CONTRIBUTING.md`.

## License

This project is licensed under the MIT License. For details, please see the `LICENSE` file.

## Contact Information

If you have any questions or need support, please contact xiaoyu991214@gmail.com.

---

## Project Description

This project aims to develop a large language model (LLM) for stage plays, containing four to twelve independent agents, each capable of understanding and generating dialogues and actions related to their roles.

### Goals

- Train a language model with four to twelve agents.
- Each agent should be able to accurately generate dialogues based on the plot and character backgrounds.
- Implement effective interactions between agents to support the unfolding of complex plots.

### Final Deliverables

- Autonomous performing agents.
- A complete deployment solution including backend model deployment and frontend interaction interface.

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

## Milestones and Delivery Schedule

- **May 2024**: System framework and agent prototypes completed.
- **June 2024**: Complete agent training and preliminary testing.
- **July 2024**: System integration and comprehensive testing.
- **August 2024**: User acceptance testing and deployment preparation.
- **September 2024**: Project officially goes live.

## Goals
- Train a language model with four to twelve agents.
- Each agent should be able to accurately generate dialogues based on the plot and character backgrounds.
- Implement effective interactions between agents to support the unfolding of complex plots.

## Final Deliverables
- Autonomous performing agents.
- A complete deployment solution including backend model deployment and frontend interaction interface.

---

# Kamen Rider Blade Multi-Agent System User Guide

## System Overview

This multi-agent system is based on the main characters from the Japanese tokusatsu series "Kamen Rider Blade". It aims to provide a versatile problem-solving and decision-support tool. The system includes four core agents, each simulating the characteristics and abilities of the characters in the series, to help users handle complex problems and situations.

## Agent Introduction

### Kazuma Kenzaki Agent

- **Main Function**: Decision-making and execution
- **Characteristics**: Brave, strong sense of justice, high adaptability
- **Use Case**: Situations requiring quick decisions and decisive actions

### Sakuya Tachibana Agent

- **Main Function**: Strategic analysis and risk assessment
- **Characteristics**: Calm, rational, cautious
- **Use Case**: Situations requiring detailed analysis and long-term planning

### Hajime Aikawa Agent

- **Main Function**: Information gathering and processing
- **Characteristics**: Sharp, observant, flexible
- **Use Case**: Situations requiring extensive information gathering and detailed insights

### Mutsuki Kamijo Agent

- **Main Function**: Support coordination and emotional analysis
- **Characteristics**: Gentle, empathetic, insightful
- **Use Case**: Situations involving interpersonal relationships and emotional factors

## System Usage

### System Initialization

```python
from blade_agents import BladeAgentSystem

system = BladeAgentSystem()
system.initialize()
```

### Setting Problems or Tasks

```python
problem = "How to improve internal communication efficiency in the company?"
system.set_task(problem)
```

### Activating Agents and Getting Feedback

```python
# Decision by Kazuma Kenzaki Agent
decision = system.activate_agent("kazuki")
print("Kazuma's Decision:", decision)

# Analysis by Sakuya Tachibana Agent
analysis = system.activate_agent("tachibana")
print("Sakuya's Analysis:", analysis)

# Information by Hajime Aikawa Agent
intel = system.activate_agent("aikawa")
print("Hajime's Information:", intel)

# Coordination suggestion by Mutsuki Kamijo Agent
coordination = system.activate_agent("mutsuki")
print("Mutsuki's Coordination Suggestion:", coordination)
```

### Comprehensive Analysis

```python
final_solution = system.synthesize_solutions()
print("Final Solution:", final_solution)
```

## Application Scenarios

### Project Management

```python
system.set_task("How to optimize the progress of a software development project?")
```

### Market Strategy

```python
system.set_task("What kind of marketing strategy should be adopted to promote a new product?")
```

### Crisis Management

```python
system.set_task("How should the company handle a public relations crisis?")
```

### Team Building

```python
system.set_task("How to improve team cohesion and work efficiency?")
```

## Notes

- Each agent provides suggestions based on a specific perspective.
- Final decisions should consider all outputs.
- System suggestions are for reference only; actual execution should be based on specific circumstances.
- Regularly update the system's knowledge base to ensure agents provide the latest and most relevant suggestions.
- For highly confidential or important decisions, it is recommended to combine the opinions of human experts.

## Customization and Expansion

### Adjusting Agent Parameters

```python
system.customize_agent("kazuki", risk_tolerance=0.8)
system.customize_agent("tachibana", analysis_depth="high")
```

### Adding New Functions

```python
system.add_new_capability("aikawa", "social_media_analysis")
```

### Creating New Agents

```python
system.create_new_agent("hirose", role="technical_expert")
```

## Troubleshooting

If the system is slow or results are abnormal, try the following steps:

- Reinitialize the system
- Ensure the problem description is clear and specific
- Adjust agent parameters
- Update the system knowledge base
- Contact the technical support team for help

## Conclusion

The Kamen Rider Blade Multi-Agent System provides a unique perspective on solving complex problems. By simulating the thinking patterns of different characters, it helps users comprehensively analyze problems and make more precise decisions. We hope this system becomes a powerful assistant in your work, bringing new ideas and inspiration. If you have any questions or suggestions, please feel free to contact our support team. Enjoy using it!

---




# Model Architecture for Stage Play LLM

## Overview

This document provides a detailed overview of the model architecture used in our large language model (LLM) for stage play interactions. The LLM is designed to simulate four distinct agents, each responsible for a different aspect of the stage play. The architecture is built on top of transformer-based models, leveraging pre-trained models and fine-tuning them for specific roles in the play.

## Model Description

### Base Model

- **Model Type:** Transformer
- **Pre-trained Models Used:**
  - GPT-3 for generative tasks
  - BERT for understanding and classification tasks

### Agent-Specific Enhancements

#### Agent 1: Dialog Generator

- **Purpose:** Generates dialogues based on the current context of the play.
- **Architecture:**
  - Base: GPT-3
  - Modifications: Additional layers for emotional tone adjustment.

#### Agent 2: Context Analyzer

- **Purpose:** Analyzes the stage play's context to provide cues for other agents.
- **Architecture:**
  - Base: BERT
  - Modifications: Custom layers for extracting stage-specific context features.

#### Agent 3: Action Suggester

- **Purpose:** Suggests actions to actors based on the script and current dialogue.
- **Architecture:**
  - Base: GPT-3
  - Modifications: Integration with an external knowledge base about stage directions.

#### Agent 4: Audience Engagement Monitor

- **Purpose:** Monitors and reacts to audience engagement in real-time.
- **Architecture:**
  - Base: LSTM (Long Short-Term Memory)
  - Modifications: Real-time analysis modules for sentiment analysis.

## Integration

- **Data Flow:** The agents receive inputs from both the live stage environment and pre-processed scripts. Outputs from each agent are fed into a central coordinator, which synthesizes responses and directs them back to the stage directions or dialogue generators.
- **APIs Used:** TensorFlow, PyTorch for model operations. Flask for creating a lightweight API for real-time interactions.

## Training and Fine-tuning

- **Training Data Sources:** Script data, historical play performances, audience feedback logs.
- **Fine-tuning Approach:** Each agent is fine-tuned using specific scenarios extracted from past plays, augmented with synthetic data generated to cover rare events.

## Performance Metrics

- **Evaluation Methods:** Each agent's performance is evaluated using a combination of scripted scenarios and live staged interactions in controlled environments.
- **Metrics:**
  - Dialogue quality (BLEU score, perplexity)
  - Context relevance (F1 score, accuracy)
  - Action appropriateness (human ratings)
  - Audience engagement levels (sentiment analysis results, engagement metrics)

## Conclusion

The multi-agent LLM architecture is designed to be robust and adaptive, capable of handling the dynamic nature of live stage plays. By leveraging state-of-the-art AI technology, the model aims to enhance the theatrical experience both for performers and audiences alike.




# Local Hardware Configuration Guide for Large Model Training

## English Version

### Server/Compute Nodes

- **CPU**: At least 16-core processors such as Intel Xeon or AMD EPYC to handle multiple tasks simultaneously.
- **GPU**: At least 2 NVIDIA Tesla V100 or A100 GPUs, suitable for deep learning training and inference.
- **RAM**: At least 128GB, more is better, to support large datasets and models.
- **Storage**: At least 2TB SSD + 10TB HDD for fast data read during training and for storing large amounts of data.

### Network Devices

- **High-Speed Ethernet Switch**: At least 10Gbps connection to ensure fast data transmission.
- If distributed training is involved, ensure that the network bandwidth and latency meet the inter-node communication needs.

### Edge Devices (if agents need to interact with the real world)

- Deploy a certain number of edge computing devices such as Raspberry Pi 4 or NVIDIA Jetson series for real-time data processing and control.
- **Sensors**: Select sensors (e.g., cameras, temperature sensors, distance sensors) according to the perception needs of the agents.
- **Actuators**: Select actuators (e.g., motors, servos, LED lights) according to the action needs of the agents.

### Auxiliary Devices

- **Uninterruptible Power Supply (UPS)**: Ensure that critical hardware continues to operate during power outages.
- **Server Rack**: For mounting and protecting servers and network equipment.
- **Cooling System**: Ensure that hardware devices operate at appropriate temperatures.

### Software and Development Tools

- **Operating System**: Choose a stable server operating system such as Ubuntu Server or CentOS.
- **Development and Training Tools**: Ensure that the necessary deep learning frameworks (e.g., TensorFlow or PyTorch) and programming language environments (e.g., Python) are installed and configured.

### Security Measures

- **Network Security**: Set up firewalls and network isolation, especially when the system is connected to external networks.
- **Physical Security**: Protect critical hardware from unauthorized access.

# Evaluation Metrics for Dialogue Chatbots and Multi-Agent Systems Based on Large Models

Below are the key parameters and parameter counts for two different sizes of Gemma models used to evaluate the performance of dialogue chatbots and multi-agent systems.

## Key Model Parameters

| Parameters                  | 2B         | 7B         |
|-----------------------------|------------|------------|
| d_model                     | 2048       | 3072       |
| Layers                      | 18         | 28         |
| Feedforward hidden dims     | 32768      | 49152      |
| Num heads                   | 8          | 16         |
| Num KV heads                | 1          | 16         |
| Head size                   | 256        | 256        |
| Vocab size                  | 256128     | 256128     |

**Table 1 | Key model parameters for different sizes of Gemma models.**

## Parameter Counts

| Model | Embedding Parameters | Non-embedding Parameters |
|-------|-----------------------|--------------------------|
| 2B    | 524,550,144           | 1,981,884,416            |
| 7B    | 786,825,216           | 7,751,248,896            |

**Table 2 | Parameter counts for both sizes of Gemma models.**

## Evaluation Metrics

### 1. Performance Metrics

| Metric                    | Description                                                  |
|---------------------------|--------------------------------------------------------------|
| Training Speed            | Time required per epoch, measured in hours or minutes.       |
| Inference Speed           | Number of requests processed per second.                     |
| GPU Utilization           | GPU usage percentage during training and inference.          |

### 2. Model Accuracy Metrics

| Metric                    | Description                                                  |
|---------------------------|--------------------------------------------------------------|
| Precision                 | The ratio of correctly predicted positive samples to all predicted positive samples. |
| Recall                    | The ratio of correctly predicted positive samples to all actual positive samples.     |
| F1 Score                  | The harmonic mean of precision and recall, balancing both metrics.                    |
| pass@1                    | Measures the model's ability to return the correct result in one attempt, typically used for answer accuracy. |

### 3. Resource Utilization Metrics

| Metric                    | Description                                                  |
|---------------------------|--------------------------------------------------------------|
| Memory Utilization        | Memory usage during training and inference.                  |
| Disk I/O                  | Disk read/write speed during training and inference, usually measured in MB/s. |
| Network Bandwidth         | Network bandwidth for inter-node communication in distributed training, usually measured in Gbps. |

### 4. Reliability and Stability Metrics

| Metric                    | Description                                                  |
|---------------------------|--------------------------------------------------------------|
| Crash Rate                | Frequency of crashes during training or inference.           |
| Restart Count             | Number of times the model needs to be restarted during training or inference. |
| Error Rate                | Frequency of errors during training or inference.            |

### 5. User Experience Metrics

| Metric                    | Description                                                  |
|---------------------------|--------------------------------------------------------------|
| Response Time             | Average time from when a user sends a request to when they receive a response, measured in milliseconds. |
| User Satisfaction         | User satisfaction with model responses, typically collected through surveys or rating systems. |

These metrics help comprehensively evaluate the performance, accuracy, resource utilization, reliability, and user experience of dialogue chatbots and multi-agent systems based on large models. Choose appropriate metrics for evaluation and optimization according to different application scenarios and needs.