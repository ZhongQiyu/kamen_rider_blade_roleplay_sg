

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

# Stage Play Language Model Project Requirements Document

## Project Overview

This project aims to develop a large language model (LLM) specifically for a stage play, containing four to twelve independent agents, each capable of understanding and generating dialogues and actions related to their roles.

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

- See `requirements.txt` for a list of all required Python libraries.

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

# Stage Play Language Model Project Requirements Document

## Project Overview

This project aims to develop a large language model (LLM) specifically for a stage play, containing four to twelve independent agents, each capable of understanding and generating dialogues and actions related to their roles.

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

### Hardware Requirements

- **GPU**: NVIDIA RTX 2080 Ti or higher, at least 4 cards.
- **CPU**: Intel i7 or higher.
- **RAM**: At least 64GB.
- **Storage**: SSD at least 2TB.

### Dependency Libraries

- See `requirements.txt` for a list of all required Python libraries.

## Functional Requirements

### Agent Functions

- **Language Understanding**: Understand complex language inputs and respond.
- **Emotion Expression**: Express and understand emotions in dialogue.
- **Memory Ability**: Remember previous dialogues and actions.

### System Functions

- **User Interaction Interface**: A clean interface that allows users to interact and watch performances.
- **Performance Monitoring**: Monitor the performance of agents and the system.

## Security and Compliance

- Comply with GDPR and other data protection regulations.
- The system has security measures to prevent data leakage.

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

# Stage Play Multi-Agent Language Model Project

## Project Introduction

This project aims to develop a multi-agent large language model (LLM) that can interact specifically with a given stage play. The project includes four main agents, each capable of understanding and generating dialogue and actions relevant to their roles.

## Getting Started

### Environment Setup

Clone the repository to your local machine:
```bash
git clone https://github.com/your-repository/stage-play-llm.git
```

Prepare the data:

Place related stage play data in the data/raw/ directory and run the preprocessing script

```bash
python scripts/preprocessing.py
```

Start the model training process:
```bash
python scripts/train.py
```

Fine-tune the model according to specific needs:
```bash
python scripts/finetune.py
```

Evaluate the model's performance:
```bash
python scripts/evaluate.py
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Technology Stack
- TensorFlow or PyTorch
- LangChain
- Python 3.x

## Contributor's Guide
We welcome more developers to join our project. If you have any suggestions for improvement or would like to contribute code, please read the `CONTRIBUTING.md`.

## License
This project is licensed under the MIT license. For details, please see the LICENSE file.

## Contact Information
If you have any questions or need support, please contact xiaoyu991214@gmail.com.


This Markdown document provides a clear structure for your project documentation, offering potential contributors and users all the necessary information to get started, understand the technology stack, and know how to contribute or get in touch.

# Kamen Rider Blade Multi-Agent System User Guide

## 1. System Overview

This multi-agent system is designed based on the main characters of the Japanese tokusatsu drama "Kamen Rider Blade" and aims to provide a multifaceted problem-solving and decision-support tool. The system includes four core agents, each simulating the characteristics and abilities of a character from the drama, to assist users in handling complex problems and situations.

## 2. Introduction to Agents

### 2.1 Kazuma Kenzaki Agent
- Main Function: Decision Making and Action Execution
- Characteristics: Brave, Strong Sense of Justice, Highly Adaptable
- Use Cases: Situations requiring quick decisions and decisive actions

### 2.2 Sakuya Tachibana Agent
- Main Function: Strategic Analysis and Risk Assessment
- Characteristics: Calm, Rational, Cautious
- Use Cases: Situations requiring detailed analysis and long-term planning

### 2.3 Hajime Aikawa Agent
- Main Function: Information Gathering and Processing
- Characteristics: Keen, Observant, Flexible
- Use Cases: Situations requiring extensive information gathering and detailed insights

### 2.4 Mutsuki Kamijo Agent
- Main Function: Support Coordination and Emotion Analysis
- Characteristics: Gentle, Compassionate, Insightful
- Use Cases: Situations involving interpersonal relationships and emotional factors

## 3. How to Use the System

### 3.1 System Initialization

```python
from blade_agents import BladeAgentSystem

system = BladeAgentSystem()
system.initialize()
```

### 3.2 Setting the Problem or Task

```python
problem = "How to improve internal communication efficiency in the company?"
system.set_task(problem)
```

### 3.3 Activating Agents and Obtaining Feedback

```python
# Kazuma Kenzaki Agent's Decision
decision = system.activate_agent("kazuki")
print("Kazuma's Decision:", decision)

# Sakuya Tachibana Agent's Analysis
analysis = system.activate_agent("tachibana")
print("Sakuya's Analysis:", analysis)

# Hajime Aikawa Agent's Information
intel = system.activate_agent("aikawa")
print("Hajime's Information:", intel)

# Mutsuki Kamijo Agent's Coordination Advice
coordination = system.activate_agent("mutsuki")
print("Mutsuki's Coordination Advice:", coordination)
```

### 3.4 Comprehensive Analysis

```python
final_solution = system.synthesize_solutions()
print("Final Solution:", final_solution)
```

## 4. Application Scenarios

### 4.1 Project Management

```python
system.set_task("How to optimize the progress of a software development project?")
```

### 4.2 Market Strategy

```python
system.set_task("What marketing strategy should be adopted for a new product?")
```

### 4.3 Crisis Management

```python
system.set_task("How should the company respond to a public relations crisis?")
```

### 4.4 Team Building

```python
system.set_task("How to improve team cohesion and work efficiency?")
```

## 5. Notes

Each agent provides advice based on a specific perspective.
Final decisions should be made considering all outputs. The system's advice is for reference purposes.
Decisions should be made based on specific situations.
Regularly update the system's knowledge base to ensure agents provide the latest and most relevant advice.
It is recommended to consult human experts' opinions when making highly confidential or significant decisions.

## 6. Customization and Extension

### 6.1 Adjusting Agent Parameters

```python
system.customize_agent("kazuki", risk_tolerance=0.8)
system.customize_agent("tachibana", analysis_depth="high")
```

### 6.2 Adding New Capabilities

```python
system.add_new_capability("aikawa", "social_media_analysis")
```

### 6.3 Creating New Agents

```python
system.create_new_agent("hirose", role="technical_expert")
```

## 7. Troubleshooting

If the system response is slow or unexpected results occur, try the following steps:
- Reinitialize the system
- Ensure the problem description is clear and specific
- Adjust agent parameters
- Update the system's knowledge base
- Contact the technical support team for assistance

## 8. Conclusion

The Kamen Rider Blade Multi-Agent System provides unique perspectives for solving complex problems.
By simulating the thinking patterns of different characters, it helps users comprehensively analyze problems and make more informed decisions.
We hope this system becomes a powerful assistant in your work, bringing new ideas and inspiration.
If you have any questions or suggestions, please feel free to contact our support team. Enjoy using the system!

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