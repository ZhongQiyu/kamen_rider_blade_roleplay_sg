# Stage Play Multi-Agent Language Model Project

## Introduction

All characters in the TV Series of Kamen Rider Blade (仮面ライダー剣 in 2004). I inherit the work from the first training session in InternLM (书生·浦语) in 2024 to build this system. This is a repository for building the agent system that enables chatting features with the copies the stage. It mirrors my understanding towards a large part of my life so far, while working as an AI project that attempts to chain my learning path in computer science over the years too.

## Overview

This project aims to develop a large language model (LLM) specifically for a stage play, containing four to twelve independent agents, each capable of understanding and generating dialogues and actions related to their roles.

This document provides detailed information about the interfaces of the Stage Play LLM API. It is intended for developers who need to integrate the LLM models into stage play applications. A detailed overview of the model architecture used in our large language model (LLM) for stage play interactions.

The LLM is designed to simulate four distinct agents, each responsible for a different aspect of the stage play. The architecture is built on top of transformer-based models, leveraging pre-trained models and fine-tuning them for specific roles in the play.

This project aims to develop a large language model (LLM) specifically for a stage play. The project will include four to twelve main agents, each capable of understanding and generating dialogues and actions related to their roles.

## System Overview

This multi-agent system is based on the main characters from the Japanese tokusatsu series "Kamen Rider Blade". It aims to provide a versatile problem-solving and decision-support tool. The system includes four core agents, each simulating the characteristics and abilities of the characters in the series, to help users handle complex problems and situations.

## Methodologies

### Web Page Configurations

<div align="center">
<img src="./assets/method.gif" alt="Method" title="method">
</div>

## Technical Requirements

### Software Requirements

### Hardware Requirements

- GPU: NVIDIA RTX 2080 Ti or higher, at least 4 cards.
- CPU: Intel i7 or higher.
- RAM: At least 64GB.
- Storage: SSD at least 2TB.

### Dependency Libraries

- Python 3.8+: Primary programming language.
- TensorFlow 2.x / PyTorch 1.8+: For model training and inference.
- Flask/Django: For API development and deployment.
- Docker: For application packaging and deployment.
- torch: Deep learning framework for dynamic neural networks.
- transformers: Library for state-of-the-art natural language processing.
- pysrt: Python library for parsing and modifying SubRip (.srt) subtitle files.

See requirements.txt for a list of all required Python libraries.

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

- Language Understanding: Understand complex language inputs and respond.
- Emotion Expression: Express and understand emotions in dialogue.
- Memory Ability: Remember previous dialogues and actions.

### System Functions

- User Interaction Interface: A clean interface that allows users to interact and watch performances.
- Performance Monitoring: Monitor the performance of agents and the system.

### Security and Compliance

- Comply with GDPR and other data protection regulations.
- The system has security measures to prevent data leakage.

### Testing Requirements

- Unit Testing: Test key functions.
- Integration Testing: Ensure system components work together.
- Performance Testing: System performance under high load.
- User Acceptance Testing: Ensure compliance with user expectations and requirements.

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
```

## Getting Started

### 1. Clone the repository to your local machine:

```bash
git clone https://github.com/your-repository/stage-play-llm.git
```

### 2. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 3. Data preparation: Place the stage play related data in the `data/raw/` directory and run the preprocessing script:

```bash
python scripts/preprocessing.py
```

### 4. Model training: Start the model training process:

```bash
python scripts/train.py
```

### 5. Model fine-tuning: Fine-tune the model according to specific needs:

```bash
python scripts/finetune.py
```

### 6. Model evaluation: Evaluate the model performance:

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

## Project Description

This project aims to develop a large language model (LLM) for stage plays, containing four to twelve independent agents, each capable of understanding and generating dialogues and actions related to their roles.

### Goals

- Train a language model with four to twelve agents.
- Each agent should be able to accurately generate dialogues based on the plot and character backgrounds.
- Implement effective interactions between agents to support the unfolding of complex plots.

### Final Deliverables

- Autonomous performing agents.
- A complete deployment solution including backend model deployment and frontend interaction interface.

### Milestones and Delivery Schedule

- May 2024: System framework and agent prototypes completed.
- June 2024: Complete agent training and preliminary testing.
- July 2024: System integration and comprehensive testing.
- August 2024: User acceptance testing and deployment preparation.
- September 2024: Project officially goes live.

### Goals

- Train a language model with four to twelve agents.
- Each agent should be able to accurately generate dialogues based on the plot and character backgrounds.
- Implement effective interactions between agents to support the unfolding of complex plots.

### Final Deliverables

- Autonomous performing agents.
- A complete deployment solution including backend model deployment and frontend interaction interface.

## Reference

## Acknowledgement

## Technical Requirements

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



# Model Architecture for Stage Play LLM

## Overview

## Model Description

## Base Model

- **Model Type:** Transformer
- **Pre-trained Models Used:**
  - GPT-3 for generative tasks
  - BERT for understanding and classification tasks

## Agent-Specific Enhancements

### Agent 1: Dialog Generator

- **Purpose:** Generates dialogues based on the current context of the play.
- **Architecture:**
  - Base: GPT-3
  - Modifications: Additional layers for emotional tone adjustment.

### Agent 2: Context Analyzer

- **Purpose:** Analyzes the stage play's context to provide cues for other agents.
- **Architecture:**
  - Base: BERT
  - Modifications: Custom layers for extracting stage-specific context features.

### Agent 3: Action Suggester

- **Purpose:** Suggests actions to actors based on the script and current dialogue.
- **Architecture:**
  - Base: GPT-3
  - Modifications: Integration with an external knowledge base about stage directions.

### Agent 4: Audience Engagement Monitor

- **Purpose:** Monitors and reacts to audience engagement in real-time.
- **Architecture:**
  - Base: LSTM (Long Short-Term Memory)
  - Modifications: Real-time analysis modules for sentiment analysis.

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

## Functional Requirements

### Agent Functions

- Language Understanding: Understand complex language inputs and respond.
- Emotion Expression: Express and understand emotions in dialogue.
- Memory Ability: Remember previous dialogues and actions.

### System Functions

- User Interaction Interface: A clean interface that allows users to interact and watch performances.
- Performance Monitoring: Monitor the performance of agents and the system.

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

## Example Usage

### Asynchronous Agent Communication

#### Initiate Asynchronous Communication
```http
POST /agent_comm
Content-Type: application/json

{
  "agent_id": "agent_123",
  "message": "Hello, Agent!",
  "callback_url": "https://example.com/callback"
}
```

#### Response:
```json
{
  "confirmation_message": "Asynchronous communication initiated."
}
```

### Retrieve Messages
```http
GET /agent_comm
Content-Type: application/json

{
  "agent_id": "agent_123"
}
```

#### Response:
```json
{
  "messages": [
    {
      "from": "agent_456",
      "message": "Hello, Agent 123!"
    },
    {
      "from": "agent_789",
      "message": "How can I assist you today?"
    }
  ]
}
```

### Security and Compliance

- Comply with GDPR and other data protection regulations.
- The system has security measures to prevent data leakage.

### Testing Requirements

- Unit Testing: Test key functions.
- Integration Testing: Ensure system components work together.
- Performance Testing: System performance under high load.
- User Acceptance Testing: Ensure compliance with user expectations and requirements.

### Milestones and Delivery Schedule

- May 2024: System framework and agent prototypes completed.
- June 2024: Complete agent training and preliminary testing.
- July 2024: System integration and comprehensive testing.
- August 2024: User acceptance testing and deployment preparation.
- September 2024: Project officially goes live.

## Conclusion

The Kamen Rider Blade Multi-Agent System provides a unique perspective on solving complex problems. By simulating the thinking patterns of different characters, it helps users comprehensively analyze problems and make more precise decisions. We hope this system becomes a powerful assistant in your work, bringing new ideas and inspiration. If you have any questions or suggestions, please feel free to contact our support team. Enjoy using it!
