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

### Download Link

To generate a download link for this markdown file:

1. **GitHub**:
   - Create a new repository or navigate to an existing one.
   - Upload the markdown file.
   - Once uploaded, the file will have a download link in the repository.

2. **Google Drive**:
   - Upload the markdown file to your Google Drive.
   - Share the file and get a shareable link.

3. **Dropbox**:
   - Upload the markdown file to your Dropbox.
   - Share the file and get a shareable link.

Feel free to choose the method that best suits your needs.

## InternLM Model

### Key Model Parameters

| Parameter                 | Tiny Model    | Large Model   |
|---------------------------|---------------|---------------|
| d_model                   | 1024          | 4096          |
| Layers                    | 12            | 24            |
| Feedforward Hidden Dims   | 16384         | 65536         |
| Num Heads                 | 4             | 32            |
| Num KV Heads              | 1             | 32            |
| Head Size                 | 128           | 128           |
| Vocab Size                | 128000        | 128000        |

**Table 1 | Key parameters for different scales of InternLM models.**

### Parameter Counts

| Model       | Embedding Parameters | Non-embedding Parameters |
|-------------|----------------------|--------------------------|
| Tiny Model  | 262,275,072          | 524,288,000              |
| Large Model | 1,048,576,128        | 4,194,304,000            |

**Table 2 | Parameter counts for various scales of InternLM models.**

### Performance Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Training Speed           | Time required per epoch, measured in hours or minutes.      |
| Inference Speed          | Number of requests processed per second.                    |
| GPU Utilization          | GPU usage percentage during training and inference.         |

### Model Accuracy Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Precision                | The ratio of correctly predicted positive samples to all predicted positive samples. |
| Recall                   | The ratio of correctly predicted positive samples to all actual positive samples.     |
| F1 Score                 | The harmonic mean of precision and recall, balancing both metrics.                    |
| pass@1                   | Measures the model's ability to return the correct result in one attempt, typically used for answer accuracy. |

### Resource Utilization Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Memory Utilization       | Memory usage during training and inference.                 |
| Disk I/O                 | Disk read/write speed during training and inference, usually measured in MB/s. |
| Network Bandwidth        | Network bandwidth for inter-node communication in distributed training, usually measured in Gbps. |

### Reliability and Stability Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Crash Rate               | Frequency of crashes during training or inference.          |
| Restart Count            | Number of times the model needs to be restarted during training or inference. |
| Error Rate               | Frequency of errors during training or inference.           |

### User Experience Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Response Time            | Average time from when a user sends a request to when they receive a response, measured in milliseconds. |
| User Satisfaction        | User satisfaction with model responses, typically collected through surveys or rating systems. |

These metrics help comprehensively evaluate the performance, accuracy, resource utilization, reliability, and user experience of dialogue chatbots and multi-agent systems based on large models. Choose appropriate metrics for evaluation and optimization according to different application scenarios and needs.

## Japanese Small Model

### Key Model Parameters
| Parameter                 | Small Model    | Large Model    |
|---------------------------|----------------|----------------|
| d_model                   | 768            | 1536           |
| Layers                    | 6              | 12             |
| Feedforward Hidden Dims   | 12288          | 24576          |
| Num Heads                 | 6              | 12             |
| Num KV Heads              | 1              | 12             |
| Head Size                 | 64             | 64             |
| Vocab Size                | 32000          | 32000          |

**Table 1 | Key parameters for different scales of Japanese small models.**

### Parameter Counts

| Model        | Embedding Parameters | Non-embedding Parameters |
|--------------|----------------------|--------------------------|
| Small Model  | 24,576,000           | 98,304,000               |
| Large Model  | 49,152,000           | 393,216,000              |

**Table 2 | Parameter counts for various scales of Japanese small models.**

### Performance Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Training Speed           | Time required per epoch, measured in hours or minutes.      |
| Inference Speed          | Number of requests processed per second.                    |
| GPU Utilization          | GPU usage percentage during training and inference.         |

### Model Accuracy Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Precision                | The ratio of correctly predicted positive samples to all predicted positive samples. |
| Recall                   | The ratio of correctly predicted positive samples to all actual positive samples.     |
| F1 Score                 | The harmonic mean of precision and recall, balancing both metrics.                    |
| pass@1                   | Measures the model's ability to return the correct result in one attempt, typically used for answer accuracy. |

### Resource Utilization Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Memory Utilization       | Memory usage during training and inference.                 |
| Disk I/O                 | Disk read/write speed during training and inference, usually measured in MB/s. |
| Network Bandwidth        | Network bandwidth for inter-node communication in distributed training, usually measured in Gbps. |

### Reliability and Stability Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Crash Rate               | Frequency of crashes during training or inference.          |
| Restart Count            | Number of times the model needs to be restarted during training or inference. |
| Error Rate               | Frequency of errors during training or inference.           |

### User Experience Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Response Time            | Average time from when a user sends a request to when they receive a response, measured in milliseconds. |
| User Satisfaction        | User satisfaction with model responses, typically collected through surveys or rating systems. |

These metrics help comprehensively evaluate the performance, accuracy, resource utilization, reliability, and user experience of dialogue chatbots and multi-agent systems based on large models. Choose appropriate metrics for evaluation and optimization according to different application scenarios and needs.

Hope these adjustments help to continue optimizing your model parameters.

