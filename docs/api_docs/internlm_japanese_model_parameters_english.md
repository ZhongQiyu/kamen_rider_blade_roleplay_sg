
### InternLM Model

#### Key Model Parameters

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

#### Parameter Counts

| Model       | Embedding Parameters | Non-embedding Parameters |
|-------------|----------------------|--------------------------|
| Tiny Model  | 262,275,072          | 524,288,000              |
| Large Model | 1,048,576,128        | 4,194,304,000            |

**Table 2 | Parameter counts for various scales of InternLM models.**

#### Evaluation Metrics

##### 1. Performance Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Training Speed           | Time required per epoch, measured in hours or minutes.      |
| Inference Speed          | Number of requests processed per second.                    |
| GPU Utilization          | GPU usage percentage during training and inference.         |

##### 2. Model Accuracy Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Precision                | The ratio of correctly predicted positive samples to all predicted positive samples. |
| Recall                   | The ratio of correctly predicted positive samples to all actual positive samples.     |
| F1 Score                 | The harmonic mean of precision and recall, balancing both metrics.                    |
| pass@1                   | Measures the model's ability to return the correct result in one attempt, typically used for answer accuracy. |

##### 3. Resource Utilization Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Memory Utilization       | Memory usage during training and inference.                 |
| Disk I/O                 | Disk read/write speed during training and inference, usually measured in MB/s. |
| Network Bandwidth        | Network bandwidth for inter-node communication in distributed training, usually measured in Gbps. |

##### 4. Reliability and Stability Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Crash Rate               | Frequency of crashes during training or inference.          |
| Restart Count            | Number of times the model needs to be restarted during training or inference. |
| Error Rate               | Frequency of errors during training or inference.           |

##### 5. User Experience Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Response Time            | Average time from when a user sends a request to when they receive a response, measured in milliseconds. |
| User Satisfaction        | User satisfaction with model responses, typically collected through surveys or rating systems. |

These metrics help comprehensively evaluate the performance, accuracy, resource utilization, reliability, and user experience of dialogue chatbots and multi-agent systems based on large models. Choose appropriate metrics for evaluation and optimization according to different application scenarios and needs.

### Japanese Small Model

#### Key Model Parameters

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

#### Parameter Counts

| Model        | Embedding Parameters | Non-embedding Parameters |
|--------------|----------------------|--------------------------|
| Small Model  | 24,576,000           | 98,304,000               |
| Large Model  | 49,152,000           | 393,216,000              |

**Table 2 | Parameter counts for various scales of Japanese small models.**

#### Evaluation Metrics

##### 1. Performance Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Training Speed           | Time required per epoch, measured in hours or minutes.      |
| Inference Speed          | Number of requests processed per second.                    |
| GPU Utilization          | GPU usage percentage during training and inference.         |

##### 2. Model Accuracy Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Precision                | The ratio of correctly predicted positive samples to all predicted positive samples. |
| Recall                   | The ratio of correctly predicted positive samples to all actual positive samples.     |
| F1 Score                 | The harmonic mean of precision and recall, balancing both metrics.                    |
| pass@1                   | Measures the model's ability to return the correct result in one attempt, typically used for answer accuracy. |

##### 3. Resource Utilization Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Memory Utilization       | Memory usage during training and inference.                 |
| Disk I/O                 | Disk read/write speed during training and inference, usually measured in MB/s. |
| Network Bandwidth        | Network bandwidth for inter-node communication in distributed training, usually measured in Gbps. |

##### 4. Reliability and Stability Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Crash Rate               | Frequency of crashes during training or inference.          |
| Restart Count            | Number of times the model needs to be restarted during training or inference. |
| Error Rate               | Frequency of errors during training or inference.           |

##### 5. User Experience Metrics

| Metric                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| Response Time            | Average time from when a user sends a request to when they receive a response, measured in milliseconds. |
| User Satisfaction        | User satisfaction with model responses, typically collected through surveys or rating systems. |

These metrics help comprehensively evaluate the performance, accuracy, resource utilization, reliability, and user experience of dialogue chatbots and multi-agent systems based on large models. Choose appropriate metrics for evaluation and optimization according to different application scenarios and needs.

Hope these adjustments help to continue optimizing your model parameters.
