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

## 中文版

### 服务器/计算节点

- **CPU**：至少 16 核心的处理器，例如 Intel Xeon 或 AMD EPYC，以便同时处理多个任务。
- **GPU**：至少 2 块 NVIDIA Tesla V100 或 A100 GPU，适合深度学习训练和推理。
- **RAM**：至少 128GB，更高更好，以支持大型数据集和模型。
- **存储**：至少 2TB SSD + 10TB HDD，用于快速读取训练数据和存储大量数据。

### 网络设备

- **高速以太网交换机**：至少支持 10Gbps 连接，以确保数据快速传输。
- 如果涉及分布式训练，确保网络带宽和延迟能满足多节点间通信的需求。

### 边缘设备（如果智能体需要与真实世界交互）

- 根据需要部署一定数量的边缘计算设备，如 Raspberry Pi 4 或 NVIDIA Jetson 系列，用于实时数据处理和控制。
- **传感器**：根据智能体的感知需求选择（如摄像头、温度传感器、距离传感器等）。
- **执行器**：根据智能体的行动需求选择（如电机、伺服器、LED灯等）。

### 辅助设备

- **不间断电源供应（UPS）**：以确保关键硬件在电源中断时仍能正常运行。
- **服务器机柜**：用于安装和保护服务器及网络设备。
- **冷却系统**：确保硬件设备运行在合适的温度下。

### 软件与开发工具

- **操作系统**：选择稳定的服务器操作系统，如 Ubuntu Server 或 CentOS。
- **开发和训练工具**：确保所需的深度学习框架（如 TensorFlow 或 PyTorch）和编程语言环境（如 Python）已安装配置。

### 安全措施

- **网络安全**：设置防火墙和网络隔离，尤其是当系统与外部网络连接时。
- **物理安全**：保护关键硬件不受未授权访问。

## 日本語版

### サーバー/計算ノード

- **CPU**: 複数のタスクを同時に処理するために、少なくとも16コアのプロセッサ（例：Intel XeonまたはAMD EPYC）。
- **GPU**: ディープラーニングの訓練と推論に適した、少なくとも2枚のNVIDIA Tesla V100またはA100 GPU。
- **RAM**: 大規模なデータセットとモデルをサポートするために、少なくとも128GB、できればもっと多く。
- **ストレージ**: トレーニングデータの高速読み取りおよび大量データの保存のために、少なくとも2TBのSSD + 10TBのHDD。

### ネットワークデバイス

- **高速イーサネットスイッチ**: 少なくとも10Gbpsの接続をサポートし、データの高速転送を確保。
- 分散トレーニングが関係する場合、ノード間通信のニーズを満たすネットワーク帯域幅と遅延を確保。

### エッジデバイス（エージェントが現実世界と相互作用する必要がある場合）

- リアルタイムデータ処理と制御のために、Raspberry Pi 4またはNVIDIA Jetsonシリーズなどのエッジコンピューティングデバイスを必要に応じて配置。
- **センサー**: エージェントの感知ニーズに応じてセンサー（例：カメラ、温度センサー、距離センサーなど）を選択。
- **アクチュエータ**: エージェントの行動ニーズに応じてアクチュエータ（例：モーター、サーボ、LEDライトなど）を選択。

### 補助デバイス

- **無停電電源装置（UPS）**: 電源障害時に重要なハードウェアが正常に動作し続けることを確保。
- **サーバーラック**: サーバーおよびネットワーク機器の取り付けと保護のため。
- **冷却システム**: ハードウェアデバイスが適切な温度で動作することを確保。

### ソフトウェアと開発ツール

- **オペレーティングシステム**: 安定したサーバーオペレーティングシステム（例：Ubuntu ServerまたはCentOS）を選択。
- **開発とトレーニングツール**: 必要なディープラーニングフレームワーク（例：TensorFlowまたはPyTorch）およびプログラミング言語環境（例：Python）がインストールおよび構成されていることを確認。

### セキュリティ対策

- **ネットワークセキュリティ**: システムが外部ネットワークに接続されている場合、ファイアウォールとネットワーク分離を設定。
- **物理的セキュリティ**: 重要なハードウェアを不正アクセスから保護。
