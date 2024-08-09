
# agent_comm

### Description
Communicates between agents asynchronously and facilitates multi-agent collaboration.

# data_processor API Documentation

## Description
Processes input data for analysis and supports asynchronous data processing for multi-agent systems.

## API Endpoints

### GET /data_processor
**Description**: Retrieves processed data.
**Parameters**:
- `data_id` (str): The unique identifier for the data being retrieved.
**Response**:
- `processed_data` (JSON): The processed data in JSON format.

### GET /messages
**Description**: Retrieves messages for the specified agent.
**Parameters**:
- `agent_id` (str): The unique identifier for the agent whose messages are being retrieved.
**Response**:
- `messages` (JSON): A list of messages received by the agent.

### POST /data_processor
**Description**: Submits data for processing.
**Parameters**:
- `raw_data` (JSON): The raw data to be processed.
**Response**:
- `processing_id` (str): The unique identifier for the processing job.

### POST /agent_comm
**Description**: Initiates communication between agents.
**Parameters**:
- `agent_id` (str): The unique identifier for the agent initiating the communication.
- `message` (str): The message to be sent to the recipient agent.
**Response**:
- `confirmation_message` (str): A message confirming that the communication was initiated.

### POST /data_processor (Asynchronous)
**Description**: Submits data for asynchronous processing.
**Parameters**:
- `raw_data` (JSON): The raw data to be processed.
- `callback_url` (str): URL to which the processed data will be sent asynchronously.
**Response**:
- `processing_id` (str): The unique identifier for the asynchronous processing job.

### POST /agent_comm (Asynchronous)
**Description**: Initiates asynchronous communication between agents.
**Parameters**:
- `agent_id` (str): The unique identifier for the agent initiating the communication.
- `message` (str): The message to be sent to the recipient agent.
- `callback_url` (str): URL to which the response will be sent asynchronously.
**Response**:
- `confirmation_message` (str): A message confirming that the asynchronous communication was initiated.

### GET /processing_status
**Description**: Retrieves the status of a processing job.
**Parameters**:
- `processing_id` (str): The unique identifier for the processing job.
**Response**:
- `status` (str): The current status of the processing job (e.g., 'pending', 'processing', 'completed').

### GET /processing_result
**Description**: Retrieves the result of a completed processing job.
**Parameters**:
- `processing_id` (str): The unique identifier for the processing job.
**Response**:
- `processed_data` (JSON): The processed data for the specified processing job.

## Example Usage

### Asynchronous Data Processing
#### Submit Data for Asynchronous Processing
```http
POST /data_processor
Content-Type: application/json

{
  "raw_data": {
    "field1": "value1",
    "field2": "value2"
  },
  "callback_url": "https://example.com/callback"
}
```
