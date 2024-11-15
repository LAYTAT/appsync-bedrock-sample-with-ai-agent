import boto3, json
from botocore.config import Config

bedrock = boto3.client('bedrock-runtime', config=Config(region_name='us-east-1'))

def process_model_response(raw_body):
    # Parse the raw JSON response
    response_json = json.loads(raw_body)
    
    # Access the response content
    content_list = response_json.get("content", [])
    if content_list:
        # Extract the text from the first content entry
        text_content = content_list[0].get("text")
        
        # Print the extracted text content
        print("Extracted Response Text:")
        print(text_content)
        
        # Return the extracted response text
        return text_content
    else:
        print("No content available in response.")
        return None

def claude_bedrock (prompt, stopWords = []):

    response = bedrock.invoke_model(
        body=json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "temperature": 0,
            "top_p": 0.999,
            "top_k": 250,
            "stop_sequences": []
        }),
        modelId='anthropic.claude-3-sonnet-20240229-v1:0'
    )


     # Read and decode the response body
    raw_body = response['body'].read().decode("utf-8")
    
    # Print the raw response body
    print("Raw response body:", raw_body)
    
    # Parse the JSON response
    response_json = json.loads(raw_body)
    
    return (process_model_response(raw_body))