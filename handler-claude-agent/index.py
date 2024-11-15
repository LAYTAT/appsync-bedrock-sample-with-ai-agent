from dotenv import load_dotenv
from src.agent import buildAgent
from src.claudeInvoker import claude_bedrock
import json
load_dotenv()
from src.chatResponder import ChatResponder

def handler(event, context):
    # Print the incoming event data
    print("Event received:", json.dumps(event, indent=2))
    
    # Connect and record user message
    chatResponder = ChatResponder(event['conversationData']['id'])
    
    try:
        # Check for AppSync connection
        if len(event['agentData']['actions']) == 0:
            print('Note: this FM handler is intended for use with an AppSync tool, but none was specified. Engaging in plain chat with Claude.')
            chat_message = claude_bedrock(event['chatString'])
            print("Chat message response:", chat_message)
            chatResponder.publish_agent_message(chat_message)
            return

        # Build the agent with provided data
        print("Building agent with parameters:")
        print("- GraphQL Endpoint:", event['agentData']['actions'][0]['resource'])
        print("- System Prompt:", event['agentData']['systemPrompt'])
        print("- Authorization Header:", event['headers']['authorization'])
        
        agent = buildAgent(
            graphql_endpoint=event['agentData']['actions'][0]['resource'],
            system=event['agentData']['systemPrompt'],
            authHeader=event['headers']['authorization']
        )
        
        # Run the agent with the provided schema and chat string
        result = agent.run('''
            The schema above is live. Please continue this conversation using it as needed. 
            Please use the schema and any previous queries to move towards a solution to the questions the human asks.
        ''' + event['chatString'])
        
        print("Agent run result:", result)
        chatResponder.publish_agent_message(result)

    except Exception as e:
        # Print the exception message
        print("An error occurred:", str(e))
        pass

    # Mark metadata as done responding
    print("Publishing stop responding message.")
    chatResponder.publish_agent_stop_responding()