from dotenv import load_dotenv
from src.agent import buildAgent
from src.claudeInvoker import claude_bedrock
import json
load_dotenv()
from src.chatResponder import ChatResponder
import traceback

def handler(event, context):
    # Print the incoming event data
    print("Event received:", json.dumps(event, indent=2))
    
    # Connect and record user message
    chatResponder = ChatResponder(event['conversationData'].get('id'))
    
    try:
        # Get actions from event data
        actions = event['agentData'].get('actions', [])
        
        if actions:
            action = actions[0]
            action_resource = action.get('resource')

            # Build the agent with provided data
            print("Building agent with parameters:")
            print("- GraphQL Endpoint:", action_resource)
            print("- System Prompt:", event['agentData'].get('systemPrompt', ''))
            print("- Authorization Header:", event['headers'].get('authorization', ''))
            
            agent = buildAgent(
                graphql_endpoint=action_resource,
                system=event['agentData'].get('systemPrompt', ''),
                authHeader=event['headers'].get('authorization', '')
            )

            # Run the agent with the provided schema and chat string
            result = agent.run(
                f'''
                The schema above is live. Please continue this conversation using it as needed. 
                Please use the schema and any previous queries to move towards a solution to the questions the human asks.
                {event['chatString']}
                '''
            )
            
            print("Agent run result:", result)
            chatResponder.publish_agent_message(result)
        else:
            # Handle case where no actions are available
            print('No actions specified. Engaging in plain chat with Claude.')
            chat_message = claude_bedrock(event['chatString'])
            print("Chat message response:", chat_message)
            chatResponder.publish_agent_message(chat_message)

    except Exception as e:
        # Print the exception message
        print("An error occurred:", str(e))
        traceback.print_exc()

    # Mark metadata as done responding
    print("Publishing stop responding message.")
    chatResponder.publish_agent_stop_responding()