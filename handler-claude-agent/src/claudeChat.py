"""OpenAI chat wrapper."""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
)

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatGeneration, ChatMessage, ChatResult
from .claudeInvoker import claude_bedrock
from .chatResponder import ChatResponder

class ChatBedrockClaude(BaseChatModel):
    system_role: str = ''

    def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("Unimplemented method _agenerate")

    def _llm_type(self) -> str:
        return "claude-chat"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        if not messages:
            raise ValueError("No messages provided for claude prompt generation.")

        escaped_system_role = self.system_role.replace('Human: ', 'user:').replace('Assistant: ', 'agent:')
        claude_prompt = f'''Human: {escaped_system_role}\n{messages[0].content}\nAssistant: '''
        claude_result = claude_bedrock(claude_prompt, stop)

        # Log the received result for debugging
        print("Claude raw result:", claude_result)

        # Custom parsing to get inner dialog for chat
        claude_result_lower = claude_result.lower()

        if 'final answer:' in claude_result_lower:
            claude_result = claude_result_lower.split('final answer:', 1)[1]
        else:
            print("Warning: 'Final Answer:' not found in response.")

        if 'action:' in claude_result.lower():
            claude_thoughts = claude_result.split('Action:', 1)[0]

            # Safely handle splitting by 'Thought:'
            if 'Thought:' in claude_thoughts:
                thought_parts = claude_thoughts.split('Thought:', 1)
                if len(thought_parts) > 1:
                    claude_thoughts = thought_parts[1]
                else:
                    claude_thoughts = "No specific thought found."
            else:
                claude_thoughts = "No 'Thought:' part found in the response."

            ChatResponder.instance.publish_agent_dialog(claude_thoughts)
        else:
            print("Warning: 'Action:' not found in response.")

        if "Thought:" not in claude_result and "Final Answer:" not in claude_result and "Action:" not in claude_result:
            claude_result = f'Final Answer: {claude_result}'

        return ChatResult(generations=[ChatGeneration(message=ChatMessage(content=claude_result, role='assistant'))])