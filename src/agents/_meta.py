from __future__ import annotations

import logging

from collections.abc import AsyncGenerator, AsyncIterable
from typing import TYPE_CHECKING, Any, ClassVar

from azure.ai.projects import models
from semantic_kernel.agents import Agent

from semantic_kernel.agents.channels.agent_channel import AgentChannel
from semantic_kernel.agents.channels.chat_history_channel import ChatHistoryChannel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.const import DEFAULT_SERVICE_NAME
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.exceptions import KernelServiceNotFoundError
from semantic_kernel.utils.experimental_decorator import experimental_class

if TYPE_CHECKING:
    from semantic_kernel.kernel import Kernel

logger: logging.Logger = logging.getLogger(__name__)


@experimental_class
class FoundryAgent(Agent):
    """A KernelAgent specialization based on ChatCompletionClientBase, integrated with Azure AI Foundry Agent.

    Note: enable `function_choice_behavior` on the PromptExecutionSettings to enable function
    choice behavior which allows the kernel to utilize plugins and functions registered in
    the kernel.
    """

    service_id: str
    execution_settings: PromptExecutionSettings | None = None
    channel_type: ClassVar[type[AgentChannel]] = ChatHistoryChannel

    def __init__(
        self,
        foundry_agent: models.Agent,
        foundry_thread: models.AgentThread,
        service_id: str | None = None,
        kernel: "Kernel | None" = None,
        execution_settings: PromptExecutionSettings | None = None,
    ) -> None:
        """Initialize a new instance of FoundryAgent.

        Args:
            service_id: The service id for the chat completion service. (optional) If not provided,
                the default service name `default` will be used.
            kernel: The kernel instance. (optional)
            execution_settings: The execution settings for the agent. (optional)
        """
        self.foundry_agent = foundry_agent
        self.foundry_thread = foundry_thread

        if not service_id:
            service_id = DEFAULT_SERVICE_NAME

        args: dict[str, Any] = {
            "service_id": service_id,
            "execution_settings": execution_settings,
            "description": self.foundry_agent.description,
            "instructions": self.foundry_agent.instructions,
            "name": self.foundry_agent.name,
            "id": self.foundry_agent.id,
        }
        if kernel is not None:
            args["kernel"] = kernel
        super().__init__(**args)

    async def invoke(self, history: ChatHistory) -> AsyncIterable[ChatMessageContent]:
        """Invoke the chat history handler.

        Args:
            kernel: The kernel instance.
            history: The chat history.

        Returns:
            An async iterable of ChatMessageContent.
        """
        # Get the chat completion service
        chat_completion_service = self.kernel.get_service(service_id=self.service_id, type=ChatCompletionClientBase)

        if not chat_completion_service:
            raise KernelServiceNotFoundError(f"Chat completion service not found with service_id: {self.service_id}")

        if not isinstance(chat_completion_service, ChatCompletionClientBase):
            raise TypeError("Chat completion service must be an instance of ChatCompletionClientBase")

        settings = (
            self.execution_settings
            or self.kernel.get_prompt_execution_settings_from_service_id(self.service_id)
            or chat_completion_service.instantiate_prompt_execution_settings(
                service_id=self.service_id, extension_data={"ai_model_id": chat_completion_service.ai_model_id}
            )
        )

        chat = self._setup_agent_chat_history(history)

        message_count = len(chat)

        logger.debug("[%s] Invoking %s.", type(self).__name__, type(chat_completion_service).__name__)

        messages = await chat_completion_service.get_chat_message_contents(
            chat_history=chat,
            settings=settings,
            kernel=self.kernel,
        )

        logger.info(
            "[%s] Invoked %s with message count: %d.",
            type(self).__name__,
            type(chat_completion_service).__name__,
            message_count,
        )

        for message_index in range(message_count, len(chat)):
            message = chat[message_index]
            message.name = self.name
            history.add_message(message)

        for message in messages:
            message.name = self.name
            yield message

    async def invoke_stream(self, history: ChatHistory) -> AsyncIterable[StreamingChatMessageContent]:
        """Invoke the chat history handler in streaming mode.

        Args:
            kernel: The kernel instance.
            history: The chat history.

        Returns:
            An async generator of StreamingChatMessageContent.
        """
        # Get the chat completion service
        chat_completion_service = self.kernel.get_service(service_id=self.service_id, type=ChatCompletionClientBase)

        if not chat_completion_service:
            raise KernelServiceNotFoundError(f"Chat completion service not found with service_id: {self.service_id}")

        if not isinstance(chat_completion_service, ChatCompletionClientBase):
            raise TypeError("Chat completion service must be an instance of ChatCompletionClientBase")

        settings: PromptExecutionSettings = (
            self.execution_settings
            or self.kernel.get_prompt_execution_settings_from_service_id(self.service_id)
            or chat_completion_service.instantiate_prompt_execution_settings(
                service_id=self.service_id, extension_data={"ai_model_id": chat_completion_service.ai_model_id}
            )
        )

        chat = self._setup_agent_chat_history(history)

        message_count = len(chat)

        logger.debug("[%s] Invoking %s.", type(self).__name__, type(chat_completion_service).__name__)

        messages: AsyncGenerator[list[StreamingChatMessageContent], Any] = (
            chat_completion_service.get_streaming_chat_message_contents(
                chat_history=chat,
                settings=settings,
                kernel=self.kernel,
            )
        )

        logger.info(
            "[%s] Invoked %s with message count: %d.",
            type(self).__name__,
            type(chat_completion_service).__name__,
            message_count,
        )

        role = None
        message_builder: list[str] = []
        async for message_list in messages:
            for message in message_list:
                role = message.role
                message.name = self.name
                message_builder.append(message.content)
                yield message

        # Capture mutated messages related function calling / tools
        for message_index in range(message_count, len(chat)):
            message = chat[message_index]  # type: ignore
            message.name = self.name
            history.add_message(message)

        if role != AuthorRole.TOOL:
            history.add_message(
                ChatMessageContent(
                    role=role if role else AuthorRole.ASSISTANT, content="".join(message_builder), name=self.name
                )
            )

    def __sync_chat_history(self, history: ChatHistory) -> None:
        """Sync the chat history."""
        for message in history.messages:
            message.name = self.name

    def _setup_agent_chat_history(self, history: ChatHistory) -> ChatHistory:
        """Setup the agent chat history."""
        chat = []

        if self.instructions is not None:
            chat.append(ChatMessageContent(role=AuthorRole.SYSTEM, content=self.instructions, name=self.name))

        chat.extend(history.messages if history.messages else [])

        return ChatHistory(messages=chat)
