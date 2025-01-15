from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List

from azure.ai.projects import AIProjectClient, models
from azure.identity import DefaultAzureCredential

from src.agents._meta import FoundryAgent


class AgentFactory(ABC):
    """
    The Creator class declares the factory method that is supposed to return an
    object of a Product class. The Creator's subclasses usually provide the
    implementation of this method.
    """

    def __init__(self):
        self.project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=os.getenv("PROJECT_CONNECTION_STRING", "")
        )

    def create_agent(self, tools: List[models.AsyncFunctionTool]) -> models.Agent:
        """
        Note that the Creator may also provide some default implementation of the factory method.
        """
        with self.project_client as agent_client:
            agent: models.Agent = agent_client.agents.create_agent(
                model="gpt-4o-mini",
                name="my-agent",
                instructions="You are helpful agent",
                tools=[tool.definitions for tool in tools],
                tool_resources=[tool.resources for tool in tools]
            )
        return agent

    @abstractmethod
    def factory_method(self, agents: List[models.Agent]) -> List[FoundryAgent]:
        pass

    def __call__(self, *args, **kwargs) -> str:
        """
        Also note that, despite its name, the Creator's primary responsibility
        is not creating products. Usually, it contains some core business logic
        that relies on Product objects, returned by the factory method.
        Subclasses can indirectly change that business logic by overriding the
        factory method and returning a different type of product from it.
        """
        product = self.create_agent(*args, **kwargs)
        return result


class MultiAgentContext(ABC):

    def __init__(self) -> None:
        self.agents: List[models.Agent] = []

    def add_agent(self, agent: models.Agent) -> None:
        self.agents.append(agent)


class ImplementedFunctionFactory(AgentFactory):

    def create_agent(self, tools: List[models.AsyncFunctionTool]) -> models.Agent:
        return super().create_agent(tools)


class CreationalFunctionFactory(AgentFactory):

    def create_agent(self, tools: List[models.AsyncFunctionTool]) -> models.Agent:
        return super().create_agent(tools)
