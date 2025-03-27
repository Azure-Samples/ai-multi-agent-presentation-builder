import datetime
import os, json, re
from jinja2 import Environment, FileSystemLoader
from openai import AzureOpenAI
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import (
    KernelFunctionSelectionStrategy,
)
from semantic_kernel.functions import KernelFunctionFromMethod
from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import (
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt

from azure.identity import DefaultAzureCredential
from semantic_kernel.connectors.search_engine import BingConnector
from semantic_kernel.core_plugins import WebSearchEnginePlugin
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.exceptions.function_exceptions import FunctionExecutionException
from azure.core.exceptions import ClientAuthenticationError

from src.plugins.presentation import PresentationPlugin


def evaluate_logic():
    return "Logical Evaluation Complete."


class Orchestrator:

    def __init__(self, user_input, num_agents):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_version="2024-12-01-preview"
        )

        self.env = Environment(loader=FileSystemLoader(os.getenv('TEMPLATE_DIR_PROMPTS', '')))
        self.template = self.env.get_template(os.getenv('TEMPLATE_SYSTEM_ORCHESTRATOR', ''))
        self.theme = user_input
        self.num_agents = num_agents

    def get_response(self):
        response = self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL_ORCHESTRATOR", ""),
            messages=[
                {
                    "role": "user",
                    "content": self.template.render(
                        theme=self.theme,
                        num_agents=self.num_agents
                    )
                },
            ],
            max_completion_tokens=5000
        )
        return response

    def parse_response(self, response):
        json_response = json.loads(response.model_dump_json())
        json_response = json.loads(json_response['choices'][0]['message']['content'].replace('```json\n', '').replace('```', ''))
        return json_response

    def get_dynamic_agents(self, json_response):
        agents = []
        for agent_info in json_response.get('agents', []):
            agent = {
                'name': agent_info['name'],
                'role': agent_info['role'],
                'system_prompt': agent_info['system_prompt'],
            }
            agents.append(agent)
        return agents

    def run(self):
        response = self.get_response()
        json_response = self.parse_response(response)
        print(f'Creating dynamic agents who will be responsible for creating the presentation about: {self.theme}')
        dynamic_agents = self.get_dynamic_agents(json_response)
        return dynamic_agents


class ApprovalTerminationStrategy(KernelFunctionTerminationStrategy):
    """A strategy for determining when an agent should terminate."""

    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        return "approved" in history[-1].content.lower()


class MultiAgent:
    def __init__(self):
        self.model = os.getenv("AZURE_OPENAI_MODEL")
        self.bing_connector = BingConnector(os.getenv("BING_API_KEY"))
        self.env = Environment(loader=FileSystemLoader(os.getenv('TEMPLATE_DIR_PROMPTS')))
        self.template_termination = self.env.get_template(os.getenv('TEMPLATE_TERMINATION'))
        self.template_selection = self.env.get_template(os.getenv('TEMPLATE_SELECTION'))
    
    @staticmethod
    def _create_kernel_with_chat_completion(service_id: str, deployment_name: str) -> Kernel:
        kernel = Kernel()
        kernel.add_service(AzureChatCompletion(
            service_id=service_id,
            deployment_name=deployment_name
        ))
        return kernel

    @staticmethod
    def _standardize_string(input_string: str) -> str:
        return re.sub(r'[^0-9A-Za-z_-]', '_', input_string)

    def create_agents(self, dynamic_agents):
        expert_agents = []
        for agent in dynamic_agents:

            agent_name = self._standardize_string(agent['name'])
            kernel = self._create_kernel_with_chat_completion(agent_name, self.model)

            # Configure the function choice behavior to auto invoke kernel functions
            settings = kernel.get_prompt_execution_settings_from_service_id(service_id=agent_name)
            settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
            # Create a plugin and add it to the Kernel
            kernel.add_plugin(KernelFunctionFromMethod(evaluate_logic, "LogicEvaluator"))
            kernel.add_plugin(WebSearchEnginePlugin(BingConnector()), "WebSearch")
            kernel.add_plugin(PresentationPlugin(), "Presentation")

            expert = ChatCompletionAgent(service_id=agent_name,
                                         kernel=kernel,
                                         name=agent_name,
                                         instructions=agent['system_prompt'],
                                         execution_settings=settings            
                                        )
            expert_agents.append(expert)

            print(f"Created agent: {expert.name}, agent ID: {expert.id} ")
        return expert_agents

    def create_selection_function(self, expert_agents):
        selection_function = KernelFunctionFromPrompt(
            function_name="selection",
            prompt=self.template_selection.render(
                expert_agents=expert_agents,
                history=f"{{{{$history}}}}"
            )
        )
        return selection_function

    def create_termination_function(self, termination_keyword):
        selection_function = KernelFunctionFromPrompt(
            function_name="termination",
            prompt=self.template_termination.render(
                termination_keyword=termination_keyword,
                history=f"{{{{$history}}}}"
            )
        )

        return selection_function

    def create_chat_group(self, expert_agents, selection_function, termination_function, termination_keyword):
        group = AgentGroupChat(agents=expert_agents,
                selection_strategy=KernelFunctionSelectionStrategy(
                    function=selection_function,
                    kernel=self._create_kernel_with_chat_completion("selection", self.model),
                    result_parser=lambda result: str(result.value[0]) if result.value is not None else expert_agents[-1].name,
                    agent_variable_name="agents",
                    history_variable_name="history",
                ),
                termination_strategy=KernelFunctionTerminationStrategy(
                    agents=[expert_agents[-1]],
                    function=termination_function,
                    kernel=self._create_kernel_with_chat_completion("termination", self.model),
                    result_parser=lambda result: termination_keyword in str(result.value[0]).lower(),
                    history_variable_name="history",
                    maximum_iterations=2,
                ),
        )

        return group

    def auth_callback_factory(self, scope):
        auth_token = None
        async def auth_callback() -> str:
            """Auth callback for the SessionsPythonTool.
            This is a sample auth callback that shows how to use Azure's DefaultAzureCredential
            to get an access token.
            """
            nonlocal auth_token
            current_utc_timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())

            if not auth_token or auth_token.expires_on < current_utc_timestamp:
                credential = DefaultAzureCredential()

                try:
                    auth_token = credential.get_token(scope)
                except ClientAuthenticationError as cae:
                    err_messages = getattr(cae, "messages", [])
                    raise FunctionExecutionException(
                        f"Failed to retrieve the client auth token with messages: {' '.join(err_messages)}"
                    ) from cae

            return auth_token.token
        
        return auth_callback
