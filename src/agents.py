"""Module containing agent configurations for financial analysis."""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from src.data_loader import load_prompt_template

STOP_AFTER_ATTEMPT = 3
WAIT_EXPONENTIAL_JITTER = True


class StepsAndAnswer(BaseModel):
    """
    Model representing the structure of a financial analysis response.

    Attributes:
        steps (List[str]): Calculation steps.
        answer (str): Final numerical answer.
    """

    steps: list[str] = Field(..., description="Calculation steps for the analysis")
    answer: str = Field(..., description="Final numerical answer")


class FinancialAnalysisAgents:
    """
    Class containing agent configurations for financial analysis workflow.
    """

    @staticmethod
    def get_financial_analyst_prompt() -> ChatPromptTemplate:
        """
        Create a system prompt for the financial analyst agent.

        Returns:
            ChatPromptTemplate: Configured prompt for financial analysis.
        """
        financial_analyst_system_message = load_prompt_template("financial_analyst")

        return ChatPromptTemplate.from_messages(
            [
                ("system", financial_analyst_system_message),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

    @staticmethod
    def get_critic_prompt() -> ChatPromptTemplate:
        """
        Create a system prompt for the critic agent.

        Returns:
            ChatPromptTemplate: Configured prompt for critical analysis.
        """
        critic_system_message = load_prompt_template("critic")

        return ChatPromptTemplate.from_messages(
            [
                ("system", critic_system_message),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

    @classmethod
    def create_agents(cls, model: str = "gpt-4o", temperature: float = 0.0):
        """
        Create agents for financial analysis workflow.

        Args:
            model (str, optional): LLM model to use. Defaults to "gpt-4o".
            temperature (float, optional): Sampling temperature. Defaults to 0.0.

        Returns:
            Tuple containing generator and reflection agents.
        """
        llm = AzureChatOpenAI(model=model, temperature=temperature)

        parser = JsonOutputParser(pydantic_object=StepsAndAnswer)

        generate = cls.get_financial_analyst_prompt() | llm.with_retry(
            stop_after_attempt=STOP_AFTER_ATTEMPT,
            wait_exponential_jitter=WAIT_EXPONENTIAL_JITTER,
        )

        reflect = cls.get_critic_prompt() | llm.with_retry(
            stop_after_attempt=STOP_AFTER_ATTEMPT,
            wait_exponential_jitter=WAIT_EXPONENTIAL_JITTER,
        )

        return generate, reflect, parser
