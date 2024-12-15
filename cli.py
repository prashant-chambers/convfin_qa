"""Main script for running financial analysis workflow."""

import argparse
import asyncio

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from src.agents import FinancialAnalysisAgents
from src.data_conversion import (
    convert_to_markdown_table,
    convert_to_paragraph,
    fix_invalid_json,
)
from src.data_loader import load_financial_data, load_prompt_template
from src.graph import FinancialAnalysisGraph


def temperature_range(value: str) -> float:
    """
    Validate that the temperature is between 0.0 and 1.0.
    """
    try:
        temp = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid float.")

    if not (0.0 <= temp <= 1.0):
        raise argparse.ArgumentTypeError(
            f"Temperature must be between 0.0 and 1.0, got {value}."
        )

    return temp


async def main(model: str, temperature: float, data_path: str, verbose: bool):
    """
    Main async function to run financial analysis workflow.
    """

    print(f"Running model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Data Path: {data_path}")

    # Load environment variables
    load_dotenv()

    # Create agents and graph
    generate, reflect, parser = FinancialAnalysisAgents.create_agents(
        model=model, temperature=temperature
    )
    graph = FinancialAnalysisGraph.create_graph(generate, reflect)

    # Process financial data
    for idx, data in enumerate(load_financial_data(data_path)):
        config = {"configurable": {"thread_id": f"{idx}"}}

        # Prepare context
        pre_text = convert_to_paragraph(data["pre_text"])
        post_text = convert_to_paragraph(data["post_text"])
        table = convert_to_markdown_table(data["table"])

        # Process questions
        question_answer = []
        for key in ["qa_0", "qa_1", "qa"]:
            if data.get(key):
                question_answer.append((data[key]["question"], data[key]["answer"]))

        # Analyze each question
        for question, expected_answer in question_answer:
            user_proxy_message = load_prompt_template(
                "user_proxy",
                question=question,
                pre_text=pre_text,
                table=table,
                post_text=post_text,
            )
            response = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=user_proxy_message)],
                },
                config,
            )

            # Filter messages with json from Agent conversation
            ai_messages = [
                x.content
                for x in response["messages"]
                if isinstance(x, AIMessage) and x.content.__contains__("json")
            ]

            # Select final json message from AI
            if len(ai_messages) >= 1:
                content = ai_messages[-1]
                parsed_content = parser.parse(fix_invalid_json(content))
                if verbose:
                    print(f"Record ID: {data['id']}")
                    print(f"Question: {question}")
                    print(f"Expected Answer: {expected_answer}")
                    print(f"Generated Answer: {parsed_content['answer']}")
                    print("-" * 50)

        # Optional: Break after processing a few records for testing
        if idx == 5:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a model with specific parameters asynchronously."
    )

    # Add arguments
    parser.add_argument(
        "--model", type=str, required=True, help="The name of the model to use."
    )
    parser.add_argument(
        "--temperature",
        type=temperature_range,
        required=True,
        help="The temperature setting (must be between 0.0 and 1.0).",
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="The path to the input data."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")

    # Parse arguments
    args = parser.parse_args()

    # Pass parsed arguments to the async function
    asyncio.run(main(args.model, args.temperature, args.data_path, args.verbose))
