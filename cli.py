"""Main script for running financial analysis workflow."""

import argparse
import asyncio

import mlflow
import pandas as pd
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

    mlflow.set_experiment("financial_qa")

    with mlflow.start_run() as _:
        mlflow.log_param("model", model)
        mlflow.log_param("temperature", temperature)
        mlflow.log_param("data_path", data_path)

        # Create agents and graph
        generate, reflect, parser = FinancialAnalysisAgents.create_agents(
            model=model, temperature=temperature
        )
        graph = FinancialAnalysisGraph.create_graph(generate, reflect)

        financial_analyst_message = (
            generate.get_prompts()[0].messages[0].prompt.template
        )
        critic_message = reflect.get_prompts()[0].messages[0].prompt.template

        mlflow.log_param("financial_analyst_message", financial_analyst_message)
        mlflow.log_param("critic_message", critic_message)

        records = []

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
            for question, ground_truth in question_answer:
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
                    _id = data["id"]
                    prediction = parsed_content["answer"]
                    if verbose:
                        print(f"Record ID: {_id}")
                        print(f"Question: {question}")
                        print(f"Expected Answer: {ground_truth}")
                        print(f"Generated Answer: {prediction}")
                        print("-" * 50)
                    records.append(
                        {
                            "id": _id,
                            "question": question,
                            "ground_truth": ground_truth,
                            "prediction": prediction,
                        }
                    )

            # Optional: Break after processing a few records for testing
            if idx == 2:
                break

        mlflow.log_table(pd.DataFrame(records), "output.json")


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
