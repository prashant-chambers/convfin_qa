"""Main script for running financial analysis workflow."""

import argparse
import asyncio

import mlflow
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from src.fin_qa import setup_logger
from src.fin_qa.agents import FinancialAnalysisAgents
from src.fin_qa.data_conversion import (
    convert_to_markdown_table,
    convert_to_paragraph,
    fix_invalid_json,
)
from src.fin_qa.data_loader import load_financial_data, load_prompt_template
from src.fin_qa.evaluate import exact_match, numerical_match_with_units
from src.fin_qa.graph import FinancialAnalysisGraph

logger = setup_logger(__file__)


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


async def main(model: str, temperature: float, data_path: str, n: int, verbose: bool):
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
            logger.info(f"Answering question #{idx + 1} of {n}")

            config = {
                "configurable": {"thread_id": f"{idx}"},
            }

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
                        logger.info(f"Record ID: {_id}")
                        logger.info(f"Question: {question}")
                        logger.info(f"Expected Answer: {ground_truth}")
                        logger.info(f"Generated Answer: {prediction}")
                        logger.info("-" * 50)
                    records.append(
                        {
                            "id": _id,
                            "question": question,
                            "ground_truth": ground_truth,
                            "prediction": prediction,
                        }
                    )

            # Break after processing n records
            if idx == n - 1:
                break

        logger.info("Running evaluations")

        # Dataframe with question, ground_truth, and prediction
        output_df = pd.DataFrame(records)

        # Lambda for applying exact match
        em = lambda row: exact_match(row["ground_truth"], row["prediction"])

        # Lambda for applying numerical match with units
        nm = lambda row: numerical_match_with_units(
            row["ground_truth"], row["prediction"]
        )

        # Add evaluation metrics to output dataframe
        output_cols = ["ground_truth", "prediction"]
        output_df["exact_match"] = output_df[output_cols].apply(em, axis=1)
        output_df["numerical_match_with_units"] = output_df[output_cols].apply(
            nm, axis=1
        )

        # Compute metrics
        exact_match_percentage = (output_df["exact_match"].mean()) * 100
        logger.info(f"Exact Match: {exact_match_percentage}%")

        numerical_match_percentage = (
            output_df["numerical_match_with_units"].mean()
        ) * 100
        logger.info(f"Numerical Match: {numerical_match_percentage}%")

        # Log metrics
        mlflow.log_metric("exact_match", round(exact_match_percentage, 2))
        mlflow.log_metric(
            "numerical_match_with_units", round(numerical_match_percentage, 2)
        )

        # Log data
        mlflow.log_table(output_df, "output.json")


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

    parser.add_argument(
        "--n",
        type=int,
        default=10,
        required=False,
        help="Number of records to be processed.",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")

    # Parse arguments
    args = parser.parse_args()

    # Pass parsed arguments to the async function
    asyncio.run(
        main(args.model, args.temperature, args.data_path, args.n, args.verbose)
    )
