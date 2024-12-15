"""Main script for running financial analysis workflow."""

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


async def main():
    """
    Main async function to run financial analysis workflow.
    """
    # Load environment variables
    load_dotenv()

    # Data
    data_path = "data/train.json"

    # Create agents and graph
    generate, reflect, parser = FinancialAnalysisAgents.create_agents(
        model="gpt-4o-mini"
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
                print(f"Record ID: {data['id']}")
                print(f"Question: {question}")
                print(f"Expected Answer: {expected_answer}")
                print(f"Generated Answer: {parsed_content['answer']}")
                print("-" * 50)

        # Optional: Break after processing a few records for testing
        if idx == 5:
            break


if __name__ == "__main__":
    asyncio.run(main())
