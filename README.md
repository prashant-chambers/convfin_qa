# Multi-Agent Financial QA System

The objective of this project is to develop a Language Model (LLM) driven prototype capable of answering questions based on text and table context. The prototype will be evaluated using the ConvFinQA dataset, specifically the train.json file.

## Solution Overview

This system leverages a dual-agent approach to perform accurate financial calculations and analysis through iterative refinement.

It consists of two specialized agents working in tandem:

### Financial Analysis Agent

- Interprets complex financial reports
- Extracts relevant numerical data
- Performs precise calculations
- Provides detailed step-by-step analysis

### Critic Agent

- Validates data extraction accuracy
- Verifies calculation steps
- Ensures logical consistency
- Confirms proper formatting and rounding

## Methodology

The system employs an iterative approach where calculations are continuously refined through agent collaboration until reaching optimal accuracy. This implementation is based on latest research in numerical reasoning as detailed in this [paper](https://arxiv.org/abs/2410.21741).

![alt text](images/two_agent_setting.png "Two-agent Setting")

The solution was implemented using LangGraph, which provides a flexible framework for orchestrating agent interactions. MLflow was utilized to track experiment parameters, metrics, and model performance throughout the development process.

LLM: Azure OpenAI GPT 4o

```mermaid
%%{init: { 'theme': 'default', 'themeVariables': { 'fontFamily': 'JetBrains Mono' } } }%%
graph LR
    A["Input Question"] --> B["Financial Analysis Agent"]
    B --> C["Initial Analysis"]
    C --> D["Critic Agent"]
    D --> E{"Needs Refinement?"}
    E -- Yes --> B
    E -- No --> F["Final Answer"]
```

The graph illustrates the iterative refinement process where the Financial Analysis Agent's output is validated by the Critic Agent. If refinement is needed, the process loops back for further analysis until the desired accuracy is achieved.

> [!NOTE]
> The table data in 2D list was converted to markdown table before supplying as a context to Agent.

## Set Up

### Running the CLI App

1. Set up environment and install dependencies

```bash
python -m venv .venv
.venv/Scripts/Activate.ps1 # Windows
source .venv/bin/activate # Linux / MacOS
pip install --upgrade pip
pip install -r requirements-dev.txt
```

> [!NOTE]
> The application was developed and tested using Python 3.10, hence Python >= 3.10 is recommended.

2. Set environment variables

```bash
AZURE_OPENAI_ENDPOINT=****
AZURE_OPENAI_API_KEY=****
OPENAI_API_VERSION=****
```

> [!NOTE]
> The application uses Azure OpenAI services for LLM

3. Start MLflow server locally

```bash
mlflow server
```

4. Test cli app

```bash
python cli.py --model "gpt-4o" --temperature "0.0" --data-path "data/train.json" --n "100" --verbose

# cli options:
#   -h, --help                  show this help message and exit
#   --model MODEL               The name of the model to use.
#   --temperature TEMPERATURE   The temperature setting (must be between 0.0 and 1.0).
#   --data-path DATA_PATH       The path to the input data.
#   --n N                       Number of records to be processed.
#   --verbose                   Enable verbose mode.
```

5. Running the CLI app using Docker

Update the command parameters as required in `compose.yaml` and run the following command.

```bash
docker compose up --build
```

### Running tests

```
pytest -vv --junitxml=test-unit.xml --cov=. --cov-report=html
```

## Features

- CLI Application
- MLflow integration
- Multi-Agent Reflection
- LLM with retry
- Output parser with retry
- Containerized app
- Code quality checks
- Tests

## Evaluation Metrics

Two key metrics were chosen to evaluate the systems performance:

### 1. Exact Match (EM)

The Exact Match metric measures whether the systems output perfectly matches the ground truth answer, character for character.

**Pros**
- Standard metric in QA evaluation
- Easy to interpret - binary success/failure
- Ensures high precision requirements

**Cons**
- Very strict - small formatting differences cause failure
- Doesn't account for semantically correct but differently formatted answers
- May understate system performance

### 2. Numerical Match (NM)

A custom metric that extracts and compares numerical values between prediction and ground truth, allowing for small absolute differences (0.5 tolerance).

**Pros**
- More forgiving than EM for financial calculations
- Handles different number formats
- Accounts for acceptable rounding differences

**Cons**
- Does not take into account the units

Using both metrics provides a balanced evaluation approach - EM ensures strict accuracy where needed, while NM provides a more nuanced view of numerical reasoning capabilities.


## Results

> [!IMPORTANT]
> The results are based on the evaluation of first 100 records from `train.json`

The parameters, metrics, and output are all logged in MLFlow.

![alt text](images/mlflow_metrics.png "MLflow")

![alt text](images/mlflow_artifact.png "MLflow output")

### Numerical Match

Since, we do not have units in the output generated by the model, we use numerical match metric to assess the accuracy of the model.

Numerical Match: 75.81%

> [!NOTE]
> The results are in `output.csv`

### Latency Analysis

The systems response time was thoroughly analyzed using various percentile metrics to understand performance characteristics:

- Mean latency: 20.18 seconds for the complete two-agent interaction
- Latency distribution:
    - P25: 11.14 seconds
    - P50 (median): 17.62 seconds
    - P75: 27.3 seconds
    - P95: 46.11 seconds
    - P99: 58.93 seconds
    - Range metrics:
    - Minimum latency: 1.53 seconds
    - Maximum latency: 65.49 seconds

These metrics indicate that while most queries are processed within a reasonable timeframe (median ~13 seconds), there can be occasional longer processing times for more complex calculations requiring multiple agent iterations.

The latency distribution shows a right-skewed pattern, with 75% of queries completing within 18.4 seconds, but some edge cases requiring up to 54.43 seconds for thorough analysis and verification.

## Findings

> [!IMPORTANT]
> The findings are based on the evaluation of first 100 records from `train.json`

### Sign Interpretation Errors

Incorrect handling of negative percentages or changes
Example: "what was the percentage change in benefits obligations from 2018 to 2019?"

Actual: 19.4%  
Predicted: -19.35%

Shows difficulty in correctly interpreting directional changes.

### Precision and Rounding Issues

Some predictions are close but not exactly matching the actual value
Example: "what portion of the authorized shares of class b common stock is outstanding?"

Actual: 56.6%  
Predicted: 0.565588

The question asks for portion, but the ground truth has percentage value.  
This highlights how correct numerical calculations may be penalized due to format mismatches between decimal and percentage representations.

Example: "if the 2003 growth rate is the same as 2002 , what would 2003 gas transmission throughput be in bcf?"

Actual: 645  
Predicted: 645.75

According to the tolerance setting, this is considered a mismatch. This highlights the precision issue between ground truth and prediction values—using standardized values in ground truth data would help provide clearer instructions to the model.

### Contextual Misinterpretation

Some errors suggest misunderstanding of the specific financial context
Example: "did 2015 adjusted ebitda increase more than 2015 actual ebitda?"

Predicted: 241.0 (which doesn't directly answer the yes/no question)

This is because the model is explicitly instructed to provide numerical answers based on the calculation steps.

### Content filter

1 out of 100 questions was filtered due to sensitive content in the question / context. Depending on the policy, an appropriate adjustment to the content filter may be required.

```
An unexpected BadRequestError occurred for request Single_ZBH/2002/page_46.pdf-3: Error code: 400 - {'error': {'message': "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766", 'type': None, 'param': 'prompt', 'code': 'content_filter', 'status': 400}}
```

## Shortcomings

### Latency

Being a two-agent system with reflection, there is some additional processing involved before arriving at the final answer. This increases the latency of the application, especially when the question is complex and requires deeper understanding of the context and multiple iterations.

### Accuracy

Being a financial QA application, it needs to have high accuracy. However, sometimes due to the ambiguous nature of questions and limitations in the prompt, it can lead to incorrect answers which could be costly. Instead of answering questions using provided free-form context, better accuracy could be achieved if the financial data were stored in a database, with an API provided as a tool to a ReAct agent.

### Evaluation Metrics

While the numerical match is a custom metric used to evaluate accuracy, additional metrics could be implemented to better assess the systems performance. The current evaluation approach might not capture all aspects of the systems accuracy and reliability in handling complex financial queries.

## Agent Interaction

**User** 
```
Read the following texts and table with financial data from earnings report carefully. Present your answer in the following JSON format.
{{
    "steps": ["show your calculation steps"], 
    "answer": "final numerical answer"
}}


Below is the context along with the Question.

during the years ended december 31 , 2013 , 2012 , and 2011 , we recognized approximately $ 6.5 million , $ 5.1 million and $ 4.7 million of compensation expense , respectively , for these options . as of december 31 , 2013 , there was approximately $ 20.3 million of total unrecognized compensation cost related to unvested stock options , which is expected to be recognized over a weighted average period of three years . stock-based compensation effective january 1 , 1999 , we implemented a deferred compensation plan , or the deferred plan , covering certain of our employees , including our executives . the shares issued under the deferred plan were granted to certain employees , including our executives and vesting will occur annually upon the completion of a service period or our meeting established financial performance criteria . annual vesting occurs at rates ranging from 15% ( 15 % ) to 35% ( 35 % ) once performance criteria are reached . a summary of our restricted stock as of december 31 , 2013 , 2012 and 2011 and charges during the years then ended are presented below: .

|                                                                         | 2013           | 2012               | 2011           |
|-----------------------------------------------------------------------|--------------|------------------|--------------|
| balance at beginning of year                                            | 2804901        | 2912456            | 2728290        |
| granted                                                                 | 192563         | 92729              | 185333         |
| cancelled                                                               | -3267 ( 3267 ) | -200284 ( 200284 ) | -1167 ( 1167 ) |
| balance at end of year                                                  | 2994197        | 2804901            | 2912456        |
| vested during the year                                                  | 21074          | 408800             | 66299          |
| compensation expense recorded                                           | $ 6713155      | $ 6930381          | $ 17365401     |
| weighted average fair value of restricted stock granted during the year | $ 17386949     | $ 7023942          | $ 21768084     |

weighted average fair value of restricted stock granted during the year $ 17386949 $ 7023942 $ 21768084 the fair value of restricted stock that vested during the years ended december 31 , 2013 , 2012 and 2011 was $ 1.6 million , $ 22.4 million and $ 4.3 million , respectively . as of december 31 , 2013 , there was $ 17.8 million of total unrecognized compensation cost related to unvested restricted stock , which is expected to be recognized over a weighted average period of approximately 2.7 years . for the years ended december 31 , 2013 , 2012 and 2011 , approximately $ 4.5 million , $ 4.1 million and $ 3.4 million , respectively , was capitalized to assets associated with compensation expense related to our long-term compensation plans , restricted stock and stock options . we granted ltip units , which include bonus , time-based and performance based awards , with a fair value of $ 27.1 million , zero and $ 8.5 million as of 2013 , 2012 and 2011 , respectively . the grant date fair value of the ltip unit awards was calculated in accordance with asc 718 . a third party consultant determined the fair value of the ltip units to have a discount from sl green&#39;s common stock price . the discount was calculated by considering the inherent uncertainty that the ltip units will reach parity with other common partnership units and the illiquidity due to transfer restrictions . as of december 31 , 2013 , there was $ 5.0 million of total unrecognized compensation expense related to the time-based and performance based awards , which is expected to be recognized over a weighted average period of approximately 1.5 years . during the years ended december 31 , 2013 , 2012 and 2011 , we recorded compensation expense related to bonus , time-based and performance based awards of approximately $ 27.3 million , $ 12.6 million and $ 8.5 million , respectively . 2010 notional unit long-term compensation plan in december 2009 , the compensation committee of the company&#39;s board of directors approved the general terms of the sl green realty corp . 2010 notional unit long-term compensation program , or the 2010 long-term compensation plan . the 2010 long-term compensation plan is a long-term incentive compensation plan pursuant to which award recipients could earn , in the aggregate , from approximately $ 15.0 million up to approximately $ 75.0 million of ltip units in the operating partnership based on our stock price appreciation over three years beginning on december 1 , 2009 ; provided that , if maximum performance had been achieved , approximately $ 25.0 million of awards could be earned at any time after the beginning of the second year and an additional approximately $ 25.0 million of awards could be earned at any time after the beginning of the third year . in order to achieve maximum performance under the 2010 long-term compensation plan , our aggregate stock price appreciation during the performance period had to equal or exceed 50% ( 50 % ) . the compensation committee determined that maximum performance had been achieved at or shortly after the beginning of each of the second and third years of the performance period and for the full performance period and , accordingly , 366815 ltip units , 385583 ltip units and 327416 ltip units were earned under the 2010 long-term compensation plan in december 2010 , 2011 and 2012 , respectively . substantially in accordance with the original terms of the program , 50% ( 50 % ) of these ltip units vested on december 17 , 2012 ( accelerated from the original january 1 , 2013 vesting date ) , 25% ( 25 % ) of these ltip units vested on december 11 , 2013 ( accelerated from the original january 1 , 2014 vesting date ) and the remainder is scheduled to vest on january 1 , 2015 based on .

Question: for the years ended december 31 , 2013 , 2012 and 2011 , what was the total in millions capitalized to assets associated with compensation expense related to long-term compensation plans , restricted stock and stock options?\\n
```

**Assistant**

```
{
    "steps": [
        "Identify the capitalized amounts for each year from the provided text.",
        "For 2013, the capitalized amount is $4.5 million.",
        "For 2012, the capitalized amount is $4.1 million.",
        "For 2011, the capitalized amount is $3.4 million.",
        "Sum the capitalized amounts for all three years: 4.5 + 4.1 + 3.4."
    ],
    "answer": "12.00 million"
}
```

**User**

```
The analysis provided in the response has several aspects that need to be critiqued:
Understanding of the Question: The question asks for the total capitalized amounts associated with compensation expense related to long-term compensation plans, restricted stock, and stock options for the years ended December 31, 2013, 2012, and 2011. The response correctly identifies this requirement.
Extraction of Numbers: The response accurately extracts the capitalized amounts from the context:
2013: $4.5 million
2012: $4.1 million
2011: $3.4 million These figures are correctly taken from the provided text.
Calculations: The calculation steps are clearly laid out:
The response sums the amounts: 4.5 + 4.1 + 3.4.
The sum is calculated as follows:
4.5 + 4.1 = 8.6
8.6 + 3.4 = 12.0 The calculation is correct.
Final Answer: The final answer is presented as "12.00 million." This is correctly rounded to two decimal places, and the unit "million" is preserved.
Logic of Steps: The logic used in the steps is sound. The response clearly outlines how the total was derived from the individual years' capitalized amounts.
Overall, the response is accurate, and all calculations and logic are correct. There are no issues detected.
ALL_OK
```

**Assistant**

```
{
    "steps": [
        "Identify the capitalized amounts for each year from the provided text.",
        "For 2013, the capitalized amount is $4.5 million.",
        "For 2012, the capitalized amount is $4.1 million.",
        "For 2011, the capitalized amount is $3.4 million.",
        "Sum the capitalized amounts for all three years: 4.5 + 4.1 + 3.4."
    ],
    "answer": "12.00 million"
}
```