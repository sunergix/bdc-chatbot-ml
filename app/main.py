import os
from typing import Dict
import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType

load_dotenv()
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# df = pd.read_csv("investments_output.csv")
df = pd.read_csv("investments_output_new.csv")
# df = pd.read_csv("SOI - (Goldman Sachs BDC, Inc.- 30 Jun, 2025) - 11-08-2025.csv")

# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

data_columns_description = """
**Investee Name**
    Borrower Name, a Portfolio Company in which the BDC has invested (A Normalized Name)

**Raw Name**
    Borrower Name, a Portfolio Company in which the BDC has invested (This is Raw information directly coming from the filing)

**Industry**
    Sector, Borrower or Portfolio Company in which it belongs to. (Normalized Industry)

**Raw Industry**
    Sector, Borrower or Portfolio Company in which it belongs to. (Raw information directly coming from the filing)

**Par ($M) / Shares (M)**
    Asset, Holdings
    The values coming from the filing and converted into Millions

**Cost ($M)**
    Asset, Holdings
    The values coming from the filing and converted into Millions

**Fair Value ($M)**
    Asset, Holdings
    The values coming from the filing and converted into Millions

**FV%  of Cost**
    This is a ratio to identify how good performing the investment is:
    For example: 
        if the value is less than 80% then it is not a good performing
        If the value is greater than or equal to 100%, then it is a good performing

**FV / Par**
    This is a ratio to identify how good performing the investment is:
    For example: 
        If the value is less than 80% then it is not a good Asset / Holding
        If the value is greater than or equal to 100%, then it is a good Asset / Holding

**Avg. Bid**
    It is the secondary Pricing of the investment. The Secondary Price means that if the Investment is traded, then How much percentage against its fair value will be traded
    For example:
        If the value is 99 and the investment Fair value is $100 million then it indicates that it would be bought at $99 million.

**Avg. Ask**
    It is the secondary Pricing of the investment. The Secondary Price means that if the Investment is traded, then How much percentage against its fair value will be traded.
    For example:
        If the value is 99 and the investment Fair value is $100 million, then it indicates that it would be sold at $99 million.

**% of Net Assets**
    This is a simple calculation of the Investment; the fair value is divided by the BDC’s Net Asset (which is available in the Financials). It is more kind of a proportion of the Net Assets by its Fair Value.

**% of Total Assets**
    This is a simple calculation of the Investment, the fair value is divided by the BDC’s Total Assets (which is available in the Financials). It is more kind of a proportion of the Total Assets by its Fair Value.

**Maturity**
    It represents when the Investment is going to mature.

**Acquisition Date**
    It represents when the investment was acquired by the BDC.

**Security Class**
    It is Raw information coming from the BDC Filings. It is a type of Investment.

**Security Class Normalized**
    The Raw information is grouped and classified into different types, like Term Loan, Delayed Draw Term Loan, Revolver, and Revolver/DDTL.

**Security Type**
    It represents the Seniority of the Investment.

**Security Type Raw**
    It is Raw information coming from the BDC filings. It is a more descriptive version of the Seniority Type.

**Structured Product Type**
    It is a type of Borrower investment. (CLO, Loan or Other)

**Non Qualifying**
    The “Y” flag means that it is a Non-Qualified Asset, which means the Borrower is located outside the United States.
    The “N” flag means that the borrower is located inside the United States.

**Unitranche Loans**
    The “Y” flag means that the loan only has a single Tranche loan.
    The “N” flag means that it is not a single Tranche loan.

**Non-Accrual Loans**
    The “Y” flag means that the investment is in Distress. It means that the borrower has not paid the Interest payment in the last 90 days.

**% of Portfolio (FV)**
    This is a simple calculation of the Investment; the fair value is divided by the BDC’s Total Fair Value. It is more of a proportion of the Total Fair Value by the Investment’s Fair Value.

**% of Portfolio (Cost)**
    This is a simple calculation of the Investment; the cost is divided by the BDC’s Total Cost. It is more of a proportion of the Total Cost by the Investment’s Cost.

**Interest Rate**
    It is Raw information. It provides information about the Interest Rate.

**Cash Rate/Spread**
    Spread / Cash Rate is basically the borrower paying above the Benchmark rate.

**PIK Rate**
    It provides information about the borrower paying in addition to the principal amount of the Investment.

**Base Rate Floor**
    It represents the minimum rate of the Benchmark Rate

**Ceiling Rate**
    It represents the maximum Interest Rate that the borrower will be paying for the Investment.

**ETP Rate**
    It represents that the applicable Early Exit rate applies to the investment for the Borrower. If a borrower wants to avail an early Exit before the Maturity of the Investment.

**Interest Rate Floor**
    It represents the minimum total interest rate that the borrower will be paying for the Investment.

**Rate Type**
    It represents the Benchmark Type (SOFR, SONIA, etc.) that will be applied to the Investment.

**Benchmark Rate**
    It represents the actual rate of the Benchmark type.

**All-In-Yield**
    The total Interest Rate a borrower will be paying for that Investment.

**All-In-Yield Calculated**
    The Y flag means the system's calculated value in the All-in-Yield column. The N flag means that the BDC reported the value in the filing.

**LIN**
    It is an Identifier that is registered in the LSEG products that represent that particular investment.

**PE Sponsors**
    Private Equity Sponsors
    This represents that this company has sponsored the Investment.

**Investee ID**
    It is an Identifier that is registered in our System for the Borrower.

**Security ID**
    It is an Identifier that is registered in our System for that Investment.

**Type**
    It represents the Type of Investment.

"""


PREFIX = """
You are a data analysis assistant working in Python with a Pandas DataFrame named `df`.

Your job:
- Try to break the user query into multiple steps and combine them after executing each to help better answer the query.
- ALWAYS base answers on the DataFrame content.
- Use Python and Pandas to compute answers, do not guess or invent values.
- If calculations are needed, run them before answering.
- Answer in a clear, concise, and structured way.

Column descriptions for reference (use when relevant, but do not repeat them unless necessary):

{data_columns_description}

When responding:
- Include only the necessary code and results to answer the query.
- Do NOT engage in casual conversation.
- Do NOT summarize the dataset unless asked.
- Focus entirely on accurate data analysis.
"""



SUFFIX = """
Provide the final answer in a clear and structured format.
"""


agent = create_pandas_dataframe_agent(
    llm,
    df,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    prefix=PREFIX.format(data_columns_description=data_columns_description),
    suffix=SUFFIX,
    # include_df_in_prompt=True,  # Include the DataFrame head in the prompt
    # number_of_head_rows=5,      # Number of rows to include
    allow_dangerous_code=True,
    verbose=True
)

# Define the FastAPI app
app = FastAPI()

# Request model
class QueryRequest(BaseModel):
    question: str
    bdc: str
    quarter: str

@app.post("/api/v1/query")
async def query_agent(request: QueryRequest) -> Dict:
    try:
        response = agent.invoke({"input": request.question})
        return {"response": response["output"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)