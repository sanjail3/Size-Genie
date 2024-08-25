
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
import pandas as pd
from sqlalchemy import create_engine


class AIAgent:
    def __init__(self, model_name="gpt-4", temperature=0, db_url=None):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.output = JsonOutputParser()
        self.db_url = db_url
        self.sql_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 """You are an expert SQL analyst and logical reasoning. I will provide you with a dress size chart and a database schema, including column descriptions.
                    Your task is to analyze the dress size chart and schema, then generate a SQL query to retrieve the specified columns from the database that correspond to the size chart. 
                    If a specific column isn't available, return the most relevant column from the database instead.
 
                    <Notes>
                    - Rename the column similar to the size chart.
                    - Always include purchased size column and remove all other size-related data.
                    - Don't choose id data column.
                    - Don't choose columns not in the database schema.
                    - Always return a single SQL query.
                    - Only write `SELECT` queries; never write `UPDATE`, `INSERT`, or `DELETE` queries.
                    <format>
                    Use the following format:
                    ```json
                    {{
                      "query": "SELECT CustomerName, City FROM Customers;"
                    }}
                    ```"""
                 ),
                ("human", """
                size_chart : {size_chart}
                database_schema : {database_schema}""")
            ]
        )
        self.sql_chain = self.sql_prompt | self.llm | self.output | RunnableLambda(self.get_output)

    def get_output(self, query):
        if not isinstance(query, dict):
            raise Exception("Query must be a dictionary")
        query_str = query.get("query")
        if not isinstance(query_str, str):
            raise Exception("Query must be a string")
        try:
            engine = create_engine(self.db_url)
            df = pd.read_sql(query_str, engine)
            return df, query_str
        except Exception as e:
            print("Exception occurred:", query_str)
            return str(e)

    def generate_query(self, size_chart, database_schema):
        result = self.sql_chain.invoke({"size_chart": size_chart, "database_schema": database_schema})
        df = result[0].copy()
        return df
