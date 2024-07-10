import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
from google.cloud import bigquery
from google.api_core.exceptions import PermissionDenied
import json
import pandas as pd
import re

# Ensure db-dtypes is installed
try:
    import db_dtypes
except ImportError:
    st.error("Please install the 'db-dtypes' package to use this function.")
    st.stop()

# Initialize Vertex AI
def initialize_vertexai():
    vertexai.init(project="flash-aviary-426023-j0", location="us-central1")

# Fetch Database Schema
def fetch_database_schema():
    client = bigquery.Client()
    dataset_id = 'bigquery-public-data'
    table_id = 'sdoh_cdc_wonder_natality.county_natality'
    
    table_ref = client.get_table(f"{dataset_id}.{table_id}")
    schema_info = {table_id: {field.name: field.field_type for field in table_ref.schema}}
    
    st.write("Fetched Schema Info:", schema_info)  

# Clean SQL Query
def clean_sql_query(query):
    query = query.replace("```", "").strip()
    return query

# Validate and Convert Data Types in SQL Query
def validate_and_convert_sql_query(sql_query, schema_info):
    # Extract the list of columns in the query
    select_clause = re.search(r"SELECT (.*?) FROM", sql_query, re.IGNORECASE | re.DOTALL)
    if not select_clause:
        return sql_query

    column_list = select_clause.group(1).split(",")
    column_list = [col.strip() for col in column_list]

    # Identify if there are any aggregate functions in the query
    has_aggregate = any(
        func in sql_query.upper() for func in ["SUM(", "AVG(", "MAX(", "MIN(", "COUNT("]
    )

    if has_aggregate:
        # Extract table name from the query
        table_name_match = re.search(r"FROM `([^`]+)`", sql_query, re.IGNORECASE)
        if not table_name_match:
            return sql_query
        
        table_name = table_name_match.group(1)
        columns = schema_info.get(table_name, {})
        
        # Identify non-aggregated columns to be included in GROUP BY
        non_aggregated_columns = [
            col for col in column_list if not any(
                agg in col.upper() for agg in ["SUM(", "AVG(", "MAX(", "MIN(", "COUNT("]
            )
        ]

        # Add GROUP BY clause
        if non_aggregated_columns:
            group_by_clause = "GROUP BY " + ", ".join(non_aggregated_columns)
            sql_query += " " + group_by_clause

    # Convert DATE types and handle CAST for SUM on STRING columns
    for table, columns in schema_info.items():
        for column, dtype in columns.items():
            if dtype == "DATE":
                # Handle date comparisons
                sql_query = re.sub(
                    rf"({column}\s*[<>=!]+\s*)(\d{{4}}-\d{{2}}-\d{{2}})",
                    rf"\1PARSE_DATE('%Y-%m-%d', '\2')",
                    sql_query
                )
            elif "SUM" in sql_query and dtype == "STRING":
                # SUM is applied only to numeric columns
                sql_query = sql_query.replace(f"SUM({column})", f"SUM(CAST({column} AS NUMERIC))")
    
    st.write("Validated and Converted SQL Query:", sql_query)  # Debugging line
    return sql_query

# Generate SQL Query from User Prompt
def generate_sql_query(prompt, schema_info):
    try:
        table_name = list(schema_info.keys())[0]
        columns = schema_info[table_name]
        table_description = ", ".join([f"{field}: {dtype}" for field, dtype in columns.items()])

        system_instruction = f"""
        You are a BigQuery Expert. Generate SQL queries based on user prompts.
        Here is the table description for the table {table_name}: {table_description}.
        Your response should be in JSON format. Do NOT include the three backticks (```) at the beginning of the query. For example: 
        {{
            "query": "SELECT * FROM `bigquery-public-data.sdoh_cdc_wonder_natality.county_natality`",
            "description": "This query will return all the records from the table"
        }}"""

        model = GenerativeModel(
            "gemini-1.0-pro-002",
            system_instruction=system_instruction
        )

        st.write("User Prompt:", prompt)
        response = model.generate_content(
            [prompt],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False,
        )
        st.write("Raw model response:", response.text)

        response_text = response.text.replace("```json", "").replace("```", "").strip()
        response_json = json.loads(response_text)
        sql_query = response_json.get("query", "").strip()

        st.write("Generated SQL Query:", sql_query)  # Debugging line

        sql_query = clean_sql_query(sql_query)
        sql_query = validate_and_convert_sql_query(sql_query, schema_info)
        
        return sql_query

    except PermissionDenied as e:
        st.error(f"Permission denied: {e.message}")
        return ""
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON response: {e}")
        return ""
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return ""

# Execute SQL Query on BigQuery
def execute_sql_query(sql_query):
    if not sql_query:
        raise ValueError("The SQL query is empty.")
    
    st.write("Final SQL Query for Execution:", sql_query)  # Debugging line
    
    client = bigquery.Client()
    try:
        query_job = client.query(sql_query)
        results = query_job.result()
        return list(results)
    except Exception as e:
        st.error(f"An error occurred while executing the SQL query: {e}")
        return None

# Generate Final Response from Query Result
def generate_final_response(sql_query, query_result):
    if query_result is None:
        return "No results to generate a response from."

    # Limit the number of records to pass to the model
    max_records = 100  
    limited_result_data = query_result[:max_records]
    
    # Summarize the data if it exceeds the maximum records
    if len(query_result) > max_records:
        limited_result_data.append({"summary": f"Data truncated. Showing first {max_records} records out of {len(query_result)}."})

    result_data = [dict(row) for row in limited_result_data]

    concise_sql_query = sql_query[:1000]  # Limit the query length if needed

    model = GenerativeModel(
        "gemini-1.0-pro-002",
        system_instruction=f"Generate a response based on the SQL query and its results. SQL Query: {concise_sql_query}, Results: {result_data}"
    )
    response = model.generate_content(
        [f"SQL Query: {concise_sql_query}\nResults: {result_data}"],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    final_response = "".join([chunk.text for chunk in response if chunk.text])  # streaming response
    return final_response

# Streamlit Interface
def main():
    st.title("Natality by County Chatbot")
    user_prompt = st.text_input("Enter your prompt:")
    
    if st.button("Submit"):
        initialize_vertexai()

        with st.spinner('Fetching database schema...'):
            schema_info = fetch_database_schema()

        with st.spinner('Generating SQL query...'):
            sql_query = generate_sql_query(user_prompt, schema_info)
            st.write("Generated SQL Query:")
            st.code(sql_query)

        if sql_query:
            try:
                with st.spinner('Executing SQL query...'):
                    query_result = execute_sql_query(sql_query)
                    if query_result:
                        st.write("Query Result:")
                        st.write(query_result)
                        st.write("Database Result:")
                        st.write(pd.DataFrame(query_result))
            except Exception as e:
                st.error(f"An error occurred while executing the SQL query: {e}")
                return

            with st.spinner('Generating final response...'):
                final_response = generate_final_response(sql_query, query_result)
                st.write("Final Response:")
                st.write(final_response)

# Configuration for Vertex AI Generation
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

if __name__ == "__main__":
    main()