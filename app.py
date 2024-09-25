import streamlit as st
import google.generativeai as genai
import mysql.connector
import psycopg2
import pandas as pd
import plotly.express as px
from langchain.sql_database import SQLDatabase
import urllib.parse
from sqlalchemy import create_engine

GOOGLE_API_KEY = 'APIKEY'

genai.configure(api_key=GOOGLE_API_KEY)

# Define message classes
class AIMessage:
    def __init__(self, content):
        self.content = content

class HumanMessage:
    def __init__(self, content):
        self.content = content

# Functions to initialize database connections
def init_database_mysql(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    password = urllib.parse.quote(password)
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def init_database_postgresql(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    password = urllib.parse.quote(password)
    db_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_mysql_table_schema(db_uri):
    engine = create_engine(db_uri)
    query_tables = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = DATABASE();"
    table_names = pd.read_sql(query_tables, engine)
    
    schema_info = ""
    for table_name in table_names['TABLE_NAME']:
        query_schema = f"""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT, COLUMN_KEY
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = DATABASE();
        """
        schema_df = pd.read_sql(query_schema, engine)
        schema_info += f"Schema for table {table_name}:\n"
        schema_info += schema_df.to_string(index=False)
        schema_info += "\n\n"
    
    return schema_info

# Function to get schema for all tables in PostgreSQL
def get_postgresql_table_schema(db_uri):
    engine = create_engine(db_uri)
    query_tables = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
    table_names = pd.read_sql(query_tables, engine)
    
    schema_info = ""
    for table_name in table_names['table_name']:
        query_schema = f"""
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = '{table_name}' AND table_schema = 'public';
        """
        schema_df = pd.read_sql(query_schema, engine)
        schema_info += f"Schema for table {table_name}:\n"
        schema_info += schema_df.to_string(index=False)
        schema_info += "\n\n"
    
    return schema_info

# Functions to connect to databases
def connect_mysql(user, password, host, port, database):
    conn = mysql.connector.connect(
        user=user,
        password=password,
        host=host,
        port=port,
        database=database
    )
    return conn

def connect_postgresql(user, password, host, port, database):
    conn = psycopg2.connect(
        user=user,
        password=password,
        host=host,
        port=port,
        dbname=database
    )
    return conn

# Streamlit app configuration
st.set_page_config(page_title="DataBot", page_icon=":speech_balloon:")

# Initialize schema selection
schema = 0
with st.sidebar:
    st.subheader("Hello !")
    st.write("Connect to the database and start Interacting.")

    option_to_value = {
        "MySQL": 1,
        "PostgreSQL": 2,
    }

    selected_option_label = st.sidebar.selectbox("Choose a database", list(option_to_value.keys()))
    selected_option_value = option_to_value[selected_option_label]
    schema = selected_option_value

    host = st.text_input("Host", value="localhost", key="Host")
    port = st.text_input("Port", value="3306" if selected_option_value == 1 else "5432", key="Port")
    user = st.text_input("User", value="root", key="User")
    password = st.text_input("Password", type="password", value="", key="Password")
    database = st.text_input("Database", value="chinook", key="Database")

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            if selected_option_value == 1:
                db = connect_mysql(user, password, host, port, database)
                st.session_state.db = db
                st.session_state.db_type = "mysql"
                st.success("Connected to MySQL database!")
            elif selected_option_value == 2:
                db = connect_postgresql(user, password, host, port, database)
                st.session_state.db = db
                st.session_state.db_type = "postgresql"
                st.success("Connected to PostgreSQL database!")

if 'db_uri' in st.session_state:
    db_uri = st.session_state.db_uri
    db_type = st.session_state.db_type

    if db_type == "mysql":
        schema_info = get_mysql_table_schema(db_uri)
    elif db_type == "postgresql":
        schema_info = get_postgresql_table_schema(db_uri)

# Prompt for the AI model
prompt = [
    """
    Imagine you're an SQL expert and data visualization advisor adept at translating English questions into precise SQL queries and recommending visualization types for the given database. Based on the table schema below, your expertise enables you to select the most appropriate chart type based on the expected query result set to effectively communicate the insights.

    <SCHEMA>{schema}</SCHEMA>
    Here are examples to guide your query generation and visualization recommendation:

    - - Example Question 1: "How many unique artists are there?"
      SQL Query: SELECT COUNT(DISTINCT ArtistId) FROM Artist;
      Recommended Chart: None (The result is a single numeric value.)

    - Example Question 2: "What are the total number of tracks in each genre?"
      SQL Query: SELECT g.Name AS Genre, COUNT(t.TrackId) AS TotalTracks FROM Track t JOIN Genre g ON t.GenreId = g.GenreId GROUP BY g.Name;
      Recommended Chart: Bar chart (Genres on the X-axis and total tracks on the Y-axis.)

    - Example Question 3: "List all customers who made purchases totaling more than $50."
      SQL Query: SELECT c.FirstName, c.LastName, SUM(i.Total) AS TotalSpent FROM Customer c JOIN Invoices i ON c.CustomerId = i.CustomerId GROUP BY c.CustomerId HAVING TotalSpent > 50;
      Recommended Chart: None (The result is a list of customers.)

    - Example Question 4: "Which tracks have the highest unit price?"
      SQL Query: SELECT Name, UnitPrice FROM Track ORDER BY UnitPrice DESC LIMIT 10;
      Recommended Chart: Bar chart (Track names on the X-axis and unit prices on the Y-axis.)

    Your task is to craft the correct SQL query in response to the given English questions and suggest an appropriate chart type for visualizing the query results, if applicable. Please ensure that the SQL code generated does not include triple backticks (\`\`\`) at the beginning or end and avoids including the word "sql" within the output. Also, provide clear and concise chart recommendations when the query results lend themselves to visualization.
    """
]

def get_gemini_response(question, prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt[0], question])
    return response.text

def read_sql_query(sql, config):
    conn = mysql.connector.connect(**config)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df

def get_sql_query_from_response(response):
    try:
        query_start = response.index('SELECT')
        query_end = response.index(';') + 1
        sql_query = response[query_start:query_end]
        return sql_query
    except ValueError:
        st.error("Could not extract SQL query from the response.")
        return None

def determine_chart_type(df):
    if len(df.columns) == 2:
        if df.dtypes[1] in ['int64', 'float64'] and len(df) > 1:
            return 'bar'
        elif df.dtypes[1] in ['int64', 'float64'] and len(df) <= 10:
            return 'pie'
    elif len(df.columns) >= 3 and df.dtypes[1] in ['int64', 'float64']:
        return 'line'
    return None

def generate_chart(df, chart_type):
    if chart_type == 'bar':
        fig = px.bar(df, x=df.columns[0], y=df.columns[1],
                     title=f"{df.columns[0]} vs. {df.columns[1]}",
                     template="plotly_white", color=df.columns[0])
    elif chart_type == 'pie':
        fig = px.pie(df, names=df.columns[0], values=df.columns[1],
                     title=f"Distribution of {df.columns[0]}",
                     template="plotly_white")
    elif chart_type == 'line':
        fig = px.line(df, x=df.columns[0], y=df.columns[1],
                      title=f"{df.columns[0]} vs. {df.columns[1]}",
                      template="plotly_white")
    else:
        st.error("Unsupported chart type.")
        return None
    return fig

# Main application logic
st.title("DataBot")
st.write("Ask a question in natural language, and I will generate the SQL query and visualize the results.")

question = st.text_input("Enter your question:")
if st.button("Generate SQL and Visualize"):
    with st.spinner("Generating SQL query and visualization..."):
        response = get_gemini_response(question, prompt)
        sql_query = get_sql_query_from_response(response)
        st.write("Generated SQL Query:")
        st.code(sql_query)

        if sql_query:
            if schema == 1:
                config = {
                    'user': user,
                    'password': password,
                    'host': host,
                    'port': port,
                    'database': database
                }
                df = read_sql_query(sql_query, config)
            elif schema == 2:
                config = {
                    'user': user,
                    'password': password,
                    'host': host,
                    'port': port,
                    'dbname': database
                }
                df = read_sql_query(sql_query, config)

            if not df.empty:
                st.write("Query Results:")
                st.write(df)
                chart_type = determine_chart_type(df)
                if chart_type:
                    st.write("Generated Visualization:")
                    fig = generate_chart(df, chart_type)
                    if fig:
                        st.plotly_chart(fig)
                else:
                    st.write("No suitable chart type found for the query results.")
            else:
                st.write("No data returned from the query.")
