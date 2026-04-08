# ====================================================
# tools.py — ERP Chatbot Toolset (Groq + PostgreSQL)
# Purpose: Define database tools used for function calling via Groq API
# ====================================================

import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
import re
import json
from groq import Groq
from prediction_engine import analyze_forecast_results

# ------------------------
# SECTION: Load Config
# ------------------------

load_dotenv()

# PostgreSQL connection string
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "erp_chatbot")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------------
# SECTION: Engine Connection (create single global engine for reuse)
# ------------------------

_ENGINE = None

def get_engine():
    """
    Return a module-level SQLAlchemy Engine, creating it once.
    Reusing the Engine avoids repeated connection initialization overhead.
    """
    global _ENGINE
    if _ENGINE is None:
        if not DB_URL:
            raise ValueError("Database connection string not available. Check your .env file.")
        _ENGINE = create_engine(
            DB_URL,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=False
        )
    return _ENGINE

# ------------------------
# SECTION: Schema Cache (In-memory only)
# ------------------------

schema_cache = {}

# ------------------------
# SECTION: SQL Query Validator
# ------------------------

def validate_sql_query(query: str) -> bool:
    """
    Ensures that only safe, read-only SELECT or WITH + SELECT queries are executed.
    Blocks any query with modification statements or dangerous functions.
    """
    print(f"[VALIDATION] Query: {query[:100]}...")
    query_original = query.strip()

    # Allow conversational responses
    if query_original.startswith("CONVERSATIONAL_RESPONSE:"):
        print("Conversational response detected — bypassing validation")
        return True

    # Normalize query (remove comments, extra spaces)
    query_clean = re.sub(r'--.*?\n', ' ', query_original)
    query_clean = re.sub(r'/\*.*?\*/', ' ', query_clean, flags=re.DOTALL)
    query_clean = ' '.join(query_clean.split()).lower()

    print("[VALIDATION] Cleaned Query:", query_clean)

    # Ensure query starts with SELECT or WITH
    if not (query_clean.startswith("select") or query_clean.startswith("with")):
        raise ValueError("Only SELECT statements and CTEs (WITH clauses) are allowed.")

    # Forbidden SQL commands (for safety)
    forbidden = [
        "insert", "update", "delete", "drop", "alter", "truncate",
        "merge", "exec", "execute", "create", "grant", "revoke", "backup", "restore"
    ]
    for word in forbidden:
        if re.search(rf"\b{word}\b", query_clean):
            raise ValueError(f"Dangerous SQL keyword detected: {word.upper()}")

    return True

# Tool 1: Get all usable tables
# ------------------------

def get_table_list() -> list:
    """Returns list of all base tables (schema-qualified) excluding log tables"""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                  AND table_schema NOT IN ('pg_catalog', 'information_schema')
                  AND LOWER(table_name) NOT LIKE '%_log%'
                  AND LOWER(table_name) NOT LIKE '%log_%'
                ORDER BY table_schema, table_name
            """))
            return [f"{row.table_schema}.{row.table_name}" for row in result.fetchall()]
    except Exception as e:
        print(f"Error getting table list: {str(e)}")
        return []

# ------------------------
# Tool 2: Describe a table (cached)
# ------------------------

def describe_table(table_name: str) -> dict:
    """Returns schema details (column name + type) for a given table. Cached for reuse."""
    if table_name in schema_cache:
        return schema_cache[table_name]
    
    engine = get_engine()
    try:
        inspector = inspect(engine)
        
        # Parse schema and table
        if '.' in table_name:
            schema, table = table_name.split('.', 1)
        else:
            schema, table = 'public', table_name
        
        columns = inspector.get_columns(table, schema=schema)
        schema_desc = {
            "table": table_name,
            "columns": [{"name": col["name"], "type": str(col["type"])} for col in columns]
        }
        
        schema_cache[table_name] = schema_desc  # Cache it
        return schema_desc
        
    except Exception as e:
        print(f"Error describing table {table_name}: {str(e)}")
        return {"table": table_name, "columns": [], "error": str(e)}

# ------------------------
# Tool 3: Generate SQL Query from Natural Language
# ------------------------

def generate_sql_query(user_question: str, schema: dict) -> str:
    """Uses Groq LLM to generate a SELECT SQL query from user question and table schema"""
    table_name = schema.get("table", "unknown")
    columns = schema.get("columns", [])
    column_details = "\n".join([f"- {col['name']} ({col['type']})" for col in columns])
    
    prompt = f"""
You are a SQL expert assistant. Given the following schema and a business question, write a safe SELECT query.

### Table:
{table_name}

### Columns:
{column_details}

### Rules:
- Only use SELECT queries.
- Do not use UPDATE, DELETE, INSERT, DROP, or any unsafe SQL.
- Avoid joins unless necessary.
- Be concise.
- Use proper PostgreSQL syntax.

### Question:
{user_question}

### SQL:
"""
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500
        )
        
        raw_query = response.choices[0].message.content.strip()
        
        # Clean up the query
        if raw_query.startswith("```"):
            raw_query = raw_query[3:]
        if raw_query.endswith("```"):
            raw_query = raw_query[:-3]
        
        # Debug log: print it always
        print(f"[LangChain Generated SQL] {raw_query}")
        
        # Try validation
        try:
            validate_sql_query(raw_query)
        except Exception as ve:
            return {"query": raw_query, "error": f"Unsafe query blocked: {str(ve)}"}
        
        return {"query": raw_query.strip()}
        
    except Exception as e:
        return {"error": f"-- Error generating query: {str(e)}"}

# Tool 4: Execute a validated SQL query
# ------------------------

def execute_sql_query(query: str) -> list:
    """Executes SELECT SQL query OR passes through conversational responses"""
    
    query = query.strip()
    
    try:
        validate_sql_query(query)
    except Exception as e:
        return [{"error": f"Query validation failed: {str(e)}"}]
    
    # Handle conversational responses
    if query.strip().startswith("CONVERSATIONAL_RESPONSE:"):
        conversational_text = query.replace("CONVERSATIONAL_RESPONSE:", "").strip()
        return [{"conversational_response": conversational_text}]
    
    # Rest of SQL execution logic...
    engine = get_engine()
    try:
        # Add LIMIT clause if not present (safety fallback)
        if "limit " not in query.lower():
            query = query.rstrip(";") + " LIMIT 500"
        
        # Enforce a timeout (20 seconds) to avoid hanging forever
        with engine.connect().execution_options(timeout=20) as conn:
            df = pd.read_sql(query, conn)
        
        # Convert to records and handle data types
        result = df.to_dict(orient="records")
        
        # Convert datetime objects to strings
        for row in result:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
                elif isinstance(value, pd.Timestamp):
                    row[key] = str(value)
                elif hasattr(value, 'isoformat'):  # datetime objects
                    row[key] = value.isoformat()
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        return [{"error": error_msg}]

# ------------------------
# Tool 5: Format answer to user
# ------------------------

def format_answer(query: str, result: list) -> str:
    """Formats SQL result into readable string (used as assistant final message)"""
    if not result:
        return "No data found for your query."
    
    if isinstance(result, list) and len(result) > 0 and "error" in result[0]:
        return f"Query failed with error: {result[0]['error']}"
    
    # Show all rows (truncate only if too long for safety)
    full_text = json.dumps(result, indent=2, default=str)
    
    if len(full_text) > 10000:  # safeguard against massive outputs
        full_text = full_text[:10000] + "... (truncated)"
    
    return f"Result:\n{full_text}"

# ------------------------
# Tool 6: Get Sample Rows
# ------------------------

def get_sample_rows(table_name: str, limit: int = 5) -> list:
    """Returns a sample of rows from a given table (default 5)"""
    engine = get_engine()
    try:
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        df = pd.read_sql(query, engine)
        
        # Convert to records and handle data types
        result = df.to_dict(orient="records")
        
        # Convert datetime objects to strings
        for row in result:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
                elif isinstance(value, pd.Timestamp):
                    row[key] = str(value)
                elif hasattr(value, 'isoformat'):  # datetime objects
                    row[key] = value.isoformat()
        
        return result
        
    except Exception as e:
        return [{"error": f"Failed to get sample rows: {str(e)}"}]

# ------------------------
# Tool 7: Get Column Names Only
# ------------------------

def get_column_names(table_name: str) -> list:
    """Returns just the list of column names for a given table"""
    engine = get_engine()
    try:
        inspector = inspect(engine)
        
        if '.' in table_name:
            schema, table = table_name.split('.', 1)
        else:
            schema, table = 'public', table_name
        
        columns = inspector.get_columns(table, schema=schema)
        return [col["name"] for col in columns]
        
    except Exception as e:
        print(f"Error getting column names for {table_name}: {str(e)}")
        return []

# ------------------------
# Tool 8: LangChain SQL Agent Wrapper
# ------------------------

def run_langchain_query_tool(user_question: str, context_messages: list = None) -> dict:
    """
    Wrapper tool for LangChain SQL Agent with context support.
    Includes safety check for missed forecast questions.
    """
    # Safety net: detect if this should be a forecast question
    if detect_forecast_question(user_question):
        print("[WARNING] Forecast question detected in wrong tool! Redirecting...")
        return run_prediction_tool(user_question, context_messages)
    
    try:
        from langagent import run_langchain_query
        return run_langchain_query(user_question, context_messages=context_messages)
    except Exception as e:
        return {"error": str(e)}

# forecasting related tool 
def detect_forecast_question(user_question: str) -> bool:
    """Detect if user is asking a forecasting/prediction question."""
    
    # STRICT forecast keywords - require strong intent indicators
    forecast_keywords = [
        "forecast", "predict", "prediction", "project", "projection",
        "expected", "will sell", "will make", "going to sell", "going to make",
        "projected", "future trend", "projected trend", "cash"
    ]
    
    # Temporal future indicators - but ONLY if combined with action verbs
    temporal_indicators = ["next year", "following year", "upcoming year", "2026"]
    action_verbs = ["sell", "make", "revenue", "profit", "forecast", "predict", "project", "estimate"]
    
    question_lower = user_question.lower()
    
    # Check for explicit forecast keywords
    is_forecast = any(keyword in question_lower for keyword in forecast_keywords)
    
    # OR check for temporal indicator + action verb combination
    if not is_forecast:
        has_temporal = any(temporal in question_lower for temporal in temporal_indicators)
        has_action = any(verb in question_lower for verb in action_verbs)
        
        # REQUIRE BOTH temporal AND action verb to avoid false positives
        if has_temporal and has_action:
            historical_count = sum(1 for h in ["this year", "last year", "current year"] if h in question_lower)
            future_count = sum(1 for t in temporal_indicators if t in question_lower)
            is_forecast = (future_count > historical_count)
    
    from datetime import datetime
    current_year = datetime.now().year
    
    # Check if question mentions future year (e.g., "2026")
    future_year_pattern = rf"\b(20[3-9][0-9]|2[1-9][0-9][0-9])\b"
    year_matches = re.findall(future_year_pattern, question_lower)
    
    for year_str in year_matches:
        try:
            year = int(year_str)
            if year > current_year:
                is_forecast = True
                break
        except ValueError:
            continue
    
    if is_forecast:
        print(f"[FORECAST DETECTION] Forecast question detected: {user_question[:100]}...")
    
    return is_forecast


def run_prediction_tool(user_question: str, context_messages: list = None) -> dict:
    """
    Wrapper tool for forecasting pipeline.
    
    Flow:
    1. Calls langagent.py to generate FORECAST SQL (with is_forecast=True flag)
    2. Executes the SQL
    3. Passes results to prediction_engine.py for Groq-powered forecasting
    4. Returns forecast response
    """
    print(f"\n{'='*60}")
    print(f"🎯 PREDICTION TOOL CALLED!")
    print(f"Question: {user_question}")
    print(f"{'='*60}\n")

    try:
        print(f"[PREDICTION TOOL] Processing forecast question: {user_question}")
        from langagent import run_langchain_query
        
        # Call langagent with is_forecast=True to trigger special SQL generation
        sql_response = run_langchain_query(
            user_question, 
            context_messages=context_messages,
            is_forecast=True
        )
        
        if "error" in sql_response:
            return sql_response
        
        sql_result = sql_response.get("result", [])
        sql_query = sql_response.get("query", "")
        print(f"[PREDICTION TOOL] SQL QUERY GENERATED:\n{sql_query}\n")

        
        if not sql_result:
            return {"error": "No historical data found for forecasting", "query": sql_query}
        
        print(f"[PREDICTION TOOL] SQL returned {len(sql_result)} rows")
        
        # Pass results to prediction_engine for Groq-powered forecasting
        forecast_response = analyze_forecast_results(user_question, sql_result)
        forecast_response["query"] = sql_query
        
        return forecast_response
        
    except Exception as e:
        print(f"[PREDICTION TOOL] Error: {str(e)}")
        return {"error": f"Prediction tool failed: {str(e)}"}
