# ====================================================
# app.py — Streamlit Frontend for ERP QnA Chatbot (Groq + PostgreSQL)
# ====================================================

import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from tools import run_langchain_query_tool, format_answer, detect_forecast_question
import time
import matplotlib.pyplot as plt
import matplotlib
import base64
import io
from groq import Groq

# Lazy load heavy modules only when needed
run_prediction_tool = None


current_year = datetime.now().year


# ------------------------
# Chart Generation Function
# ------------------------

def generate_chart_config(user_question: str, result_data: list) -> dict:
    """
    Uses Groq LLM to analyze user question and data to suggest optimal chart configuration
    """
    if not result_data or not isinstance(result_data, list) or len(result_data) == 0:
        return {"error": "No data available for chart generation"}
    
    # Get first few rows for analysis
    sample_data = result_data[:5] if len(result_data) > 5 else result_data
    
    # Extract column names and types
    columns_info = {}
    if sample_data:
        for key, value in sample_data[0].items():
            if isinstance(value, (int, float)):
                columns_info[key] = "numeric"
            elif isinstance(value, str):
                columns_info[key] = "text"
            else:
                columns_info[key] = "other"
    
    prompt = f"""You are a data visualization expert. Analyze this user question and data to suggest the best chart configuration.

User Question: "{user_question}"

Available Columns:
{json.dumps(columns_info, indent=2)}

Sample Data (first few rows):
{json.dumps(sample_data, indent=2, default=str)}

Rules:
- ONLY return valid JSON, no explanations
- If no numeric columns exist, return: {{"error": "Cannot generate chart - no numeric data found"}}
- Choose chart_type from: "bar", "line", "pie"
- For pie charts, use categorical column with numeric values
- Create meaningful titles based on the user's question
- Prefer "bar" for comparisons, "line" for trends, "pie" for proportions

Required JSON format:
{{
  "x_axis": "column_name",
  "y_axis": "numeric_column_name", 
  "chart_type": "bar|line|pie",
  "title": "Descriptive chart title"
}}

Generate chart config:"""

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        
        raw_response = response.choices[0].message.content.strip()
        
        # Clean up response 
        if raw_response.startswith("```"):
            raw_response = raw_response[3:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]
        if raw_response.startswith("json"):
            raw_response = raw_response[4:]
        
        chart_config = json.loads(raw_response.strip())
        
        print(f"[DEBUG] Generated chart config: {chart_config}")
        return chart_config
        
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON response from LLM: {str(e)}"}
    except Exception as e:
        return {"error": f"Chart generation failed: {str(e)}"}

def generate_chart_image(df, chart_config):
    """Generate chart as base64 image string with enhanced readability"""
    try:
        matplotlib.use('Agg')
        fig, ax = plt.subplots(figsize=(7, 5))
        
        x_col = chart_config.get("x_axis")
        y_col = chart_config.get("y_axis")
        chart_type = chart_config.get("chart_type", "bar")
        title = chart_config.get("title", "Chart")
        
        if chart_type == "bar":
            bars = ax.bar(df[x_col], df[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:,.0f}' if height >= 1 else f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
                       
        elif chart_type == "line":
            line = ax.plot(df[x_col], df[y_col], marker='o', linewidth=2, markersize=8)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            
            for i, (x, y) in enumerate(zip(df[x_col], df[y_col])):
                ax.annotate(f'{y:,.0f}' if y >= 1 else f'{y:.2f}', 
                           (x, y), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
                           
        elif chart_type == "pie":
            df_grouped = df.groupby(x_col)[y_col].sum()
            wedges, texts, autotexts = ax.pie(df_grouped, labels=df_grouped.index, 
                                            autopct=lambda pct: f'{pct:.1f}%\n({df_grouped.sum()*pct/100:,.0f})',
                                            startangle=90)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_fontweight('bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        if chart_type in ["bar", "line"]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}' if x >= 1 else f'{x:.2f}'))
            
        if chart_type in ["bar"] and len(str(df[x_col].iloc[0])) > 8:
            plt.xticks(rotation=45, ha='right')
            
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"[ERROR] Chart generation failed: {str(e)}")
        return None

def df_to_html_table(df, max_rows=100):
    """Convert DataFrame to HTML table string"""
    if len(df) > max_rows:
        df_display = df.head(max_rows)
        table_html = df_display.to_html(index=False, classes='dataframe-table', escape=False)
        table_html += f"<p><i>Showing first {max_rows} rows of {len(df)} total rows</i></p>"
    else:
        table_html = df.to_html(index=False, classes='dataframe-table', escape=False)
    
    return table_html

# ------------------------
# Load env
# ------------------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="ERP Data Assistant", layout="wide")
st.markdown("""
<style>

/*  Send button styling */
[data-testid="stFormSubmitButton"] button {
    background-color: #007bff !important;
    color: white !important;
    font-size: 18px !important;
    border: none !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
}

/* Chat bubbles */
.user-bubble {
    background-color: #007bff;
    color: white;
    padding: 10px 15px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 75%;
    display: inline-block;
    word-wrap: break-word;
}
            
/* Data visualization buttons container */
.data-buttons-container {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 5px 10px;
    margin: 0;
    text-align: center;
}

.data-buttons-container h4 {
    margin: 0 0 10px 0;
    font-size: 14px;
    color: #6c757d;
}

.assistant-bubble {
    background-color: #f1f3f4;
    color: #333;
    padding: 10px 15px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 75%;
    display: inline-block;
    word-wrap: break-word;
}

/* Visualization bubble for charts and tables */
.visualization-bubble {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    padding: 15px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 90%;
    display: inline-block;
    word-wrap: break-word;
}

.visualization-bubble img {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
}

.dataframe-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}

.dataframe-table th, .dataframe-table td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}

.dataframe-table th {
    background-color: #f2f2f2;
    font-weight: bold;
}

.visualization-message {
    display: flex;
    justify-content: flex-start;
    margin: 5px 0;
}

/* Compact input container */
.compact-input-container {
    background-color: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}


/* Scrollable chat container */
.chat-history-container {
    height: 800px;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 8px;
}
                        
.user-message {
    display: flex;
    justify-content: flex-end;
    margin: 5px 0;
}

.assistant-message {
    display: flex;
    justify-content: flex-start;
    margin: 5px 0;
}

/* Hide sidebar by default */
div[data-testid="stSidebar"] {
    display: none !important;
}

/* Force toggle button to show */
div[data-testid="stSidebarNav"] {
    display: block !important;
    visibility: visible !important;
}

            
/* Chat bubble spinner */
.chat-spinner {
    width: 22px;
    height: 22px;
    border: 3px solid #ddd;
    border-top: 3px solid #007bff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    display: inline-block;
    margin: 6px 8px;
    vertical-align: middle;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

</style>

<script>
// Auto-scroll to bottom of chat
function scrollToBottom() {
    const chatContainer = parent.document.querySelector('.chat-history-container');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

window.addEventListener('load', function() {
    setTimeout(scrollToBottom, 500);
});
</script>
""", unsafe_allow_html=True)



# ------------------------
# Context Management (Session-Based)
# ------------------------

def get_recent_context_messages(limit=6):
    """Get recent messages from session state for context"""
    recent_messages = []
    chat_history = st.session_state.get("chat_history", [])
    
    # Get last N messages that are user/assistant type
    msg_count = 0
    for item in reversed(chat_history):
        if len(item) == 3:
            role, content, timestamp = item
            if role in ("user", "assistant") and msg_count < limit:
                recent_messages.insert(0, {"role": role, "content": content})
                msg_count += 1
    
    return recent_messages

# ------------------------
# Sidebar for Settings & Controls
# ------------------------

with st.sidebar:
    st.header("Settings & Controls")

    # Schema Management
    st.subheader("Schema Management")
    if st.button("Refresh Schema Cache"):
        try:
            from langagent import build_schema_cache
            schema_map = build_schema_cache()
            st.success(f"Schema refreshed! Found {len(schema_map)} tables.")
        except Exception as e:
            st.error(f"Schema refresh failed: {str(e)}")

    # Conversation Management
    st.subheader("Conversation Management")
    if st.button("New Conversation"):
        st.session_state.chat_history = []
        st.session_state.last_result_df = None
        st.session_state.show_data_buttons = False
        st.session_state.data_inserted = False
        st.rerun()

    # Export Options
    st.subheader("Export Chat")
    if st.session_state.get("chat_history"):
        chat_export = json.dumps(st.session_state.chat_history, indent=2)
        st.download_button(
            "Download Chat History",
            chat_export,
            f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )

# ------------------------
# Main Title
# ------------------------

st.title("ERP Data Assistant")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_result_df" not in st.session_state:
    st.session_state.last_result_df = None

if "last_chart_config" not in st.session_state:
    st.session_state.last_chart_config = None

if "show_data_buttons" not in st.session_state:
    st.session_state.show_data_buttons = False

if "data_inserted" not in st.session_state:
    st.session_state.data_inserted = False
    
# ------------------------
# Input Section at Top
# ------------------------
st.markdown('<div class="compact-input-container">', unsafe_allow_html=True)

# Chat input form
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([9, 1])
    with col1:
        user_question = st.text_input(
            "Ask your ERP question:",
            placeholder="Ask me anything about your ERP data...",
            label_visibility="collapsed",
            key="user_input"
        )
    with col2:
        submitted = st.form_submit_button("➤")

# Data visualization buttons
st.markdown('<div class="data-buttons-container">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

if st.session_state.last_result_df is None:
    st.caption("⚠️ No data to visualize yet. Ask a question first.")

if col1.button("📋 Show data in table format", key="show_table_btn"):
    if st.session_state.last_result_df is not None:
        table_html = df_to_html_table(st.session_state.last_result_df)
        st.session_state.chat_history.append(("table", table_html))
        st.rerun()
    else:
        st.warning("No data available to show as table.")

if col2.button("📊 Show data in chart format ", key="show_chart_btn"):
    if st.session_state.last_result_df is not None:
        last_question = st.session_state.get('last_user_question', 'Show chart')
        with st.spinner("Generating chart..."):
            chart_config = generate_chart_config(last_question, st.session_state.last_result_df.to_dict('records'))
            if "error" not in chart_config:
                chart_image_b64 = generate_chart_image(st.session_state.last_result_df, chart_config)
                if chart_image_b64:
                    st.session_state.chat_history.append(("chart", chart_image_b64))
                    st.rerun()
                else:
                    st.error("Failed to generate chart")
            else:
                st.warning(f"Cannot generate chart: {chart_config['error']}")
    else:
        st.warning("No data available to show as chart.")

st.markdown('</div>', unsafe_allow_html=True)


# ------------------------
# Helper: Render Chat History 
# ------------------------

def render_chat_history():
    """Render the complete chat history with embedded visualizations"""
    html_parts = []
    html_parts.append('<div class="chat-history-container" id="chat-container">')
    
    for item in reversed(st.session_state.chat_history):
        if len(item) == 3:
            role, content, timestamp = item
            if role == "user":
                html_parts.append(f'<div class="user-message"><div class="user-bubble">{content}</div></div>')
            elif role == "assistant":
                html_parts.append(f'<div class="assistant-message"><div class="assistant-bubble">{content}</div></div>')
        elif len(item) == 2:
            viz_type, viz_data = item
            if viz_type == "table":
                html_parts.append(f'<div class="visualization-message"><div class="visualization-bubble">{viz_data}</div></div>')
            elif viz_type == "chart" and viz_data:
                html_parts.append(f'<div class="visualization-message"><div class="visualization-bubble"><img src="data:image/png;base64,{viz_data}" alt="Chart"></div></div>')
    
    html_parts.append('</div>')
    html_parts.append('<script>setTimeout(() => { const container = document.getElementById("chat-container"); if(container) container.scrollTop = container.scrollHeight; }, 100);</script>')
    html = "\n".join(html_parts)
    return html

# ------------------------
# Chat UI
# ------------------------

st.subheader("Chat History")
chat_container = st.empty()
chat_container.markdown(render_chat_history(), unsafe_allow_html=True)



if submitted and user_question:

    try:
        # Add user message and "Thinking..." bubble immediately
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_history.append(("user", user_question, timestamp))
        st.session_state.chat_history.append((
            "assistant",
            '<div style="display:flex;align-items:center;gap:8px;">'
            '<div class="chat-spinner"></div>'
            '<span style="color:#555;">Thinking...</span>'
            '</div>',
            timestamp
        ))
        chat_container.markdown(render_chat_history(), unsafe_allow_html=True)
        
        with st.spinner("Thinking..."):
            # ===== Start timing: overall
            overall_start = time.perf_counter()

            # ---------------------------
            # Get recent messages from session for context
            # ---------------------------
            t_memory_start = time.perf_counter()
            recent_messages = get_recent_context_messages(limit=6)
            t_memory_end = time.perf_counter()

            # ---------------------------
            # SMART ROUTING: Detect forecast vs general query
            # ---------------------------
            is_forecast_query = detect_forecast_question(user_question)

            t0 = time.perf_counter()
            if is_forecast_query:
                print("[ROUTING] Forecast question detected - using prediction pipeline")
                t_forecast_start = time.perf_counter()
                if run_prediction_tool is None:
                    from tools import run_prediction_tool
                response = run_prediction_tool(user_question, context_messages=recent_messages)

                t_forecast_end = time.perf_counter()
            else:
                print("[ROUTING] General query - using standard SQL pipeline")
                response = run_langchain_query_tool(user_question, context_messages=recent_messages)
            t1 = time.perf_counter()

            # === UNIFIED RESPONSE HANDLING ===
            if response.get("is_prediction", False):
                forecast_text = response.get("result", "")

                if forecast_text and isinstance(forecast_text, str):
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    st.session_state.chat_history[-1] = ("assistant", forecast_text, timestamp)

                    st.session_state.show_data_buttons = False
                    st.session_state.last_result_df = None

                    chat_container.markdown(render_chat_history(), unsafe_allow_html=True)

                    # Forecast timing
                    try:
                        overall_end = time.perf_counter()
                        memory_time = round(t_memory_end - t_memory_start, 3)
                        forecast_time = round(t_forecast_end - t_forecast_start, 3)
                        total_time = round(overall_end - overall_start, 3)

                        print("\n" + "="*60)
                        print("[FORECAST TIMING BREAKDOWN]")
                        print("="*60)
                        print(f"Memory retrieval: {memory_time}s")
                        print(f"Prediction total: {forecast_time}s")
                        print(f"Overall time: {total_time}s")
                        print("="*60 + "\n")
                    except:
                        pass

                    st.rerun()

                                            
            # Handle conversational responses
            if (isinstance(response.get("result"), list) and 
                len(response["result"]) > 0 and 
                "conversational_response" in response["result"][0]):
                conversational_text = response["result"][0]["conversational_response"]
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.chat_history[-1] = ("assistant", conversational_text, timestamp)
                st.session_state.show_data_buttons = False
                st.session_state.last_result_df = None
                chat_container.markdown(render_chat_history(), unsafe_allow_html=True)
                st.stop()
                
            else:
                # Build context for assistant
                if "error" in response:
                    context = f"The query failed. Error: {response['error']}"
                    query = None
                    result = None
                else:
                    query = response.get("query")
                    result = response.get("result")
                    try:
                        st.session_state.last_result_df = pd.DataFrame(result)
                        st.session_state.last_user_question = user_question
                        st.session_state.show_data_buttons = True
                    except Exception:
                        st.session_state.last_result_df = None
                        st.session_state.last_chart_config = None

                    instruction_context = (
                        "Summarize the query results in a natural language response. "
                        "Include key fields such as customer codes, customer names, and their total purchase value if present. "
                        "Be accurate, concise, and avoid hallucinations. "
                        "Format the summary clearly and optionally rank results if relevant (e.g., top 5 customers)."
                    )

            # ---------------------------
            # HYBRID: format locally, then send summary to Groq Chat API
            # ---------------------------
            t_format_start = time.perf_counter()
            if "error" in response:
                formatted_summary = f"The query failed. Error: {response['error']}"
            else:
                formatted_summary = format_answer(query, result)
            t_format_end = time.perf_counter()

            print("[DEBUG] format_answer output:\n", formatted_summary)

            MAX_ROWS_TO_SEND_FULL = 50
            MAX_SUMMARY_CHARS = 1500

            if result and isinstance(result, list) and len(result) <= MAX_ROWS_TO_SEND_FULL:
                formatted_summary_to_send = formatted_summary
                print("[DEBUG] Sending FULL formatted_summary to assistant (small result). Rows:", len(result))
            else:
                if formatted_summary and len(formatted_summary) > MAX_SUMMARY_CHARS:
                    formatted_summary_to_send = formatted_summary[:MAX_SUMMARY_CHARS].rstrip() + "\n\n... (truncated)"
                else:
                    formatted_summary_to_send = formatted_summary
                print("[DEBUG] Sending TRUNCATED formatted_summary_to_send length:", len(formatted_summary_to_send) if formatted_summary_to_send else 0)

            # Build instructions with the local summary
            if 'instruction_context' in locals():
                instructions = instruction_context + f"\n\nUser asked: {user_question}\nSQL executed: {query}\nSummary of result:\n{formatted_summary_to_send}"
            else:
                instructions = f"User asked: {user_question}\nResponse: {formatted_summary}"

            # ---------------------------
            # FAST DIRECT CHAT API CALL with SESSION MEMORY
            # ---------------------------
            t_chat_start = time.perf_counter()
            print("[DEBUG] Starting Groq chat API call with session memory...")

            try:
                client = Groq(api_key=GROQ_API_KEY)

                # Build messages with session context for memory
                messages = [
                    {
                        "role": "system", 
                        "content": "You are a helpful ERP assistant that answers business questions using database results. Be clear, concise, and business-friendly."
                    },
                    *recent_messages,
                    {
                        "role": "user",
                        "content": instructions
                    }
                ]

                # Streaming response
                assistant_reply = ""
                try:
                    response_stream = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",  
                        messages=messages,
                        temperature=0.1,
                        max_tokens=1300,
                        stream=True
                    )

                    for chunk in response_stream:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            new_token = delta.content
                            assistant_reply += new_token

                            # Update the last assistant message in chat history
                            st.session_state.chat_history[-1] = ("assistant", assistant_reply + " ▌", timestamp)

                            # Re-render the entire chat container with updated streaming content
                            chat_container.markdown(render_chat_history(), unsafe_allow_html=True)

                    # Final cleanup (remove cursor)
                    st.session_state.chat_history[-1] = ("assistant", assistant_reply, timestamp)
                    chat_container.markdown(render_chat_history(), unsafe_allow_html=True)

                except Exception as e:
                    t_chat_end = time.perf_counter()
                    st.error(f"Chat API error: {str(e)}")
                    assistant_reply = f"Error: {str(e)}"

                print(f"[DEBUG] Chat API response received: {assistant_reply[:200]}...")
                

            except Exception as e:
                t_chat_end = time.perf_counter()
                st.error(f"Chat API error: {str(e)}")
                print(f"[DEBUG] Chat API error: {str(e)}")

                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.chat_history.append(("user", user_question, timestamp))
                st.session_state.chat_history.append(("system", f"Chat API Error: {str(e)}", timestamp))

            overall_end = time.perf_counter()

            # Print timing breakdown
            try:
                langchain_time = round(t1 - t0, 3)
                format_time = round(t_format_end - t_format_start, 3)
                memory_time = round(t_memory_end - t_memory_start, 3)
                chat_time = round(t_chat_end - t_chat_start, 3)
                total_time = round(overall_end - overall_start, 3)
            except NameError:
                total_time = round(overall_end - overall_start, 3)
                print(f"[TIMINGS] Total time: {total_time}s (detailed breakdown unavailable)")

            print("[TIMINGS] formatted_summary length (full):", len(formatted_summary) if formatted_summary else 0)
            print("[TIMINGS] formatted_summary_to_send length:", len(formatted_summary_to_send) if formatted_summary_to_send else 0)
            print(f"[OPTIMIZED MEMORY] Retrieved {len(recent_messages)} messages from session")

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if (len(st.session_state.chat_history) > 0 and 
            st.session_state.chat_history[-1][1] == "🤔 Thinking..."):
            st.session_state.chat_history[-1] = ("system", f"Error: {str(e)}", timestamp)
        else:
            st.session_state.chat_history.append(("system", f"Error: {str(e)}", timestamp))
