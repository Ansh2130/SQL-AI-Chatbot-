# ====================================================
# prediction_engine.py — Groq-Powered Forecasting Analysis
# Purpose: Analyze SQL results and generate forecasts (NO SQL generation)
# ====================================================

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
import re
from groq import Groq


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

current_year = datetime.now().year


def detect_grouping_intent(user_question: str) -> dict:
    """
    Detect if user wants per-group forecasting (customer, item, etc.)
    Returns: {"group_by": "customer_code", "group_type": "customer"} or None
    """
    q = user_question.lower()
    
    top_pattern_match = re.search(
        r'\b(?:top|best|highest|most|leading|largest)\s+\d+(?:\s+\w+){0,3}?\s+(item|customer|product|sku)',
        q
    )
    
    if top_pattern_match:
        entity_type = top_pattern_match.group(1)
        if entity_type in ['item', 'product', 'sku']:
            print(f"[GROUPING DETECTION] Detected 'top N {entity_type}' pattern")
            return {"group_by": "product_code", "group_type": "item", "group_desc": "name"}
        elif entity_type == 'customer':
            print(f"[GROUPING DETECTION] Detected 'top N {entity_type}' pattern")
            return {"group_by": "customer_code", "group_type": "customer", "group_desc": "name"}
    
    customer_indicator_pattern = re.search(
        r"(most\s+profitable\s+customer|who\s+will\s+be|which\s+customer|best\s+customer|highest\s+customer|leading\s+customer|profitable\s+customers?)",
        q
    )
    if customer_indicator_pattern:
        print("[GROUPING DETECTION] Detected customer forecast via indicator phrase")
        return {"group_by": "customer_code", "group_type": "customer", "group_desc": "name"}
    
    customer_keywords = ["by customer", "per customer", "top customer", "each customer", 
                         "customer-wise", "customer wise", "for each customer"]
    if any(kw in q for kw in customer_keywords):
        print("[GROUPING DETECTION] Detected customer grouping via keyword")
        return {"group_by": "customer_code", "group_type": "customer", "group_desc": "name"}
    
    item_keywords = ["by item", "per item", "by product", "per product", "product-wise",
                     "item-wise", "by sku", "per sku"]
    if any(kw in q for kw in item_keywords):
        print("[GROUPING DETECTION] Detected item grouping via keyword")
        return {"group_by": "product_code", "group_type": "item", "group_desc": "name"}
    
    print("[GROUPING DETECTION] No grouping detected - total forecast")
    return None

def smart_aggregate_forecast_data(user_question: str, sql_result: list, grouping: dict) -> list:
    """
    Pre-aggregate forecast data to reduce token count while maintaining accuracy.
    """ 
    if not sql_result:
        return []
    
    if not grouping:
        return sql_result
    
    df = pd.DataFrame(sql_result)
    if 'totalsales' in df.columns and 'totalsales' not in df.columns:
        df.rename(columns={'totalsales': 'totalsales'}, inplace=True)
        print("[PRE-AGGREGATION]  Normalized 'totalsales' → 'totalsales'")

    top_n_match = re.search(
        r'\b(?:top|most|best|highest|leading|largest|biggest)\s+(\d+)(?:\s+\w+){0,3}?\s+(?:item|customer|product|sku)', 
        user_question.lower()
    )
    
    if not top_n_match:
        top_n_match = re.search(
            r'\b(?:top|most|best|highest|leading|largest|biggest)\s+(\d+)\b', 
            user_question.lower()
        )
    
    if not top_n_match:
        top_n_match = re.search(
            r'\b(\d+)\s+(?:most|best|highest|top|leading|largest|biggest)\b',
            user_question.lower()
        )
    
    if top_n_match:
        top_n = int(top_n_match.group(1))
        buffer_multiplier = max(2, min(3, 15 // top_n))
        fetch_n = min(top_n * buffer_multiplier, 50)
        
        print(f"[PRE-AGGREGATION] User wants top {top_n}, fetching top {fetch_n} for LLM analysis")
        
        group_col = grouping['group_by']
        group_totals = df.groupby(group_col)['totalsales'].sum().nlargest(fetch_n)
        top_groups = group_totals.index.tolist()
        
        df_filtered = df[df[group_col].isin(top_groups)]
        
        print(f"[PRE-AGGREGATION] Reduced from {len(df)} rows to {len(df_filtered)} rows (top {top_n} {grouping['group_type']}s)")
        return df_filtered.to_dict('records')
    
    else:
        print(f"[PRE-AGGREGATION] No 'top N' detected - keeping all {grouping['group_type']}s but aggregating")
        
        MAX_ROWS_FOR_FORECAST = 2500
        if len(df) > MAX_ROWS_FOR_FORECAST:
            print(f"[PRE-AGGREGATION] WARNING: {len(df)} rows exceeds safe limit. Truncating to {MAX_ROWS_FOR_FORECAST}")
            
            if grouping:
                group_col = grouping['group_by']
                group_totals = df.groupby(group_col)['totalsales'].sum().nlargest(50)
                top_groups = group_totals.index.tolist()
                df = df[df[group_col].isin(top_groups)]
                print(f"[PRE-AGGREGATION] Filtered to top 50 {grouping['group_type']}s: {len(df)} rows")
            else:
                df = df.head(MAX_ROWS_FOR_FORECAST)
            
            return df.to_dict('records')
        
        return sql_result


def compute_summary_values(sql_result: list, grouping: dict = None) -> dict:
    df = pd.DataFrame(sql_result)
    if 'totalsales' in df.columns and 'totalsales' not in df.columns:
        df.rename(columns={'totalsales': 'totalsales'}, inplace=True)
        print("[SUMMARY]  Normalized 'totalsales' → 'totalsales'")
    elif 'totalsales' in df.columns and 'totalsales' not in df.columns:
        df.rename(columns={'totalsales': 'totalsales'}, inplace=True)
        print("[SUMMARY]  Normalized 'totalsales' → 'totalsales'")

    if df.empty:
        return {}

    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].fillna(0).astype(int)
    df['totalsales'] = df['totalsales'].astype(float)

    current_year = datetime.now().year
    last_year = current_year - 1

    if grouping:
        group_col = grouping['group_by']
        group_desc = grouping.get('group_desc', group_col)
        
        current_year_data = df[df['year'] == current_year]
        if current_year_data.empty:
            return {"error": "No current year data for grouped forecast"}
        
        max_month_current = current_year_data['month'].max()

        start_time = time.perf_counter()

        def compute_entity_summary(entity_code: str):
            entity_df = df[df[group_col] == entity_code]
            entity_name = entity_df[group_desc].iloc[0] if group_desc in entity_df.columns else entity_code

            jan_to_max_current = entity_df[
                (entity_df['year'] == current_year) & 
                (entity_df['month'].between(1, max_month_current))
            ]['totalsales'].sum()

            jan_to_max_last = entity_df[
                (entity_df['year'] == last_year) & 
                (entity_df['month'].between(1, max_month_current))
            ]['totalsales'].sum()

            remaining_months_last = entity_df[
                (entity_df['year'] == last_year) & 
                (entity_df['month'].between(max_month_current + 1, 12))
            ]['totalsales'].sum()

            full_last_year_entity = entity_df[entity_df['year'] == last_year]['totalsales'].sum()

            try:
                growth_pct = ((jan_to_max_current - jan_to_max_last) / jan_to_max_last) * 100 if jan_to_max_last != 0 else 0
            except ZeroDivisionError:
                growth_pct = 0.0

            est_remaining = remaining_months_last * (1 + (growth_pct / 100))
            completed = jan_to_max_current + est_remaining

            return {
                "entity_code": str(entity_code).strip(),
                "entity_name": str(entity_name).strip(),
                "max_month_current": int(max_month_current),
                "jan_to_max_current": round(float(jan_to_max_current), 2),
                "jan_to_max_last": round(float(jan_to_max_last), 2),
                "full_last_year": round(float(full_last_year_entity), 2),
                "remaining_months_last": round(float(remaining_months_last), 2),
                "growth_pct": round(float(growth_pct), 2),
                "est_remaining_current": round(float(est_remaining), 2),
                "completed_current": round(float(completed), 2)
            }

        entity_summaries = []
        entity_codes = df[group_col].dropna().unique()

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(compute_entity_summary, code): code for code in entity_codes}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    entity_summaries.append(result)
                except Exception as e:
                    print(f"[ERROR] Failed for entity {futures[future]}: {str(e)}")

        entity_summaries.sort(key=lambda x: x['completed_current'], reverse=True)

        duration = time.perf_counter() - start_time
        print(f"[TIMING] compute_summary (grouped): {duration:.2f}s")

        print(f"\n [SUMMARY] Computed {len(entity_summaries)} entity-level summaries")
        print(f"[SUMMARY] Top 3 entities by completed {current_year}:")
        for i, entity in enumerate(entity_summaries[:3], 1):
            print(f"  {i}. {entity['entity_name']}: ${entity['completed_current']:,.2f}")
        
        return {
            "grouped": True,
            "group_type": grouping['group_type'],
            "max_month_current": int(max_month_current),
            "entities": entity_summaries
        }
    
    # Total forecast logic (no grouping)
    available_months_current = df[df['year'] == current_year]['month']
    if available_months_current.empty:
        print("[WARNING] No monthly data found for current year — check SQL results.")
        return {}

    max_month_current = available_months_current.max()

    jan_to_max_current = df[(df['year'] == current_year) & (df['month'].between(1, max_month_current))]['totalsales'].sum()
    jan_to_max_last = df[(df['year'] == last_year) & (df['month'].between(1, max_month_current))]['totalsales'].sum()
    remaining_months_last = df[(df['year'] == last_year) & (df['month'].between(max_month_current + 1, 12))]['totalsales'].sum()

    try:
        growth_pct = ((jan_to_max_current - jan_to_max_last) / jan_to_max_last) * 100
    except ZeroDivisionError:
        growth_pct = 0.0

    est_remaining_months_current = remaining_months_last * (1 + (growth_pct / 100))
    completed_current = jan_to_max_current + est_remaining_months_current

    full_last_year_total = df[df['year'] == last_year]['totalsales'].sum()

    return {
        "grouped": False,
        "max_month_current": int(max_month_current),
        "jan_to_max_current": round(float(jan_to_max_current), 2),
        "jan_to_max_last": round(float(jan_to_max_last), 2),
        "remaining_months_last": round(float(remaining_months_last), 2),
        "growth_pct": round(float(growth_pct), 2),
        "est_remaining_current": round(float(est_remaining_months_current), 2),
        "completed_current": round(float(completed_current), 2),
        "full_last_year_total": round(float(full_last_year_total), 2)
    }

   

def generate_forecast_with_gpt(user_question: str, sql_result: list, grouping: dict) -> str:
    """
    Use Groq LLM to:
    1. Complete partial current year using prior year pattern
    2. Forecast next year using historical trends
    3. Return business-style formatted response
    """
    if not sql_result:
        return "Unable to generate forecast - no historical data available."

    sql_result = smart_aggregate_forecast_data(user_question, sql_result, grouping)

    data_json = json.dumps(sql_result, indent=2, default=str)
    summary = compute_summary_values(sql_result, grouping)

    if not summary:
        print("\n [SUMMARY WARNING] No current year summary data found.")
        return " Unable to generate forecast - no current year data available."

    elif summary.get("grouped"):
        print("\n [SUMMARY] Entity-level summaries computed successfully")
        month_int = int(summary['max_month_current'])
        month_name = datetime(1900, month_int, 1).strftime('%b')

        entity_lines = []
        for entity in summary['entities'][:20]:
            entity_lines.append(
                f"  - **{entity['entity_name']}** (Code: {entity['entity_code']})\n"
                f"    - Full {current_year - 1}: ${entity.get('full_last_year', 'N/A'):,.2f}\n"
                f"    - Jan–{month_name} {current_year}: ${entity['jan_to_max_current']:,.2f}\n"
                f"    - Jan–{month_name} {current_year - 1}: ${entity['jan_to_max_last']:,.2f}\n"
                f"    - Growth: {entity['growth_pct']:.1f}%\n"
                f"    - Estimated {current_year} Total: ${entity['completed_current']:,.2f}\n"
            )

        summary_block = f"""
 Per-{summary['group_type'].capitalize()} Precomputed Summaries:

**USE THESE VALUES - DO NOT RECALCULATE FROM RAW DATA**

CRITICAL: When showing historical {current_year - 1} data, use the \"FULL {current_year - 1} TOTAL\" values below, NOT the sum of months from raw data.

Available months in {current_year}: Jan–{month_name}

{chr(10).join(entity_lines)}

**INSTRUCTIONS FOR FORECASTING:**
    1. Use the \"Estimated {current_year} Total\" as your baseline for each {summary['group_type']}
    2. Apply historical trends from the raw data to project {current_year + 1}
    3. DO NOT output $X,XXX,XXX placeholders - calculate actual values
"""

        wants_top_n_forecast = re.search(
            r'\b(?:top|best|highest)\s+\d+.*(?:for|in|next year|2026)',
            user_question.lower()
        )

        ranking_instruction = ""
        if wants_top_n_forecast:
            ranking_instruction = f"""

        ** RANKING REQUIREMENT:**
        The user asked for "top N {grouping['group_type']}s for next year".
        After forecasting {current_year + 1} for ALL {grouping['group_type']}s:
        1. Rank them by their PROJECTED {current_year + 1} values (NOT historical totals)
        2. Return results in descending order by forecasted value
        3. Show ONLY the top N requested in the main sections
        4. Include a summary table with all forecasted {grouping['group_type']}s for reference
        """

        grouping_context = f"""
        DATA STRUCTURE:
        - This data is grouped by {grouping['group_type'].upper()}
        - Each row represents a specific {grouping['group_type']}
        - You MUST forecast separately for EACH {grouping['group_type']}

        GROUPING COLUMNS:
        - {grouping['group_by']}: The {grouping['group_type']} identifier
        - {grouping['group_desc']}: The {grouping['group_type']} name/description

        {ranking_instruction}
        """
        

        start_time = time.perf_counter()
        client = Groq(api_key=GROQ_API_KEY)

        def forecast_entity(entity):
            print(f"[DEBUG]  Forecasting: {entity['entity_name']} (Code: {entity['entity_code']})")
            try:
                entity_data = [r for r in sql_result if r.get(grouping['group_by']) == entity['entity_code']]
                entity_json = json.dumps(entity_data[:50], indent=2, default=str)
                
                entity_prompt = f"""You are a business analyst expert in financial forecasting. 

CONTEXT:
- Today's date: {datetime.now().strftime('%Y-%m-%d')}
- Current year: {current_year} (partially complete through {datetime.now().strftime('%B')})
- Next year to forecast: {current_year + 1}

{grouping_context}

{summary_block}

ENTITY SUMMARY:
- **{entity['entity_name']}** (Code: {entity['entity_code']})
- Full {current_year - 1}: ${entity.get('full_last_year', 'N/A'):,.2f}
- Jan–{month_name} {current_year}: ${entity['jan_to_max_current']:,.2f}
- Jan–{month_name} {current_year - 1}: ${entity['jan_to_max_last']:,.2f}
- Growth: {entity['growth_pct']:.1f}%
- Estimated {current_year} Total: ${entity['completed_current']:,.2f}

IMPORTANT BASELINE RULE:
- Use the "Full {current_year - 1}" value ONLY for projecting {current_year + 1} trends.
- Do NOT use it when estimating the current year; that must be based on Jan–{month_name} actuals + estimated remaining.

HISTORICAL DATA FROM SQL:
{entity_json}

YOUR TASK:

1. **COMPLETE {current_year}** (if partial data exists):
   - Calculate actual current-year total using available months only
   - Compare same months from last year to calculate growth rate
   - Apply this growth rate to remaining months
   - Complete {current_year} = actual + estimated remaining

2. **FORECAST {current_year + 1}**:
   - Use completed {current_year} + historical data
   - Identify growth trend and apply reasonable projection

3. **FORMAT RESPONSE**:

YOU ARE FORECASTING ONLY THIS ONE ENTITY: {entity['entity_name']} (Code: {entity['entity_code']})

DO NOT generate sections for other entities.
DO NOT include document-level headers.
DO NOT include disclaimers.

Output ONLY this entity's section in this exact format:

## {entity['entity_name']} (Code: {entity['entity_code']})

* **Forecasted Value for {current_year + 1}:** $[CALCULATE_EXACT_NUMBER]

### Summary
* Completed {current_year} Estimate: ${entity['completed_current']:,.2f}
* {current_year + 1} Projection: $[VALUE_FROM_FORECAST]

### {current_year + 1} vs. {current_year}
* **Formula:** % Change = (({current_year + 1} Projection - {current_year} Estimate) / {current_year} Estimate) × 100
* **Calculation:** (Projection - Estimate) / Estimate × 100 ≈ X.XX%


### Historical Data
[List year-wise values for THIS entity from the data]

### Trend Analysis
[Describe THIS entity's specific growth pattern with percentages]

### Methodology
[Explain how you calculated {current_year + 1} for THIS entity using the ${entity['completed_current']:,.2f} baseline]

CRITICAL RULES:
- Start output directly with "## {entity['entity_name']}"
- Output ONLY this entity's section (no headers, no other entities)
- Calculate exact forecast value (no placeholders)
- Keep response 300-400 words
- Show formulas: **Formula: % Change = ((New - Old) / Old) × 100**
- Format numbers with commas: $1,234,567

Generate forecast now:"""

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": entity_prompt}],
                    temperature=0,
                    max_tokens=1000
                )

                forecast_content = response.choices[0].message.content.strip()

                if not forecast_content:
                    print(f"[ERROR]  Empty LLM response for {entity['entity_name']}")
                    forecast_content = f"## {entity['entity_name']}\n[Error: LLM returned empty response]"
                else:
                    print(f"[DEBUG]  Forecast for {entity['entity_name']}: {len(forecast_content)} chars")

                return {
                    "entity_code": entity['entity_code'],
                    "entity_name": entity['entity_name'],
                    "forecast_text": forecast_content
                }

            except Exception as e:
                print(f"[ERROR] Failed to forecast {entity['entity_name']}: {str(e)}")
                return {
                    "entity_code": entity['entity_code'],
                    "entity_name": entity['entity_name'],
                    "forecast_text": f"## {entity['entity_name']}\n[Error: {str(e)}]"
                }

        forecast_results = []
        entities = summary['entities'][:20]

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(forecast_entity, ent): ent['entity_code'] for ent in entities}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    forecast_results.append(result)
                except Exception as e:
                    print(f"[ERROR] Executor failed: {str(e)}")

        duration = time.perf_counter() - start_time
        print(f"[TIMING] Parallel LLM forecasts completed in {duration:.2f}s")

        if not forecast_results:
            return "Forecast could not be generated - no entity forecasts were produced."

        def extract_forecast_value(text: str) -> float:
            match = re.search(r'\*\*Forecasted Value for \d+:\*\*\s*\$?([\d,]+)', text)
            return float(match.group(1).replace(',', '')) if match else 0.0

        for result in forecast_results:
            result['forecast_2026'] = extract_forecast_value(result['forecast_text'])

        forecast_results.sort(key=lambda x: x['forecast_2026'], reverse=True)

        top_n_match = re.search(r'\b(?:top|best|highest)\s+(\d+)\b', user_question.lower())
        
        if top_n_match:
            requested_n = int(top_n_match.group(1))
            forecast_results = forecast_results[:requested_n]
            header = f"# Top {requested_n} {summary['group_type'].capitalize()}s - {current_year + 1} Forecast\n\n"
            print(f"[FORECAST] Returning top {requested_n} ranked by {current_year + 1} forecast")
        else:
            requested_n = len(forecast_results)
            header = f"**{summary['group_type'].capitalize()}-Wise Forecast for {current_year + 1}:**\n\n"
            print(f"[FORECAST] Returning all {requested_n} {summary['group_type']}s ranked by {current_year + 1} forecast")
        
        seen_codes = set()
        unique_results = []
        for res in forecast_results:
            if res['entity_code'] not in seen_codes:
                seen_codes.add(res['entity_code'])
                unique_results.append(res)
        
        intro = f"The following analysis provides {current_year + 1} forecasts for {requested_n} {summary['group_type']}(s) based on historical data.\n\n"
        disclaimer = "**Disclaimer:** This projection is based on historical data and should be used as a guide.\n\n"
        entity_sections = "\n\n---\n\n".join([res['forecast_text'] for res in unique_results])
        
        forecast_response = header + intro + disclaimer + "---\n\n" + entity_sections
        print(f"[DEBUG] Final forecast response: {len(forecast_response)} chars, {len(unique_results)} entities")
        return forecast_response

    else:
        print("\n[SUMMARY] Total summary computed successfully")
        month_int = int(summary['max_month_current'])
        month_name = datetime(1900, month_int, 1).strftime('%b')

        grouping_context = """
DATA STRUCTURE:
- This data shows TOTAL values (not grouped by customer/item)
- Each row represents a time period only
"""
        response_format = f"""
Write a brief 1-2 sentence introduction about the forecast analysis, then immediately follow with: **Disclaimer: This projection is based on historical data analysis and should be used as a guide rather than a definitive forecast. Actual results may vary due to market conditions, business decisions, and external factors.**

# Forecasted Total for {current_year + 1}

* **Final Forecasted Value:** $X,XXX,XXX

# Summary

* Provide structured numerical breakdown
* Show completed {current_year} estimate
* Show {current_year + 1} projection

# {current_year + 1} vs. {current_year}:

* **Formula:** % Change = (({current_year + 1} Projection - {current_year} Estimate) / {current_year} Estimate) × 100
* **Calculation:** (Projection - Estimate) / Estimate × 100 ≈ X.XX%

# Explanation

* **Historical Data:** Show year-wise figures
* **Year-over-Year Changes:** Show % changes WITH FORMULAS
  * **Formula: % Change = ((New Value - Old Value) / Old Value) × 100**
  * **Calculation: (X - Y) / Y × 100 ≈ Z%**
* **Trend Analysis:** Explain pattern
* **Projection Method:** Step-by-step calculation
"""
        is_cashflow = any(word in user_question.lower() for word in ["cashflow", "cash flow", "liquidity", "cash position", "cash"])

        def calculate_cashflow_forecast(monthly_data_text: str, user_question: str) -> dict:
            """Calculate Oct-Dec 2025 forecast using actual math (not LLM)"""
            data_lines = monthly_data_text.strip().split('\n')
            monthly_values = {}
            for line in data_lines:
                match = re.match(r'(\d{4})-(\d{1,2}):\s*([-\d,\.]+)', line)
                if match:
                    year, month, value = match.groups()
                    month_num = int(month)
                    value = float(value.replace(',', ''))
                    key = f"{year}-{month_num}"
                    monthly_values[key] = value
            
            print(f"[CALC DEBUG] Parsed {len(monthly_values)} months of data")
            jan_sep_2024 = sum(monthly_values.get(f"2024-{m}", 0) for m in range(1, 10))
            jan_sep_2025 = sum(monthly_values.get(f"2025-{m}", 0) for m in range(1, 10))
            
            if jan_sep_2024 > 0:
                growth_rate = ((jan_sep_2025 - jan_sep_2024) / jan_sep_2024) * 100
            else:
                growth_rate = 0.0
            
            multiplier = 1 + (growth_rate / 100)
            
            oct_2024 = monthly_values.get("2024-10", 0)
            nov_2024 = monthly_values.get("2024-11", 0)
            dec_2024 = monthly_values.get("2024-12", 0)
            
            oct_2025 = oct_2024 * multiplier
            nov_2025 = nov_2024 * multiplier
            dec_2025 = dec_2024 * multiplier
            total_forecast = oct_2025 + nov_2025 + dec_2025
            
            starting_balance = 0
            balance_patterns = [
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:rs|rupees|amount|dollars?|\$)?.*(?:in|at).*bank',
                r'(?:have|got)\s+(\d+(?:,\d{3})*(?:\.\d+)?)',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s+(?:in|at)\s+(?:my|the)?\s*bank'
            ]
            for pattern in balance_patterns:
                balance_match = re.search(pattern, user_question.lower())
                if balance_match:
                    starting_balance = float(balance_match.group(1).replace(',', ''))
                    break
            
            return {
                'oct_2025': oct_2025,
                'nov_2025': nov_2025,
                'dec_2025': dec_2025,
                'total_forecast': total_forecast,
                'starting_balance': starting_balance,
                'year_end_total': total_forecast + starting_balance,
                'growth_rate': growth_rate
            }

        if is_cashflow:
            df = pd.DataFrame(sql_result)
            if 'totalsales' in df.columns and 'totalsales' not in df.columns:
                df.rename(columns={'totalsales': 'totalsales'}, inplace=True)

            monthly_totals = (
                df.groupby(['Year', 'Month'])['totalsales']
                .sum()
                .reset_index()
                .sort_values(['Year', 'Month'])
            )

            monthly_data_text = "\n".join([
                f"{int(row.year)}-{int(row.month):02d}: {row.totalsales:,.2f}"
                for _, row in monthly_totals.iterrows()
            ])

            try:
                forecast_calc = calculate_cashflow_forecast(monthly_data_text, user_question)
                print(f"[DEBUG] Forecast calculation succeeded: {forecast_calc}")
            except Exception as e:
                print(f"[ERROR] Forecast calculation failed: {str(e)}")
                return f"Unable to calculate forecast: {str(e)}"
            
            full_prompt = f"""You are a financial analyst specialized in **cashflow forecasting**.
 
CONTEXT:
- Today's date: {datetime.now().strftime('%Y-%m-%d')}
- Current year: {current_year} (data available up to September)
- Domain: CASHFLOW (short-term monthly prediction)
 
IMPORTANT: Return **only** a concise user-friendly summary.
 
FORECASTED VALUES (PRE-CALCULATED):
- Growth Rate: {forecast_calc['growth_rate']:.2f}%
- Forecasted Oct 2025: ${forecast_calc['oct_2025']:,.2f}
- Forecasted Nov 2025: ${forecast_calc['nov_2025']:,.2f}
- Forecasted Dec 2025: ${forecast_calc['dec_2025']:,.2f}
- Total Forecasted Cashflow: ${forecast_calc['total_forecast']:,.2f}
- Starting Balance: ${forecast_calc['starting_balance']:,.2f}
- Year-End Total: ${forecast_calc['year_end_total']:,.2f}
 
YOUR TASK:
Format the above values with brief reasoning about business trends.
 
**Output in this exact format**:

## Forecasted Cashflow (Oct–Dec 2025)
 
* **Forecasted Oct 2025:** ${forecast_calc['oct_2025']:,.2f}
* **Forecasted Nov 2025:** ${forecast_calc['nov_2025']:,.2f}
* **Forecasted Dec 2025:** ${forecast_calc['dec_2025']:,.2f}
* **Total Forecasted 2025 Cashflow:** ${forecast_calc['total_forecast']:,.2f}
* **Starting Balance:** ${forecast_calc['starting_balance']:,.2f}
* **Year-End Total:** ${forecast_calc['year_end_total']:,.2f}
 
# Methodology
- Growth Rate Applied: {forecast_calc['growth_rate']:.2f}%
- Formula: Oct–Dec 2024 × (1 + Growth%)
- Add 2-3 sentences about business trends and seasonality
 
# Disclaimer
**This projection is based on historical data patterns and should be treated as an indicative forecast only.**
 
Generate forecast now:"""
            try:
                client = Groq(api_key=GROQ_API_KEY)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0,
                    max_tokens=1000
                )
            
                forecast_text = response.choices[0].message.content.strip()
                print(f"[DEBUG] Cashflow forecast generated: {len(forecast_text)} chars")
                return forecast_text
            
            except Exception as e:
                print(f"[ERROR] Cashflow forecast failed: {str(e)}")
                return f" Cashflow forecast generation failed: {str(e)}"   

        else:
            full_prompt = f"""You are a business analyst expert in financial forecasting. 
 
 CONTEXT:
 - Today's date: {datetime.now().strftime('%Y-%m-%d')}
 - Current year: {current_year} (partially complete through {datetime.now().strftime('%B')})
 - Next year to forecast: {current_year + 1}
 
 {grouping_context}
 
 PRECOMPUTED SUMMARY (DO NOT RECALCULATE):
 - Jan–{month_name} {current_year}: ${summary['jan_to_max_current']:,.2f}
 - Jan–{month_name} {current_year - 1}: ${summary['jan_to_max_last']:,.2f}
 - Growth Rate: {summary['growth_pct']:.2f}%
 - Estimated Remaining ({month_name}–Dec {current_year}): ${summary['est_remaining_current']:,.2f}
 - Estimated Total for {current_year}: ${summary['completed_current']:,.2f}
 - Full {current_year - 1} Total (Actual): ${summary['full_last_year_total']:,.2f}
 
 IMPORTANT BASELINE RULE:
 - Use the "Full {current_year - 1} Total (Actual)" ONLY when projecting {current_year + 1} (next year) trends.
 - Do NOT use it to recompute the current year's estimate.
 
 
 USER QUESTION: {user_question}
 
 HISTORICAL DATA FROM SQL:
 {data_json}
 
 YOUR TASK:
 
 1. **COMPLETE {current_year}** (if partial data exists):
 - Calculate actual current-year total using available months only
 - Compare same months from last year to calculate growth rate
 - Apply this growth rate to remaining months
 - Complete {current_year} = actual + estimated remaining
 
 2. **FORECAST {current_year + 1}**:
 - Use completed {current_year} + historical data
 - Identify growth trend and apply reasonable projection
 
 3. **FORMAT RESPONSE**:
 {response_format}
 
 # Disclaimer
 
 **[Disclaimer: This projection is based on historical data analysis and should be used as a guide rather than a definitive forecast.]**
 
 IMPORTANT FORMATTING RULES:
 - Use # for main section headers
 - Show formulas explicitly
 - Make disclaimers BOLD
 - Use bullet points for structured data
 - Format large numbers with commas: $1,234,567
 
 Generate forecast now:"""

            try:
                client = Groq(api_key=GROQ_API_KEY)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0,
                    max_tokens=1000
                )

                forecast_text = response.choices[0].message.content.strip()
                print(f"[DEBUG] Total forecast generated: {len(forecast_text)} chars")
                return forecast_text

            except Exception as e:
                print(f"[ERROR] LLM call failed: {str(e)}")
                return f" Forecast generation failed: {str(e)}"
       


def analyze_forecast_results(user_question: str, sql_result: list) -> dict:
    """
    Main entry point for prediction_engine.py
    Receives SQL results from langagent.py and generates forecast.
    
    Returns: {"result": forecast_text, "is_prediction": True}
    """
    print(f"[PREDICTION ENGINE] Analyzing results for: {user_question}")
    print(f"[PREDICTION ENGINE] Received {len(sql_result) if sql_result else 0} rows")

    if sql_result and isinstance(sql_result, list):
        first_row = sql_result[0]
        if "totalsales" in first_row and "totalsales" not in first_row:
            print("[PREDICTION ENGINE] Normalizing 'totalsales' → 'totalsales'")
            sql_result = [{**row, "totalsales": row["totalsales"]} for row in sql_result]
        elif "totalsales" in first_row and "totalsales" not in first_row:
            print("[PREDICTION ENGINE] Normalizing 'totalsales' → 'totalsales'")
            sql_result = [{**row, "totalsales": row["totalsales"]} for row in sql_result]
    try:
        grouping = detect_grouping_intent(user_question)
        if grouping:
            print(f"[PREDICTION ENGINE] Detected grouping: {grouping['group_type']}")
        else:
            print("[PREDICTION ENGINE] No grouping detected - forecasting total only")
        
        forecast_text = generate_forecast_with_gpt(user_question, sql_result, grouping)
        print("\n[FINAL FORECAST OUTPUT]")
        print(forecast_text)
        return {
            "result": forecast_text,
            "is_prediction": True,
            "raw_data": sql_result
        }
    except Exception as e:
        print(f"[PREDICTION ENGINE] Analysis error: {str(e)}")
        return {"error": f"Forecast analysis failed: {str(e)}"}