import re
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tools import get_table_list, describe_table, execute_sql_query, validate_sql_query
import time
from datetime import datetime

current_year = datetime.now().year

load_dotenv()

SCHEMA_FILE = Path("schema_cache.json")

# ------------------------
# Schema Cache
# ------------------------

def build_schema_cache():
    print("Building schema cache...")
    schema_map = {}
    try:
        tables = get_table_list()
        print(f"Found {len(tables)} tables")
        filtered_tables = [t for t in tables if not t.lower().endswith("_log")]
        for i, table in enumerate(filtered_tables):
            try:
                desc = describe_table(table)
                schema_map[table] = desc
            except Exception as e:
                print(f"Error processing table {table}: {str(e)}")
                continue
        with open(SCHEMA_FILE, "w") as f:
            json.dump(schema_map, f, indent=2)
        print(f"Schema cache built! Cached {len(schema_map)} tables.")
        return schema_map
    except Exception as e:
        print(f"Error building schema cache: {str(e)}")
        return {}


def load_schema_cache():
    try:
        if SCHEMA_FILE.exists():
            with open(SCHEMA_FILE, "r") as f:
                schema_data = json.load(f)
            if schema_data:
                print(f"Loaded schema cache with {len(schema_data)} tables")
                return schema_data
            else:
                return build_schema_cache()
        else:
            return build_schema_cache()
    except Exception as e:
        print(f"Error loading schema cache: {str(e)}")
        return build_schema_cache()

# ------------------------
# Table Selection
# ------------------------

def get_relevant_tables(user_question: str, schema_map: dict, max_tables: int = 6, max_columns: int = 12):
    q = user_question.lower()
    filtered = {}

    keyword_map = {
        "student": ["public.students", "public.enrollments", "public.departments"],
        "students": ["public.students", "public.enrollments", "public.departments"],
        "topper": ["public.students", "public.departments"],
        "cgpa": ["public.students", "public.departments"],
        "admission": ["public.students", "public.departments"],
        "enroll": ["public.enrollments", "public.students", "public.courses"],
        "enrollment": ["public.enrollments", "public.students", "public.courses"],
        "enrolled": ["public.enrollments", "public.students", "public.courses"],
        "course": ["public.courses", "public.faculty", "public.departments"],
        "courses": ["public.courses", "public.faculty", "public.departments"],
        "subject": ["public.courses", "public.faculty"],
        "grade": ["public.enrollments", "public.students", "public.courses"],
        "grades": ["public.enrollments", "public.students", "public.courses"],
        "marks": ["public.enrollments", "public.students", "public.courses"],
        "result": ["public.enrollments", "public.students", "public.courses"],
        "results": ["public.enrollments", "public.students", "public.courses"],
        "attendance": ["public.enrollments", "public.students", "public.courses"],
        "faculty": ["public.faculty", "public.departments", "public.courses"],
        "professor": ["public.faculty", "public.departments"],
        "teacher": ["public.faculty", "public.departments"],
        "department": ["public.departments", "public.students", "public.faculty"],
        "departments": ["public.departments", "public.students", "public.faculty"],
        "dept": ["public.departments", "public.students", "public.faculty"],
        "fee": ["public.fees", "public.fee_payments", "public.students"],
        "fees": ["public.fees", "public.fee_payments", "public.students"],
        "tuition": ["public.fees", "public.students"],
        "hostel": ["public.fees", "public.students"],
        "payment": ["public.fee_payments", "public.fees", "public.students"],
        "payments": ["public.fee_payments", "public.fees", "public.students"],
        "paid": ["public.fees", "public.fee_payments", "public.students"],
        "unpaid": ["public.fees", "public.students"],
        "pending": ["public.fees", "public.students"],
        "overdue": ["public.fees", "public.students"],
        "outstanding": ["public.fees", "public.students"],
        "collection": ["public.fee_payments", "public.fees"],
        "revenue": ["public.fee_payments", "public.fees"],
        "salary": ["public.faculty", "public.departments"],
        "top": ["public.students", "public.enrollments"],
        "fail": ["public.enrollments", "public.students"],
        "failed": ["public.enrollments", "public.students"],
        "pass": ["public.enrollments", "public.students"],
        "drop": ["public.students", "public.enrollments"],
        "dropped": ["public.students", "public.enrollments"],
        "graduate": ["public.students", "public.departments"],
        "graduated": ["public.students", "public.departments"],
        "elective": ["public.courses", "public.enrollments"],
        "lab": ["public.courses", "public.enrollments"],
        "semester": ["public.students", "public.enrollments", "public.fees"],
        "batch": ["public.students", "public.departments"],
        "year": ["public.students", "public.enrollments"],
    }

    matched = []
    for keyword, tables in keyword_map.items():
        if keyword in q.split() or keyword in q:
            matched.append(keyword)
            for t in tables:
                if t in schema_map:
                    filtered[t] = {"columns": schema_map[t].get("columns", [])[:max_columns]}

    if not filtered:
        fallback_triggers = ["student", "course", "fee", "faculty", "department", "how many", "total", "list", "show"]
        if any(w in q for w in fallback_triggers):
            for t in ["public.students", "public.enrollments", "public.departments"]:
                if t in schema_map:
                    filtered[t] = {"columns": schema_map[t].get("columns", [])[:max_columns]}

    print(f"[DEBUG] Matched keywords: {matched}, Tables: {list(filtered.keys())}")
    return filtered

# ------------------------
# SQL Generation
# ------------------------

def generate_sql_with_llm(user_question: str, schema_map: dict, last_error: str = "", context_messages: list = None, is_forecast: bool = False) -> str:
    relevant_schema = get_relevant_tables(user_question, schema_map, max_tables=8, max_columns=15)

    schema_parts = []
    for table_name, table_info in relevant_schema.items():
        columns = table_info.get("columns", [])
        col_list = [f"{col['name']} ({col['type']})" for col in columns]
        schema_parts.append(f"{table_name}: {', '.join(col_list)}")
    schema_text = "\n".join(schema_parts)

    context_text = ""
    if context_messages:
        context_text = "\n\nRecent conversation:\n"
        for msg in context_messages[-4:]:
            context_text += f"{msg.get('role','')}: {msg.get('content','')}\n"

    forecast_hint = ""
    if is_forecast:
        forecast_hint = f"""
FORECAST MODE:
- Return historical fee collection data grouped by academic_year
- Use: SELECT academic_year AS year, SUM(amount) AS totalsales FROM public.fee_payments GROUP BY academic_year ORDER BY year
- For department-wise: add dept_code grouping via JOIN with students
- Always alias as totalsales for prediction engine compatibility
"""

    prompt = f"""You are a PostgreSQL expert for a COLLEGE MANAGEMENT SYSTEM{"" if is_forecast else " that can also handle conversational questions"}.

{"" if is_forecast else '''FOR CONVERSATIONAL QUESTIONS (greetings, general knowledge, anything NOT about college data):
- Respond with: CONVERSATIONAL_RESPONSE: [short answer in 5-6 words.
Then append: "Although I can answer this, it is outside my role. I am eager to assist with college data questions."]
'''}
{forecast_hint}

DATABASE: College Management System with students, courses, enrollments, fees, faculty, departments.

AVAILABLE TABLES:
{schema_text}

PREFERRED JOINS:
  * Student + Department:
    public.students s JOIN public.departments d ON s.dept_code = d.dept_code

  * Enrollment + Student + Course:
    public.enrollments e
    JOIN public.students s ON e.roll_no = s.roll_no
    JOIN public.courses c ON e.course_code = c.course_code

  * Course + Faculty:
    public.courses c JOIN public.faculty f ON c.faculty_code = f.faculty_code

  * Fee + Student:
    public.fees f JOIN public.students s ON f.roll_no = s.roll_no

  * Fee Payment + Fee:
    public.fee_payments fp JOIN public.fees f ON fp.fee_id = f.id
    JOIN public.students s ON fp.roll_no = s.roll_no

KEY NOTES:
- students.status: 'Active', 'Graduated', 'Dropped'
- enrollments.status: 'Enrolled', 'Completed', 'Dropped', 'Failed'
- enrollments.grade: 'A+','A','B+','B','C','D','F' or NULL (ongoing)
- fees.status: 'Pending', 'Paid', 'Overdue', 'Waived'
- fees.fee_type: 'Tuition', 'Hostel', 'Library', 'Lab', 'Exam'
- fee_payments.payment_method: 'Cash', 'Card', 'UPI', 'Bank Transfer'
- faculty.designation: 'Professor', 'Associate Prof', 'Assistant Prof', 'Lecturer'
- courses.course_type: 'Core', 'Elective', 'Lab'

RULES:
- Only SELECT or WITH queries
- Never INSERT/UPDATE/DELETE/DROP
- Return only SQL as plain text, no markdown, no "SQL:" prefix
- Use proper aliases (s, d, e, c, f, fp, fac)
- Detail queries: LIMIT 100. Aggregated: no LIMIT
- Use PostgreSQL syntax only
- Current year: {current_year}

FEW-SHOT EXAMPLES:

User: How many students are there?
SQL: SELECT COUNT(*) AS total_students FROM public.students WHERE status = 'Active';

User: Show all students
SQL: SELECT roll_no, name, dept_code, admission_year, semester, cgpa, city, status FROM public.students ORDER BY roll_no;

User: Students in CSE department
SQL: SELECT s.roll_no, s.name, s.semester, s.cgpa FROM public.students s WHERE s.dept_code = 'CSE' AND s.status = 'Active' ORDER BY s.cgpa DESC;

User: Top 5 students by CGPA
SQL: SELECT s.roll_no, s.name, d.name AS department, s.cgpa FROM public.students s JOIN public.departments d ON s.dept_code = d.dept_code WHERE s.status = 'Active' AND s.cgpa > 0 ORDER BY s.cgpa DESC LIMIT 5;

User: Department-wise student count
SQL: SELECT d.dept_code, d.name AS department, COUNT(*) AS student_count FROM public.students s JOIN public.departments d ON s.dept_code = d.dept_code WHERE s.status = 'Active' GROUP BY d.dept_code, d.name ORDER BY student_count DESC;

User: Students with low attendance
SQL: SELECT s.roll_no, s.name, c.name AS course, e.attendance_pct FROM public.enrollments e JOIN public.students s ON e.roll_no = s.roll_no JOIN public.courses c ON e.course_code = c.course_code WHERE e.attendance_pct < 75 AND e.academic_year = {current_year} ORDER BY e.attendance_pct;

User: Show all courses
SQL: SELECT c.course_code, c.name, d.name AS department, f.name AS faculty, c.credits, c.course_type FROM public.courses c JOIN public.departments d ON c.dept_code = d.dept_code JOIN public.faculty f ON c.faculty_code = f.faculty_code ORDER BY c.dept_code, c.course_code;

User: Who teaches Machine Learning?
SQL: SELECT c.course_code, c.name AS course, f.name AS faculty, f.designation, d.name AS department FROM public.courses c JOIN public.faculty f ON c.faculty_code = f.faculty_code JOIN public.departments d ON c.dept_code = d.dept_code WHERE LOWER(c.name) LIKE '%machine learning%';

User: Show all faculty
SQL: SELECT f.faculty_code, f.name, f.designation, d.name AS department, f.salary FROM public.faculty f JOIN public.departments d ON f.dept_code = d.dept_code WHERE f.status = 'Active' ORDER BY f.salary DESC;

User: Faculty salary by department
SQL: SELECT d.name AS department, COUNT(*) AS faculty_count, AVG(f.salary) AS avg_salary, SUM(f.salary) AS total_salary FROM public.faculty f JOIN public.departments d ON f.dept_code = d.dept_code WHERE f.status = 'Active' GROUP BY d.name ORDER BY total_salary DESC;

User: Show grade distribution for CS301
SQL: SELECT e.grade, COUNT(*) AS count FROM public.enrollments e WHERE e.course_code = 'CS301' AND e.grade IS NOT NULL GROUP BY e.grade ORDER BY e.grade;

User: Students who failed
SQL: SELECT s.roll_no, s.name, c.name AS course, e.grade, e.academic_year FROM public.enrollments e JOIN public.students s ON e.roll_no = s.roll_no JOIN public.courses c ON e.course_code = c.course_code WHERE e.grade = 'F' ORDER BY e.academic_year DESC;

User: Show pending fees
SQL: SELECT f.id, s.roll_no, s.name, d.name AS department, f.fee_type, f.amount, f.due_date, f.status FROM public.fees f JOIN public.students s ON f.roll_no = s.roll_no JOIN public.departments d ON s.dept_code = d.dept_code WHERE f.status IN ('Pending', 'Overdue') ORDER BY f.due_date;

User: Total fee collection this year
SQL: SELECT SUM(fp.amount) AS total_collected FROM public.fee_payments fp WHERE EXTRACT(YEAR FROM fp.payment_date) = EXTRACT(YEAR FROM CURRENT_DATE);

User: Fee collection by department this year
SQL: SELECT d.name AS department, SUM(fp.amount) AS total_collected FROM public.fee_payments fp JOIN public.students s ON fp.roll_no = s.roll_no JOIN public.departments d ON s.dept_code = d.dept_code WHERE EXTRACT(YEAR FROM fp.payment_date) = EXTRACT(YEAR FROM CURRENT_DATE) GROUP BY d.name ORDER BY total_collected DESC;

User: Fee collection by payment method
SQL: SELECT fp.payment_method, COUNT(*) AS transactions, SUM(fp.amount) AS total FROM public.fee_payments fp WHERE EXTRACT(YEAR FROM fp.payment_date) = EXTRACT(YEAR FROM CURRENT_DATE) GROUP BY fp.payment_method ORDER BY total DESC;

User: Which students have overdue fees?
SQL: SELECT s.roll_no, s.name, d.name AS department, f.fee_type, f.amount, f.due_date FROM public.fees f JOIN public.students s ON f.roll_no = s.roll_no JOIN public.departments d ON s.dept_code = d.dept_code WHERE f.status = 'Overdue' ORDER BY f.due_date;

User: How many students enrolled this year?
SQL: SELECT COUNT(DISTINCT e.roll_no) AS students_enrolled FROM public.enrollments e WHERE e.academic_year = EXTRACT(YEAR FROM CURRENT_DATE);

User: Current enrollments
SQL: SELECT s.roll_no, s.name, c.course_code, c.name AS course, e.attendance_pct FROM public.enrollments e JOIN public.students s ON e.roll_no = s.roll_no JOIN public.courses c ON e.course_code = c.course_code WHERE e.status = 'Enrolled' AND e.academic_year = EXTRACT(YEAR FROM CURRENT_DATE) ORDER BY c.course_code;

User: Show departments
SQL: SELECT dept_code, name, head_of_dept, building, established_year FROM public.departments WHERE status = 'Active' ORDER BY dept_code;

User: Students from Maharashtra
SQL: SELECT roll_no, name, dept_code, city, cgpa FROM public.students WHERE state = 'Maharashtra' AND status = 'Active' ORDER BY cgpa DESC;

User: Batch-wise student count
SQL: SELECT admission_year AS batch, COUNT(*) AS students FROM public.students WHERE status = 'Active' GROUP BY admission_year ORDER BY admission_year;

User: Yearly fee collection trend
SQL: SELECT EXTRACT(YEAR FROM fp.payment_date) AS year, SUM(fp.amount) AS total_collected FROM public.fee_payments fp GROUP BY EXTRACT(YEAR FROM fp.payment_date) ORDER BY year;

User: Dropped students
SQL: SELECT roll_no, name, dept_code, admission_year, cgpa FROM public.students WHERE status = 'Dropped';

User: Average CGPA by department
SQL: SELECT d.name AS department, ROUND(AVG(s.cgpa), 2) AS avg_cgpa FROM public.students s JOIN public.departments d ON s.dept_code = d.dept_code WHERE s.status = 'Active' AND s.cgpa > 0 GROUP BY d.name ORDER BY avg_cgpa DESC;

Recent context: {context_text}
User Question: {user_question}
Previous Error: {last_error}

Generate response now:"""

    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1500)
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            temp_query = response.content.strip()
        else:
            temp_query = str(response).strip()

        temp_query = re.sub(r'^```\w*\s*|\s*```$', '', temp_query)
        temp_query = re.sub(r'^\s*(?:sql|SQL):\s*', '', temp_query, flags=re.IGNORECASE)
        temp_query = re.sub(r'^\s*(?:sql|SQL)\s*', '', temp_query, flags=re.IGNORECASE)
        temp_query = temp_query.strip()

        query = temp_query.split(" response_metadata")[0].strip()
        query = query.replace("\\n", "\n")
        print(f"[DEBUG] Generated SQL: {query[:200]}")
        return query
    except Exception as e:
        print("LLM generation error:", str(e))
        return ""

# ------------------------
# Main Query Runner
# ------------------------

def run_langchain_query(user_question: str, max_retries: int = 3, context_messages: list = None, is_forecast: bool = False):
    print(f"[DEBUG] Received: '{user_question}'")
    try:
        schema_map = load_schema_cache()
        if not schema_map:
            return {"error": "Failed to load database schema"}

        last_error = ""
        query = None

        for attempt in range(max_retries):
            print(f"Attempt {attempt + 1}/{max_retries}")

            question_lower = user_question.lower()
            if (re.search(r'\btable\b', question_lower) or
                re.search(r'\bcolumn\b', question_lower) or
                re.search(r'\bschema\b', question_lower)):
                return {"query": None, "result": list(schema_map.keys())[:7]}

            query = generate_sql_with_llm(user_question, schema_map, last_error, context_messages, is_forecast=is_forecast)

            if not query:
                return {"error": "No query generated by LLM"}

            if "CONVERSATIONAL_RESPONSE" in query:
                conv_text = query.replace("CONVERSATIONAL_RESPONSE:", "").strip()
                return {"query": None, "result": [{"conversational_response": conv_text}]}

            if isinstance(query, str) and query.startswith("-- Error"):
                return {"error": query}

            try:
                validate_sql_query(query)
            except Exception as e:
                return {"error": f"Unsafe query blocked: {str(e)}"}

            result = execute_sql_query(query)

            if result and isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and "error" in result[0]:
                last_error = result[0]["error"]
                print(f"Database error: {last_error}")
                if any(kw in last_error.lower() for kw in ["does not exist", "relation", "invalid column"]):
                    schema_map = build_schema_cache()
                    continue
            else:
                return {"query": query, "result": result}

        return {"error": f"Failed after {max_retries} attempts. Last error: {last_error}", "last_query": query}

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}