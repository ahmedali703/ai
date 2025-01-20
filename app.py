from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import redis
from flask_cors import CORS
import oracledb
import os
import re
import uuid
import schedule
import time
from sqlalchemy import create_engine, MetaData, Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator
from openai import OpenAI  # استيراد العميل من الإصدار الجديد
from dotenv import load_dotenv
import sys
import io
import threading
import numpy as np
import json
import traceback
from werkzeug.security import generate_password_hash, check_password_hash
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import subprocess
import secrets


# تحميل متغيرات البيئة من ملف .env (إن وجد)
load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials=True)  


# scheduler = BackgroundScheduler()
# scheduler.start()


# إعداد مفتاح سري للجلسات
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))  # مفتاح آمن

# إعداد Flask-Session لاستخدام Redis
app.config["SESSION_TYPE"] = "redis"
app.config["SESSION_REDIS"] = redis.from_url("redis://localhost:6379") 
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True

Session(app)

# ===== إعداد اتصال أوراكل =====
os.environ["NLS_LANG"] = "AMERICAN_AMERICA.AL32UTF8"
os.environ["TNS_ADMIN"] = r"C:\app\Mopa\product\21c\dbhomeXE\instantclient"

try:
    connection = oracledb.connect(
        user=os.getenv("ORACLE_USER", "HR"),
        password=os.getenv("ORACLE_PASSWORD", "HR"),
        dsn=os.getenv("ORACLE_DSN", "localhost/xepdb1")  # تأكد من صحة الـ SID أو Service Name
    )
    print("Successfully connected to Oracle database.")
except Exception as e:
    print(f"Error connecting to Oracle database: {e}")
    connection = None

# SQLAlchemy
# استخدام الـ Service Name في سلسلة الاتصال
engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URI", "oracle+oracledb://HR:HR@localhost/?service_name=xepdb1"))
SessionLocal = sessionmaker(bind=engine)
metadata = MetaData()
Base = declarative_base()

# نموذج المستخدم
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)

    # تعريف نموذج المحادثة
class Conversation(Base):
    __tablename__ = 'conversations'
    conversation_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    question = Column(Text, nullable=False)  # يمكنك استخدام Text أو CLOB للبيانات الكبيرة
    response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    feedback = Column(Text)  # حقل نصي لتخزين التعليقات
    classification = Column(String(100))  # الحقل الجديد لتخزين التصنيف

    user = relationship("User", backref="conversations")

# تعريف نموذج التغذية الراجعة
class Feedback(Base):
    __tablename__ = 'feedback'
    feedback_id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey('conversations.conversation_id'), nullable=False)
    rating_type = Column(Integer)  # تقييم رقمي مثلًا من 1 إلى 5
    comments = Column(Text)  # حقل نصي للتعليقات

    conversation = relationship("Conversation", backref="feedbacks")


# إنشاء الجداول إذا لم تكن موجودة
Base.metadata.create_all(bind=engine)

# إعداد OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # تهيئة العميل الجديد

smtplib.debuglevel = 1


# ====================================================================
# الدوال المساعدة
# ====================================================================

def save_conversation(user_id, question, response, classification, feedback=None):
    try:
        db_session = SessionLocal()
        new_conversation = Conversation(
            user_id=user_id,
            question=question,
            response=response,
            feedback=feedback,
            classification=classification# حفظ التصنيف
        )
        db_session.add(new_conversation)
        db_session.commit()
        db_session.refresh(new_conversation)
        db_session.close()
        return new_conversation.conversation_id
    except Exception as e:
        print(f"Error saving conversation: {e}")
        if db_session:
            db_session.rollback()
            db_session.close()
        return None


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def fetch_all_data():
    db_session = SessionLocal()
    # استخراج المحادثات
    conversations = db_session.query(Conversation).all()
    # استخراج التغذية الراجعة
    feedbacks = db_session.query(Feedback).all()
    db_session.close()

    # تحويل المحادثات إلى قائمة قواميس
    conv_data = [{
        "conversation_id": conv.conversation_id,
        "user_id": conv.user_id,
        "question": conv.question,
        "response": conv.response,
        "timestamp": conv.timestamp,
        "feedback": conv.feedback
    } for conv in conversations]

    # تحويل التغذية الراجعة إلى قائمة قواميس
    feedback_data = [{
        "feedback_id": fb.feedback_id,
        "conversation_id": fb.conversation_id,
        "rating_type": fb.rating_type,
        "comments": fb.comments
    } for fb in feedbacks]

    return conv_data, feedback_data

def analyze_question_patterns():
    conv_data, _ = fetch_all_data()
    df_conversations = pd.DataFrame(conv_data)

    # تجهيز نص الأسئلة وتحويله إلى متجهات
    vectorizer = TfidfVectorizer(stop_words='arabic')
    X = vectorizer.fit_transform(df_conversations['question'].astype(str))

    # تطبيق KMeans للتجميع
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    df_conversations['cluster'] = kmeans.labels_
    pattern_counts = df_conversations['cluster'].value_counts()
    print("أنماط الأسئلة الأكثر شيوعاً:\n", pattern_counts)


# دالة إرسال البريد الإلكتروني بصيغة HTML
def send_email_html(to_addresses, subject, html_body, smtp_server="apexexperts.net", smtp_port=465, username="ai@apexexperts.net", password="Ahmed@_240615"):
    msg = MIMEText(html_body, "html", "utf-8")
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = username
    # إذا كانت القائمة، قم بتجميعها لرأس الرسالة
    if isinstance(to_addresses, list):
        msg["To"] = ", ".join(to_addresses)
    else:
        msg["To"] = to_addresses
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(username, password)
        server.sendmail(username, to_addresses if isinstance(to_addresses, list) else [to_addresses], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def send_weekly_report():
    subject = "التقرير الأسبوعي"
    body = "<h1>التقرير الأسبوعي</h1><p>هذا هو التقرير الأسبوعي.</p>"
    to_email = "ahmed-alsaied@msn.com"
    send_email_html(to_email, subject, body)

def send_weekly_report_with_chart():
    subject = "التقرير الأسبوعي مع الرسم البياني"
    body = "<h1>التقرير الأسبوعي</h1><p>هذا هو التقرير الأسبوعي.</p>"
    to_email = "recipient@example.com"
    send_email_html(to_email, subject, body)

def dicts_to_html_table(data):
    if not data:
        return "<p>لا توجد بيانات.</p>"
    df = pd.DataFrame(data)
    # تحويل DataFrame إلى HTML مع تنسيقات بسيطة باستخدام Tailwind CSS
    return df.to_html(classes="min-w-full bg-white border border-gray-200 rounded-lg text-center", index=False)


def get_all_table_schemas():
    if not connection:
        print("No database connection.")
        return ""
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
                FROM ALL_TAB_COLUMNS
                WHERE OWNER = 'HR'
                ORDER BY TABLE_NAME, COLUMN_ID
            """)
            rows = cursor.fetchall()
        table_schemas = {}
        for row in rows:
            table, column, data_type = row
            if table not in table_schemas:
                table_schemas[table] = []
            table_schemas[table].append(f"{column} ({data_type})")
        schema_summary = ""
        for table, columns in table_schemas.items():
            schema_summary += f"\nTable {table} has the following columns:\n"
            for col in columns:
                schema_summary += f"- {col}\n"
        return schema_summary
    except Exception as e:
        print(f"Error fetching table schemas: {e}")
        return ""

def classify_question(question):
    """
    يصنف السؤال إلى أحد الأقسام الأربعة:
    - 'db_sql'    : استعلام SQL عادي (SELECT)
    - 'db_analysis': تحليل متقدم/إحصائي
    - 'db_action' : أوامر تعديل (INSERT / UPDATE / DELETE)
    - 'general'   : سؤال عام
    """

    question_lower = question.lower().strip()

    # ===== مفاتيح إجراء التعديل (Action) =====
    action_keywords = [
        "insert", "update", "delete", "remove", "add", "create record", "create table",
        "send mail", "create restful services",
        "drop", "truncate", "انشاء", "إضافة", "حذف", "تحديث", 
        "تعديل", "add column", "rename"
    ]

    # ===== مفاتيح تحليلية (Analysis) =====
    analysis_keywords = [
        "تحليل", "احصاء", "إحصاء", "إحصائي",
        "analysis", "statistic", "statistics", "describe",
        "distribution", "mean", "histogram", "correlation",
        "boxplot", "انحراف معياري", "variance", "standard deviation",
        "summarize", "summary", "trend", "forecast", "regression",
        "analyze", "aggregate", "pivot"
    ]

    # ===== مفاتيح SQL (Reading) =====
    db_sql_keywords = [
        "select", "show me", "fetch", "retrieve",
        "استعلام", "query", "بيانات الجدول", "قائمة الأعمدة",
        "sum", "مجموع", "اجمالي", "إجمالي", "متوسط",
        "maximum", "minimum", "order by", "limit",
        "عرض", "اظهار", "columns"
    ]

    # ===== مفاتيح أسئلة عامة (General) =====
    general_keywords = [
        "what is", "explain", "why", "how to", "كيفية",
        "عام", "معلومة", "what are", "متى", "أين", "لماذا"
    ]

    # 1) لو رصدنا action_keywords
    for ak in action_keywords:
        if ak in question_lower:
            print("Local classification => db_action")
            return "db_action"

    # 2) لو رصدنا analysis
    for kw in analysis_keywords:
        if kw in question_lower:
            print("Local classification => db_analysis")
            return "db_analysis"

    # 3) لو رصدنا db_sql
    for kw in db_sql_keywords:
        if kw in question_lower:
            print("Local classification => db_sql")
            return "db_sql"

    # 4) لو رصدنا general_keywords
    for kw in general_keywords:
        if kw in question_lower:
            print("Local classification => general")
            return "general"

    # 5) إذا لم نعثر على تصنيف واضح محليًا => نلجأ لـGPT
    try:
        schema_summary = get_all_table_schemas()  # يفترض أنها دالة لديك تعيد وصف الجداول
        prompt = f"""
You are a classification assistant. You have the following database schema:

{schema_summary}

You must classify any user question into exactly one category:
1) db_sql: direct query about the database, e.g. SELECT, retrieving rows/columns directly, "Show me employees"
2) db_analysis: advanced analysis or statistics on the data, e.g. "Calculate average salary or distribution"
3) db_action: modifying the database, e.g. "INSERT, UPDATE, DELETE, DROP, CREATE TABLE"
4) general: if not related to the HR database or no direct data/analysis request.

Important rules:
- If user question includes "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", or similar, classify as db_action.
- If user question includes "mean, distribution, correlation, statistic", or advanced analysis terms, classify as db_analysis.
- If user question includes "select, show me, fetch, retrieve" from a table, or direct request of rows, classify as db_sql.
- If user question is unrelated to the HR schema, or is general knowledge, classify as general.
- Use EXACT category name: db_sql, db_analysis, db_action, or general. No extra words.

Examples:
1) "Add new column to EMPLOYEES" => db_action
2) "Compute average salary of employees" => db_analysis
3) "SELECT * from employees" => db_sql
4) "What is AI?" => general

Now, classify the user question below:

Question: "{question}"
Answer with exactly one of: db_sql, db_analysis, db_action, or general.
"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0
        )
        classification = response.choices[0].message.content.strip().lower()
        print(f"GPT classification => {classification}")

        if 'db_sql' in classification:
            return 'db_sql'
        elif 'db_analysis' in classification:
            return 'db_analysis'
        elif 'db_action' in classification:
            return 'db_action'
        else:
            return 'general'
    except Exception as e:
        print(f"Error classifying question with GPT: {e}")
        return 'general'

def translate_ar_to_en(text):
    try:
        print("Translating:", text)
        translated = GoogleTranslator(source='ar', target='en').translate(text)
        print("Translated:", translated)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def clean_sql_query(sql_query):
    sql_query = sql_query.strip()
    sql_query = re.sub(r';$', '', sql_query)
    sql_query = re.sub(r'[؛؛]', '', sql_query)
    sql_query = sql_query.replace('\u200f', '')
    return sql_query

def natural_language_to_sql(question, is_chart=False):
    """
    يحول أسئلة تتعلق بعرض البيانات (SELECT) إلى جملة SQL.
    """
    try:
        question_en = translate_ar_to_en(question)
        schema_summary = get_all_table_schemas()

        if is_chart:
            prompt = f"""
Translate the following question into a SQL query that returns data suitable for a bar chart with two columns: 'label' and 'value'. Provide only the SQL code without any explanations or additional text. Do not include a semicolon at the end.

Here is the schema of the database:
{schema_summary}

Question: '{question_en}'
"""
        else:
            prompt = f"""
Translate the following question into a SQL query. Provide only the SQL code without any explanations or additional text. Do not include a semicolon at the end.

Here is the schema of the database:
{schema_summary}

Question: '{question_en}'
"""

        print(f"Sending prompt to OpenAI: {prompt}")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates natural language questions into SQL queries compatible with Oracle Database."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0
        )
        sql_query = response.choices[0].message.content.strip()
        print(f"Generated SQL Query: {sql_query}")
        return clean_sql_query(sql_query)
    except Exception as e:
        print(f"Error generating SQL query: {e}")
        return None

def natural_language_to_dml_action(question):
    """
    يحوّل الأسئلة المتعلقة بالتعديل (INSERT/UPDATE/DELETE)
    إلى جملة SQL خاصة بالأكشن.
    """
    try:
        question_en = translate_ar_to_en(question)
        schema_summary = get_all_table_schemas()

        prompt = f"""
You are an Oracle database expert with the ability to execute any complex queries, as well as add, modify, and delete data. You have complete knowledge of all tables and data in the database and can provide effective solutions to any database-related issues. Your tasks include:

Executing complex SQL queries:

Writing queries to extract required data from tables.

Using JOIN, GROUP BY, HAVING, SUBQUERIES, and other advanced operations.

Optimizing queries to ensure optimal performance.

Managing data:

Adding new data to tables using INSERT.

Updating existing data using UPDATE.

Deleting data using DELETE.

Analyzing data:

Analyzing relationships between tables and understanding the database structure.

Identifying data issues and providing solutions to fix them.

Creating and modifying tables:

Creating new tables using CREATE TABLE.

Modifying table structures using ALTER TABLE.

Dropping tables using DROP TABLE.

Managing indexes:

Creating indexes to improve query performance.

Analyzing existing indexes and determining the need for modifications.

Providing reports and results:

Delivering clear and organized results for queries.

Generating reports summarizing the required data.

Troubleshooting:

Analyzing errors in queries and fixing them.

Providing solutions for any issues related to data integrity or performance.

You are familiar with all tables, columns, and relationships between them, and you can provide accurate and effective answers to any questions or requests related to Oracle databases.

Database schema:
{schema_summary}

User request: '{question_en}'
"""

        print(f"Sending 'action' prompt to OpenAI:\n{prompt}")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that writes Oracle-compatible DML statements."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0
        )
        action_sql = response.choices[0].message.content.strip()
        print(f"Generated DML SQL: {action_sql}")
        return clean_sql_query(action_sql)
    except Exception as e:
        print(f"Error generating DML action: {e}")
        return None

def execute_sql_query(sql_query):
    """
    ينفّذ استعلام SELECT على قاعدة البيانات ويعيد DataFrame.
    """
    if not connection:
        print("No database connection.")
        return None
    try:
        with connection.cursor() as cursor:
            print(f"Executing SQL Query: {sql_query}")
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
            df = df.fillna(0)  # ملء القيم الفارغة
            # توحيد أسماء الأعمدة
            new_columns = []
            for col in df.columns:
                clean_col = col.replace(" ", "_").upper()
                new_columns.append(clean_col)
            df.columns = new_columns
            print(f"New Columns after cleaning: {df.columns.tolist()}")
        print(f"Query Results: {df}")
        return df
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return None

def execute_sql_action(sql_statement):
    """
    ينفّذ (INSERT / UPDATE / DELETE) على قاعدة البيانات ويعمل commit.
    يعيد (success, message, rows_affected).
    """
    if not connection:
        print("No database connection.")
        return False, "No DB connection", 0
    try:
        with connection.cursor() as cursor:
            print(f"Executing SQL Action: {sql_statement}")
            cursor.execute(sql_statement)
            rows_affected = cursor.rowcount  # عدد الصفوف المتأثرة
            connection.commit()
        return True, "تم تنفيذ العملية بنجاح", rows_affected
    except Exception as e:
        print(f"Error executing SQL action: {e}")
        return False, str(e), 0

def get_salary_summary():
    """
    مثال بسيط: استعلام يُعيد (MIN(SALARY), MAX(SALARY), AVG(SALARY), COUNT(*))
    يمكنك التعديل حسب ما تريد تلخيصه.
    """
    if not connection:
        return {}
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT MIN(SALARY), MAX(SALARY), AVG(SALARY), COUNT(*) FROM EMPLOYEES")
            row = cursor.fetchone()
            if row:
                min_sal, max_sal, avg_sal, cnt = row
                return {
                    "min_salary": float(min_sal or 0),
                    "max_salary": float(max_sal or 0),
                    "avg_salary": float(avg_sal or 0),
                    "total_employees": int(cnt or 0)
                }
    except:
        pass
    return {}

def generate_chart(data, x_key, y_key):
    plt.figure(figsize=(10, 6))
    plt.bar(data[x_key], data[y_key], color='skyblue')
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(f"Sum of {y_key} by {x_key}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    chart_path = f"./static/charts/{uuid.uuid4().hex}.png"
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path)
    plt.close()
    return f"./static/charts/{os.path.basename(chart_path)}"

def remove_markdown_fences(code_text):
    code_text = re.sub(r"```python\s*", "", code_text)
    code_text = re.sub(r"```", "", code_text)
    return code_text

def exec_python_code(code, df):
    code = remove_markdown_fences(code)

    old_stdout = sys.stdout
    mystdout = io.StringIO()
    sys.stdout = mystdout

    local_env = {
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt
    }

    try:
        exec(code, local_env)
    except Exception as e:
        sys.stdout = old_stdout
        error_output = mystdout.getvalue()
        tb = traceback.format_exc()
        error_output += f"\n\nحدث استثناء:\n{str(e)}\nTraceback:\n{tb}"
        return f"خطأ أثناء تنفيذ الكود:\n{error_output}", ""

    output = mystdout.getvalue()
    sys.stdout = old_stdout
    return "", output


def generate_response_with_context(question):
    conversation_history = session.get('conversation_history', [])
    context = " ".join([msg["content"] for msg in conversation_history])
    prompt = f"مع مراعاة السياق التالي: {context}\nالسؤال: {question}\nالإجابة:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,
        temperature=0
    )
    return response.choices[0].message.content.strip()


# ====================================================================
#             المسارات
# ====================================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register_user():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        if not username or not password:
            return jsonify({"error": "يرجى إدخال اسم المستخدم وكلمة المرور"}), 400

        db = SessionLocal()
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            return jsonify({"error": "اسم المستخدم موجود بالفعل"}), 400

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        db.close()

        # تسجيل الدخول تلقائي بعد التسجيل
        session['username'] = username
        session['session_id'] = str(uuid.uuid4())
        session['conversation_history'] = []
        session['memory'] = {}

        return jsonify({"message": f"تم تسجيل المستخدم {username} بنجاح!"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login_user():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        if not username or not password:
            return jsonify({"error": "يرجى إدخال اسم المستخدم وكلمة المرور"}), 400

        db = SessionLocal()
        user = db.query(User).filter(User.username == username).first()
        if not user or not check_password_hash(user.password, password):
            return jsonify({"error": "اسم المستخدم أو كلمة المرور غير صحيحة"}), 401

        db.close()

        # تسجيل الدخول وتعيين الجلسة
        session['username'] = username
        session['session_id'] = str(uuid.uuid4())
        session['conversation_history'] = []
        session['memory'] = {}

        return jsonify({"message": f"تم تسجيل الدخول بنجاح كـ {username}!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/logout', methods=['POST'])
def logout():
    try:
        session.clear()
        return jsonify({"message": "تم تسجيل الخروج بنجاح. تم مسح الـ Cookies"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        username = data.get('username')
        question = data.get('question')

        if not username:
            return jsonify({'error': 'Username is required.'}), 400
        if 'username' not in session or session['username'] != username:
            return jsonify({'error': 'User not logged in.'}), 401
        if not question:
            return jsonify({'error': 'No question provided'}), 400
            
            
        user = SessionLocal().query(User).filter(User.username == username).first()
        if not user:
            return jsonify({'error': 'User not found.'}), 404

        question_lower = question.lower()

        # تعيين قيمة افتراضية لـ `answer`
        answer = "حدث خطأ أثناء معالجة السؤال."

        # أسئلة مخصصة
        custom_responses = {
            "من قام بتطويرك؟": "تم تطويري بواسطة أحمد السعيد وفادي إسماعيل، وهما متخصصان في تقنيات الذكاء الاصطناعي ومعالجة البيانات. عملوا على تطويري لأكون أداة مفيدة تقدم المساعدة الذكية وتسهّل العمل في العديد من المجالات التقنية والإدارية.",
            "ما هو اسمك؟": "اسمي ديوان وانا  وكيل ذكي . تم اختياري لهذا الاسم لأنه يعكس قدراتي في تقديم المساعدة الذكية والإجابة على أسئلتك بطريقة تفيدك وتوفر الوقت والجهد.",
            "كيف تعمل؟": "أعمل باستخدام تقنيات متقدمة في الذكاء الاصطناعي، مثل معالجة اللغة الطبيعية والتعلم العميق. أتعلم من النصوص والبيانات الضخمة، مما يتيح لي فهم الأسئلة التي تطرحها وتقديم إجابات دقيقة ومفيدة.",
            "ما هي مهامك؟": "أنا هنا لأساعدك في العديد من المهام مثل تنفيذ استعلامات SQL، تحليل البيانات، كتابة الأكواد البرمجية، والإجابة عن أسئلتك العامة أو التقنية. بالإضافة إلى ذلك، يمكنني توفير الشروحات وتبسيط المفاهيم المعقدة.",
            "من هم مطوروك؟": "تم تطويري بواسطة أحمد السعيد وفادي إسماعيل. كلاهما متخصص في تقنيات البرمجة وتحليل البيانات، وقد عملا على إنشاء نظام متكامل يعتمد على الذكاء الاصطناعي ليكون مساعدًا فعالًا في مختلف المهام.",
            "هل يمكنك كتابة الأكواد البرمجية؟": "نعم، يمكنني مساعدتك في كتابة الأكواد بلغات برمجية متعددة مثل Python وSQL وJavaScript وغيرها. كما أستطيع مساعدتك في تصحيح الأخطاء وتحسين الأداء.",
            "ما الذي يجعلك مميزًا؟": "ما يجعلني مميزًا هو قدرتي على فهم السياق، تقديم إجابات مخصصة، وتنفيذ المهام بشكل دقيق وسريع. أهدف إلى تسهيل عملك وتقديم حلول مبتكرة.",
            "هل يمكنني الاعتماد عليك؟": "بالتأكيد! تم تصميمي خصيصًا لأكون أداة موثوقة. يمكنك الاعتماد علي في تنفيذ مهامك وتقديم المعلومات التي تحتاجها بدقة.",
            "ما هي فوائد الذكاء الاصطناعي؟": "الذكاء الاصطناعي يتيح أتمتة المهام المتكررة، تحسين الكفاءة، واستخراج رؤى قيمة من البيانات. يساهم في تطوير الأعمال، التعليم، والرعاية الصحية بطرق مبتكرة وفعالة.",
        }

        # تحقق إن كان السؤال مخصص
        for custom_question, custom_answer in custom_responses.items():
            if custom_question in question_lower:
                answer = custom_answer
                conversation_history = session.get('conversation_history', [])
                conversation_history.append({"role": "user", "content": question})
                conversation_history.append({"role": "assistant", "content": answer})
                session['conversation_history'] = conversation_history
                # حفظ المحادثة في قاعدة البيانات
                user = SessionLocal().query(User).filter(User.username == username).first()
                if user:
                    save_conversation(user.id, question, answer)

                return jsonify({
                    'results': [{'answer': answer}],
                    'classification': 'custom_response'
                })

        classification = classify_question(question)
        print(f"Question classification: {classification}")

        # معالجة خاصية إرسال البريد الإلكتروني
        if ("send mail" in question_lower or 
            "send an email" in question_lower or 
            "إرسال بريد" in question_lower or 
            "ارسل بريد" in question_lower):

            emails = []   

            # حالة: إرسال بريد لجميع الموظفين
            if "all employees" in question_lower or "كل الموظفين" in question_lower:
                query = "SELECT EMAIL FROM EMPLOYEES"
                df_emails = execute_sql_query(query)
                if df_emails is not None and not df_emails.empty:
                    emails = df_emails['EMAIL'].dropna().tolist()

            # حالة: إرسال بريد لكل من يعمل بوظيفة معينة
            elif "job" in question_lower or "وظيفة" in question_lower:
                job_match = re.search(r'وظيفة\s+(\w+)', question_lower)
                if job_match:
                    job_title = job_match.group(1)
                    query = f"SELECT EMAIL FROM EMPLOYEES WHERE LOWER(JOB_ID) LIKE '%{job_title.lower()}%'"
                    df_emails = execute_sql_query(query)
                    if df_emails is not None and not df_emails.empty:
                        emails = df_emails['EMAIL'].dropna().tolist()

            # حالة: إرسال بريد إلى موظف محدد بناءً على الاسم الأول أو الأخير
            else:
                name_match = re.search(r'الى\s+(\w+)', question_lower)
                recipient_name = name_match.group(1) if name_match else None

                if recipient_name:
                    query = f"""
                    SELECT EMAIL FROM EMPLOYEES 
                    WHERE LOWER(FIRST_NAME) = '{recipient_name.lower()}'
                    OR LOWER(LAST_NAME) = '{recipient_name.lower()}'
                    """
                    df_emails = execute_sql_query(query)
                    if df_emails is not None and not df_emails.empty:
                        emails = df_emails['EMAIL'].dropna().tolist()

            # التحقق من العثور على عناوين بريد إلكتروني
            if not emails:
                return jsonify({"error": "لا يمكن تحديد عناوين البريد الإلكتروني للمستلمين."}), 400

            # تنظيف عناوين البريد الإلكتروني للتأكد من صحتها
            emails = [email.strip() for email in emails if email.strip()]
            recipients_str = ", ".join(emails)

            conversation_history = session.get('conversation_history', [])
            last_assistant_msg = "لا توجد نتائج سابقة."
            last_assistant_data = None  # سنحاول استخراج بيانات من الرسائل السابقة

            if conversation_history:
                for msg in reversed(conversation_history):
                    if msg.get('role') == 'assistant' and msg.get('content'):
                        last_assistant_msg = msg['content']
                        break

            html_body = f"""
            <html>
            <body>
                <h2>مرسل من الوكيل الذكي 😊</h2>
                <p>مرحبًا،</p>
                <p>هذه هي النتيجة السابقة:</p>
                <pre>{last_assistant_msg}</pre>
            </body>
            </html>
            """
            subject = "نتيجة الاستعلام من الوكيل الذكي 😊"

            email_sent = send_email_html(emails, subject, html_body)
            if email_sent:
                # تحديث تاريخ المحادثة
                conversation_history.append({"role": "user", "content": question})
                conversation_history.append({"role": "assistant", "content": f"تم إرسال البريد الإلكتروني بنجاح إلى {recipients_str} 😊."})
                session['conversation_history'] = conversation_history

                return jsonify({
                    "results": [{"answer": f"تم إرسال البريد الإلكتروني بنجاح إلى {recipients_str} 😊."}],
                    "classification": "email"
                }), 200
            else:
                return jsonify({"error": "فشل في إرسال البريد."}), 500

        # =============== GENERAL ===============
        if classification == 'general':
            conversation_history = session.get('conversation_history', [])
            system_content = "You are a helpful assistant."
            if 'assistant_name' in session.get('memory', {}):
                assistant_name = session['memory']['assistant_name']
                system_content = f"You are {assistant_name}, a helpful assistant."

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_content},
                        *conversation_history,
                        {"role": "user", "content": question}
                    ],
                    max_tokens=3000,
                    temperature=0
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error in GPT response: {e}")
                answer = "حدث خطأ أثناء توليد الإجابة. يرجى المحاولة مرة أخرى."

            # حفظ المحادثة في قاعدة البيانات
            user = SessionLocal().query(User).filter(User.username == username).first()
            conv_id = None
            if user:
                conv_id = save_conversation(user.id, question, answer, classification='general')

            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": answer})
            session['conversation_history'] = conversation_history

            return jsonify({
                'results': [{'answer': answer}],
                'classification': 'general',
                'conversation_id': conv_id
            })


        # =============== DB_ANALYSIS ===============
        if classification == 'db_analysis':
            translated_en = translate_ar_to_en(question).lower()
            is_chart = any(word in translated_en for word in ["statistics", "chart", "graph", "plot"])
            base_sql = natural_language_to_sql(question, is_chart=is_chart)
            if not base_sql:
                return jsonify({'error': 'فشل في توليد استعلام SQL لجلب البيانات.'}), 500

            df = execute_sql_query(base_sql)
            if df is None:
                return jsonify({'error': 'فشل في تنفيذ الاستعلام الأساسي.'}), 500

            df = df.fillna(0)
            columns_list = list(df.columns)
            sample_records = df.head(5).to_dict(orient='records')

            analysis_prompt = f"""
أنت محلل بيانات في بايثون. لدينا DataFrame باسم df يحوي الأعمدة:
{columns_list}

عينة من البيانات:
{sample_records}

السؤال التحليلي: "{question}"

*** هام ***:
- لا تستخدم تنسيق سلسلة مثل % أو f-string على كائنات معقّدة.
- استخدم json.dumps() لإخراج النتائج في شكل JSON.
- **يجب** أن يكون الخرج النهائي في شكل:
  result = {{
    "labels": [...],   # أي عمود نصي أو اسمه department_id أو city أو ... 
    "values": [...]    # أي مجموع/أرقام أو اسمه total_salary أو salary ...
  }}
print(json.dumps(result))

(لا تستخدم مفاتيح أخرى مثل department_id أو total_salary.)

اكتب كود بايثون فقط (بدون علامات ```python) لإجراء المطلوب على df. استخدم print() لعرض النتائج.
"""
            try:
                python_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a Python data analyst with a DataFrame named df."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    max_tokens=3000,
                    temperature=0
                )
                python_code = python_response.choices[0].message.content.strip()
            except Exception as e:
                return jsonify({"error": f"خطأ في توليد كود التحليل: {str(e)}"}), 500

            print("GPT generated code:\n", python_code)
            err, output = exec_python_code(python_code, df)
            if err:
                conversation_history = session.get('conversation_history', [])
                conversation_history.append({"role": "user", "content": question})
                conversation_history.append({
                    "role": "assistant",
                    "content": "هذا كود التحليل ولكن حدث خطأ:\n" + python_code
                })
                session['conversation_history'] = conversation_history
                return jsonify({"error": err, "analysis_code": python_code}), 500
            else:
                chart_data = None
                try:
                    chart_data = json.loads(output)
                except json.JSONDecodeError:
                    chart_data = None

                conversation_history = session.get('conversation_history', [])
                conversation_history.append({"role": "user", "content": question})
                conversation_history.append({
                    "role": "assistant",
                    "content": "كود التحليل:\n" + python_code + "\n\nالمخرجات:\n" + output
                })
                session['conversation_history'] = conversation_history
                # حفظ المحادثة
                user = SessionLocal().query(User).filter(User.username == username).first()
                conv_id = None
                if user:
                    conv_id = save_conversation(user.id, question, answer, classification='db_analysis')
                    conversation_history.append({"role": "assistant", "content": answer})
                    session['conversation_history'] = conversation_history
                   

                return jsonify({
                    "analysis_code": python_code,
                    "analysis_output": output,
                    "chart_data": chart_data,
                    "classification": "db_analysis",
                    'conversation_id': conv_id
                })

        # =============== DB_ACTION (INSERT/UPDATE/DELETE) ===============
        elif classification == 'db_action':
            action_sql = natural_language_to_dml_action(question)
            if not action_sql:
                return jsonify({"error": "فشل في توليد SQL التعديل."}), 500

            # ===== معالجة حالة INSERT =====
            if action_sql.strip().lower().startswith("insert"):
                success, message, rows_affected = execute_sql_action(action_sql)
                if not success:
                    # استخدام ChatGPT لتفسير الخطأ
                    help_prompt = (
                        f"حدث خطأ في Oracle أثناء تنفيذ الأمر التالي:\n{action_sql}\n"
                        f"رسالة الخطأ: {message}\n"
                        f"يرجى شرح معنى الخطأ واقتراح حلول محتملة باللغة العربية."
                    )
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "أنت مساعد خبير في قاعدة بيانات أوراكل يرجى الرد باللغة العربية."},
                            {"role": "user", "content": help_prompt}
                        ],
                        max_tokens=3000,
                        temperature=0
                    )
                    error_explanation = response.choices[0].message.content.strip()
                    return jsonify({"error": error_explanation, "action_sql": action_sql}), 500

                # محاولة استخراج اسم الجدول من جملة INSERT
                table = None
                match = re.search(r"into\s+(\w+)", action_sql, re.IGNORECASE)
                if match:
                    table = match.group(1)

                new_data = None
                if table:
                    select_last_inserted = f"SELECT * FROM {table} WHERE ROWID IN (SELECT MAX(ROWID) FROM {table})"
                    df_new = execute_sql_query(select_last_inserted)
                    new_data = df_new.to_dict(orient='records') if df_new is not None else None

                new_table_html = dicts_to_html_table(new_data)

                final_text = (
                    f"تم تنفيذ الأمر:<br><pre>{action_sql}</pre><br>"
                    f"تأثر {rows_affected} صف.<br>"
                    f"النتيجة: {message}<br><br>"
                    f"البيانات المضافة:<br>{new_table_html}"
                )

                conversation_history = session.get('conversation_history', [])
                conversation_history.append({"role": "user", "content": question})
                conversation_history.append({"role": "assistant", "content": final_text})
                session['conversation_history'] = conversation_history
                answer = "تم الانتهاء من المعالجة."  # قيمة افتراضية
                # حفظ المحادثة
                user = SessionLocal().query(User).filter(User.username == username).first()
                if user:
                    conv_id = save_conversation(user.id, question, answer, classification='db_action')
                    conversation_history.append({"role": "user", "content": question})
                    conversation_history.append({"role": "assistant", "content": answer})
                    session['conversation_history'] = conversation_history
                    

                return jsonify({
                    "action_sql": action_sql,
                    "rows_affected": rows_affected,
                    "message": message,
                    "final_text": final_text,
                    "classification": "db_action",
                    'conversation_id': conv_id
                })

            # ===== معالجة UPDATE وDELETE =====
            else:
                before_query = None
                after_query = None
                if action_sql.strip().lower().startswith("update"):
                    match = re.search(r"update\s+(\w+)\s+set\s+(.*)\s+where\s+(.*)", action_sql, re.IGNORECASE)
                    if match:
                        table = match.group(1)
                        condition = match.group(3)
                        before_query = f"SELECT * FROM {table} WHERE {condition}"
                        after_query = before_query

                before_data = None
                if before_query:
                    df_before = execute_sql_query(before_query)
                    before_data = df_before.to_dict(orient='records') if df_before is not None else None

                success, message, rows_affected = execute_sql_action(action_sql)
                if not success:
                    # استخدام ChatGPT لتفسير الخطأ في UPDATE/DELETE
                    help_prompt = (
                        f"حدث خطأ في Oracle أثناء تنفيذ الأمر التالي:\n{action_sql}\n"
                        f"رسالة الخطأ: {message}\n"
                        f"يرجى شرح معنى الخطأ واقتراح حلول محتملة باللغة العربية."
                    )
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "أنت مساعد خبير في قاعدة بيانات أوراكل يرجى الرد باللغة العربية."},
                            {"role": "user", "content": help_prompt}
                        ],
                        max_tokens=3000,
                        temperature=0
                    )
                    error_explanation = response.choices[0].message.content.strip()
                    return jsonify({"error": error_explanation, "action_sql": action_sql}), 500

                after_data = None
                if after_query:
                    df_after = execute_sql_query(after_query)
                    after_data = df_after.to_dict(orient='records') if df_after is not None else None

                before_table_html = dicts_to_html_table(before_data)
                after_table_html = dicts_to_html_table(after_data)

                final_text = (
                    f"تم تنفيذ الأمر:<br><pre>{action_sql}</pre><br>"
                    f"تأثر {rows_affected} صف.<br>"
                    f"النتيجة: {message}<br><br>"
                    f"البيانات قبل التغيير:<br>{before_table_html}<br><br>"
                    f"البيانات بعد التغيير:<br>{after_table_html}"
                )

                conversation_history = session.get('conversation_history', [])
                conversation_history.append({"role": "user", "content": question})
                conversation_history.append({"role": "assistant", "content": final_text})
                session['conversation_history'] = conversation_history

                return jsonify({
                    "action_sql": action_sql,
                    "rows_affected": rows_affected,
                    "message": message,
                    "final_text": final_text,
                    "classification": "db_action"
                })

        # =============== DB_SQL (SELECT) ===============
        elif classification == 'db_sql':
            translated_question = translate_ar_to_en(question).lower()
            is_chart = any(word in translated_question for word in ["statistics", "chart", "graph", "plot"])

            sql_query = natural_language_to_sql(question, is_chart=is_chart)
            if not sql_query:
                return jsonify({'error': 'Failed to generate SQL query'}), 500

            df = execute_sql_query(sql_query)
            if df is None:
                return jsonify({'error': 'Failed to execute SQL query'}), 500

            df = df.fillna(0)
            df_records = df.to_dict(orient='records')

            if is_chart:
                df = df.rename(columns={
                    'JOB_TITLE': 'label',
                    'NUM_EMPLOYEES': 'value'
                })
                if 'label' in df.columns and 'value' in df.columns:
                    df_subset = df.head(5).to_dict(orient='records')
                    chart_path = generate_chart(df, "label", "value")

                    analysis_prompt = f"""
        السؤال: "{question}"
        لقد استخرجنا {len(df)} سجلاً.
        هذه عينة من أول {len(df_subset)} سجل:
        {df_subset}

        نتائج الاستعلام (للرسم):
        {df_records}

        اكتب شرحاً مفصلاً حول هذه النتائج باللغة العربية.
        """
                    try:
                        analysis_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful data analyst that uses the provided database result to respond in Arabic."},
                                {"role": "user", "content": analysis_prompt}
                            ],
                            max_tokens=3000,
                            temperature=0
                        )
                        assistant_answer = analysis_response.choices[0].message.content.strip()
                    except OpenAI.OpenAIError:
                        assistant_answer = "حدث خطأ أثناء توليد شرح النتائج."

                    conversation_history = session.get('conversation_history', [])
                    conversation_history.append({"role": "user", "content": question})
                    conversation_history.append({
                        "role": "assistant",
                        "content": assistant_answer
                    })
                    session['conversation_history'] = conversation_history

                    # حفظ المحادثة
                    user = SessionLocal().query(User).filter(User.username == username).first()
                    if user:
                        save_conversation(user.id, question, assistant_answer, classification='db_sql')

                    return jsonify({
                        'results': df_records,
                        'chart_path': chart_path,
                        'assistant_answer': assistant_answer,
                        'classification': 'db_sql'
                    })
                else:
                    return jsonify({'error': 'Invalid data for chart'}), 500
            else:
                analysis_prompt = f"""
        السؤال: "{question}"
        هذه نتائج الاستعلام:
        {df_records}

        اكتب شرحاً مفصلاً للنتائج باللغة العربية.
        """
                try:
                    analysis_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful data analyst that uses the provided database result."},
                            {"role": "user", "content": analysis_prompt}
                        ],
                        max_tokens=3000,
                        temperature=0
                    )
                    assistant_answer = analysis_response.choices[0].message.content.strip()
                except OpenAI.APIError:
                    assistant_answer = "حدث خطأ أثناء توليد الشرح."

                conversation_history = session.get('conversation_history', [])
                conversation_history.append({"role": "user", "content": question})
                conversation_history.append({
                    "role": "assistant",
                    "content": assistant_answer
                })
                session['conversation_history'] = conversation_history

                # حفظ المحادثة
                user = SessionLocal().query(User).filter(User.username == username).first()
                if user:
                    conv_id = save_conversation(user.id, question, answer, classification='db_sql')
                    conversation_history.append({"role": "user", "content": question})
                    conversation_history.append({"role": "assistant", "content": answer})
                    session['conversation_history'] = conversation_history
                    

                return jsonify({
                    'results': df_records,
                    'assistant_answer': assistant_answer,
                    'classification': 'db_sql',
                    'conversation_id': conv_id
                })

        # =============== DEFAULT ===============
        else:
            # سؤال عام
            conversation_history = session.get('conversation_history', [])
            if re.match(r"انت\s+اسمك\s+(.+)", question, re.IGNORECASE):
                match = re.match(r"انت\s+اسمك\s+(.+)", question, re.IGNORECASE)
                if match:
                    assistant_name = match.group(1).strip()
                    session['memory']['assistant_name'] = assistant_name
                    conversation_history.append({"role": "user", "content": question})
                    conversation_history.append({"role": "assistant", "content": f"تم تعيين اسمي إلى {assistant_name}."})
                    session['conversation_history'] = conversation_history
                    return jsonify({
                        'results': [{'answer': f"تم تعيين اسمي إلى {assistant_name}."}],
                        'classification': 'general'
                    })

            system_content = "You are a helpful assistant."
            if 'assistant_name' in session.get('memory', {}):
                assistant_name = session['memory']['assistant_name']
                system_content = f"You are {assistant_name}, a helpful assistant."

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_content},
                        *conversation_history,
                        {"role": "user", "content": question}
                    ],
                    max_tokens=3000,
                    temperature=0
                )
                answer = response.choices[0].message.content.strip()
            except:
                answer = "حدث خطأ أثناء الحصول على استجابة من ChatGPT."

            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": answer})
            session['conversation_history'] = conversation_history

            return jsonify({
                'results': [{'answer': answer}],
                'classification': 'general'
            })

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/feedback', methods=['POST'])
def collect_feedback():
    try:
        data = request.get_json()
        username = data.get('username')
        conversation_id = data.get('conversation_id')
        rating_type = data.get('rating_type')
        comments = data.get('comments')

        if not username or not conversation_id or ((rating_type is None or rating_type == "") and not comments):

            return jsonify({"error": "يرجى تقديم اسم المستخدم ومعرف المحادثة والتقييم."}), 400

        db = SessionLocal()
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return jsonify({"error": "المستخدم غير موجود."}), 404

        # تحقق من أن المحادثة تنتمي للمستخدم
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == conversation_id,
            Conversation.user_id == user.id
        ).first()

        if not conversation:
            return jsonify({"error": "المحادثة غير موجودة أو لا تنتمي إلى المستخدم."}), 404

        # إنشاء سجل التغذية الراجعة
        new_feedback = Feedback(
            conversation_id=conversation_id,
            rating_type=rating_type,
            comments=comments
        )
        db.add(new_feedback)
        db.commit()
        db.refresh(new_feedback)
        db.close()

        return jsonify({"message": "تم تسجيل التغذية الراجعة بنجاح."}), 201

    except Exception as e:
        print(f"Error in /feedback endpoint: {e}")
        return jsonify({"error": str(e)}), 500


# ====================================================================
#             ETL وتجهيز بيانات
# ====================================================================

def etl_process():
    session_db = SessionLocal()
    metadata.reflect(bind=engine)
    all_data = []

    for table in metadata.sorted_tables:
        # استخراج البيانات لكل جدول
        data = session_db.query(table).all()
        transformed_data = [row.__dict__ for row in data]
        
        # إزالة المفتاح _sa_instance_state
        for record in transformed_data:
            record.pop('_sa_instance_state', None)

        # إضافة بيانات الجدول مع اسمه
        all_data.append({
            "table_name": table.name,
            "columns": [column.name for column in table.columns],
            "sample_data": transformed_data[:5]  # أخذ أول 5 صفوف فقط كعينة
        })

    session_db.close()
    train_llm(all_data)

def train_llm(data):
    for table_data in data:
        table_name = table_data["table_name"]
        columns = table_data["columns"]
        sample_data = table_data["sample_data"]

        # إعداد نص التدريب
        prompt = f"""
You are a database assistant. Here is a table schema and sample data for training:

Table Name: {table_name}
Columns: {', '.join(columns)}
Sample Data:
{sample_data}

Use this data to understand the structure of the table and answer user queries related to this table effectively.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable assistant trained on the database schema."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,
                temperature=0
            )
            print(f"Training completed for table: {table_name}")
        except OpenAI.OpenAIError as e:
            print(f"Error training model for table {table_name}: {e}")

schedule.every().day.at("00:00").do(etl_process)

def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(60)



def update_fine_tune():
    # تشغيل سكريبت prepare_finetune.py لتحديث النموذج
    python_executable = sys.executable
    subprocess.run([python_executable, "prepare_finetune.py"])

scheduler = BackgroundScheduler()
scheduler.add_job(update_fine_tune, 'interval', minutes=1)  # جدولة التحديث كل دقيقه
scheduler.add_job(send_weekly_report_with_chart, 'cron', hour='*')
try:
    scheduler.start()
except Exception as e:
    print(f"Error starting scheduler: {e}")

if __name__ == '__main__':
    schedule_thread = threading.Thread(target=run_schedule)
    schedule_thread.daemon = True
    schedule_thread.start()
    app.run(debug=False, host='0.0.0.0', port=5002)
