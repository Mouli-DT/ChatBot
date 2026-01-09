# ✅ Hybrid RAG Bot Using Local Neo4j (Replaces networkx)

import os
import logging
import torch
import threading
import re
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
from neo4j import GraphDatabase

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain_core.documents import Document
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import MergerRetriever
from langchain_core.runnables import RunnableLambda
from collections import defaultdict
from nemoguardrails import LLMRails, RailsConfig

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


csv_dir = os.path.abspath("./upload_csv")
pdf_dir = os.path.abspath("./upload_pdf")
VECTOR_STORE_CSV_DIR = './csv_embedding'
VECTOR_STORE_PDF_DIR = './pdf_embedding'

os.makedirs(VECTOR_STORE_CSV_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_PDF_DIR, exist_ok=True)

llm = OllamaLLM(model="gemma3:4b-it-qat")
embedder = OllamaEmbeddings(model="mxbai-embed-large:latest")

rails_config = RailsConfig.from_path("./guardrails")
nemo_rails = LLMRails(rails_config)

# ---------------- Neo4j Config ----------------
NEO4J_URI = "bolt://172.30.13.21:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "User@12345"  

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def clear_graph():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

def insert_employee_graph(row):
    with driver.session() as session:
        session.run("""
        MERGE (e:Employee {name: $name})
        SET 
            e.emp_number = $emp_number,
            e.joining_date = $joining_date,
            e.designation = $designation,
            e.location = $location,
            e.department = $department,
            e.business_unit = $business_unit,
            e.client_location = $client_location,
            e.email = $email
        """, 
        name=str(row['Employee Name']).strip() if pd.notna(row['Employee Name']) else "",
        emp_number=str(row['Employee Number']).strip() if pd.notna(row['Employee Number']) else "",
        joining_date=str(row['Date Of Joining']).strip() if pd.notna(row['Date Of Joining']) else "",
        designation=str(row['Curr.Designation']).strip() if pd.notna(row['Curr.Designation']) else "",
        location=str(row['Curr.Location']).strip() if pd.notna(row['Curr.Location']) else "",
        department=str(row['Curr.Department']).strip() if pd.notna(row['Curr.Department']) else "",
        business_unit=str(row['Curr.Businessunit']).strip() if pd.notna(row['Curr.Businessunit']) else "",
        client_location=str(row['Curr.Clientlocation']).strip() if pd.notna(row['Curr.Clientlocation']) else "",
        email=str(row['Email']).strip() if pd.notna(row['Email']) else ""
        )

        if pd.notna(row['Reporting To']) and str(row['Reporting To']).strip():
            session.run("""
            MERGE (m:Employee {name: $manager})
            MERGE (e:Employee {name: $employee})
            MERGE (e)-[:REPORTS_TO]->(m)
            """, 
            employee=str(row['Employee Name']).strip(), 
            manager=str(row['Reporting To']).strip()
            )



def build_knowledge_graph_from_csv(data_frame):
    clear_graph()
    for _, row in data_frame.iterrows():
        insert_employee_graph(row)


def run_graph_query(query):
    query_lower = query.lower().strip()

    def is_upward_query(q):
        return any(p in q for p in [
            "who does", "reports to who", "report to who",
            "reporting head of", "who is the manager of",
            "who is the supervisor of", "who is head of",
        ])

    def is_downward_query(q):
        return any(p in q for p in [
            "who reports to", "who are all reports to",
            "are all reports to", "team under",
            "reports under", "subordinates of", "who are under"
        ])

    with driver.session() as session:
        # 🔹 List all managers
        if "list all managers" in query_lower:
            result = session.run("""
                MATCH (e:Employee)-[:HAS_ROLE]->(r:Role)
                WHERE toLower(r.title) CONTAINS 'manager'
                RETURN e.name AS name
            """)
            return "\n".join(f"- {record['name']}" for record in result)

        # 🔹 Employees in location
        elif "employees in" in query_lower or "who works in" in query_lower:
            location_match = re.search(r"(?:employees in|who works in) ([a-zA-Z\s]+)", query_lower)
            if location_match:
                loc = location_match.group(1).strip()
                result = session.run("""
                    MATCH (e:Employee)-[:WORKS_IN]->(l:Location)
                    WHERE toLower(l.name) = $loc
                    RETURN e.name AS name
                """, loc=loc.lower())
                return "\n".join(f"- {record['name']}" for record in result)

        # 🔹 Sales roles
        elif "sales" in query_lower:
            limit = int(re.search(r"\d+", query_lower).group(0)) if re.search(r"\d+", query_lower) else 5
            result = session.run("""
                MATCH (e:Employee)-[:HAS_ROLE]->(r:Role)
                WHERE toLower(r.title) CONTAINS 'sales'
                RETURN e.name AS name
                LIMIT $limit
            """, limit=limit)
            return "\n".join(f"- {record['name']}" for record in result)

        # 🔹 ↑ UPWARD queries (get manager of employee)
        if is_upward_query(query_lower):
            match = re.search(r"(?:who does |reporting head of |who is the manager of |who is head of |)([a-z\s.]+)(?: report to| reports to who| reporting head)?", query_lower)
            if match:
                employee = match.group(1).strip()
                result = session.run("""
                    MATCH (e:Employee)-[:REPORTS_TO]->(m:Employee)
                    WHERE toLower(e.name) = $employee
                    RETURN m.name AS manager
                """, employee=employee.lower())
                managers = [record['manager'] for record in result]
                return f"{employee.title()} reports to {managers[0]}" if managers else f"No manager found for {employee.title()}."

        # 🔹 ↓ DOWNWARD queries (get team under a manager)
        elif is_downward_query(query_lower):
            match = re.search(r"(?:who reports to |who are all reports to |are all reports to |team under |reports under |subordinates of |who are under )([a-z\s.]+)", query_lower)
            if match:
                manager = match.group(1).strip()
                result = session.run("""
                    MATCH (e:Employee)-[:REPORTS_TO]->(m:Employee)
                    WHERE toLower(m.name) = $manager
                    RETURN e.name AS employee
                """, manager=manager.lower())
                employees = [record['employee'] for record in result]
                return (
                    f"People who report to {manager.title()}:\n" + "\n".join(f"- {emp}" for emp in employees)
                    if employees else f"No one reports to {manager.title()}."
                )

                # 🔹 Employee ID-based queries (matches any standalone DT-style ID)
        id_match = re.search(r"\b(dt\d{4})\b", query_lower)
        if id_match:
            emp_id = id_match.group(1).strip().lower()
            result = session.run("""
                MATCH (e:Employee)
                WHERE toLower(e.emp_number) = $emp_id
                RETURN e.name AS name, e.designation AS designation, 
                       e.business_unit AS bu, e.department AS dept, 
                       e.client_location AS client, e.location AS loc, 
                       e.joining_date AS doj, e.email AS email, 
                       e.emp_number AS id
            """, emp_id=emp_id)
            record = result.single()
            if record:
                return (
                    f"{record['name']} (ID: {record['id']}) is a {record['designation']} at DigitalTrack, "
                    f"located in {record['loc']}, working in the {record['dept']} department under the {record['bu']} business unit. "
                    f"Client location: {record['client']}. Joined on {record['doj']}. "
                    f"Contact: {record['email']}"
                )
            else:
                return f"No employee found with ID {emp_id.upper()}."




    return "No relevant graph data found."


def is_structured_query(text):
    keywords = ["list all", "who works in", "employees in", "designation of", "show me", "give me", "find", "which", "contact", "people in", "members in"]
    numbers = re.findall(r"\b\d+\b", text)
    return any(k in text.lower() for k in keywords) or bool(numbers)

def vectorstore_exists(path):
    return os.path.exists(os.path.join(path, "index.faiss"))

def save_vectorstore(vectorstore, path):
    try:
        vectorstore.save_local(path)
    except Exception as e:
        logger.error(f"Save error: {e}")

def load_vectorstore(path):
    try:
        return FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Load error: {e}")
        return None

def process_csv_files():
    try:
        save_path = os.path.join(VECTOR_STORE_CSV_DIR, "vectorstore")
        if vectorstore_exists(save_path):
            return load_vectorstore(save_path)

        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if not csv_files:
            print("[WARN] No CSV files found in upload_csv/")
            return None

        dfs = [pd.read_csv(os.path.join(csv_dir, f)) for f in csv_files]
        data_frame = pd.concat(dfs, ignore_index=True)
        build_knowledge_graph_from_csv(data_frame)

        # docs = [Document(page_content=". ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))) for _, row in data_frame.iterrows()]
        docs = [
            Document(
                page_content=". ".join(
                    f"{col}: {val}"
                    for col, val in row.items()
                    if pd.notna(val) and str(val).strip()
                )
            )
            for _, row in data_frame.iterrows()
        ]

        vectorstore = FAISS.from_documents(docs, embedder)
        save_vectorstore(vectorstore, save_path)
        return vectorstore

    except Exception as e:
        logger.error(f"CSV processing error: {e}")
        return None

def process_pdf_files():
    try:
        save_path = os.path.join(VECTOR_STORE_PDF_DIR, "vectorstore")
        if vectorstore_exists(save_path):
            return load_vectorstore(save_path)

        loader = DirectoryLoader(pdf_dir, glob="**/*.pdf", recursive=True)
        docs = loader.load()
        if not docs:
            return None

        splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(split_docs, embedder)
        save_vectorstore(vectorstore, save_path)
        return vectorstore

    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        return None

csv_store = process_csv_files()
pdf_store = process_pdf_files()

if csv_store and pdf_store:
    retriever = MergerRetriever(retrievers=[csv_store.as_retriever(), pdf_store.as_retriever()])
elif csv_store:
    retriever = csv_store.as_retriever()
elif pdf_store:
    retriever = pdf_store.as_retriever()
else:
    retriever = None

prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant named DboT.
Use the facts below to answer the user's question.

Facts from knowledge graph:
{graph_facts}

Facts from documents:
{context}

Instruction:
{instruction}

Question:
{question}
""")

chain = prompt_template | llm | StrOutputParser() | RunnableLambda(lambda x: x.split("Answer: ")[-1])

session_conversations = defaultdict(list)
ollama_lock = threading.Lock()

def is_greeting(text):
    greetings = {"hi", "hello", "hey", "good", "morning", "evening"}
    # Extract words from input
    words = set(re.findall(r'\b\w+\b', text.lower()))
    # Match word-wise and look for common greetings
    return any(
        (word in greetings) or 
        (f"{word} morning" in text.lower()) or 
        (f"{word} evening" in text.lower()) 
        for word in words
    )

def is_thanking(text):
    thanks_patterns = {
        "thank you", "thanks", "thankyou", "thank u", "ty", "thx"
    }
    # Normalize text
    lowered = text.lower()
    return any(phrase in lowered for phrase in thanks_patterns)

def is_creator_question(text):
    return any(re.search(p, text.lower()) for p in [r'who (created|built|developed)', r'creator', r'designed you'])

def nemo_policy_check(user_input: str):
    # response = nemo_rails.generate(
    #     messages=[{"role": "user", "content": user_input}]
    # )

    # print("NeMo raw response:", response)

    # if isinstance(response, dict):
    #     content = response.get("content", "").strip()

        # If input rails fired, content is empty → block
        # if content == "":
            text = user_input.lower()

            if re.search(r"admin|password|credential|login", text):
                return True, "I can’t help with credentials or access-related requests."

            if re.search(r"bomb|explosive|weapon|kill", text):
                return True, "I can’t assist with anything involving violence or weapons."

            if re.search(r"hack|bypass|steal", text):
                return True, "I can’t help with illegal activities."

            # return True, "This request is not allowed."

        # Future-proof (output rails)
        # return True, content

            return False, None


def chat_response(user_input, mode="concise", session_id="default"):
    if not user_input:
        return "Please enter a question."

    # NeMo Guardrails
    blocked, message = nemo_policy_check(user_input)
    if blocked:
        return message
    
    if is_greeting(user_input):
        return "Hello! 👋 How can I assist you today?"
    if is_thanking(user_input):
        return "You're welcome! 😊"
    if is_creator_question(user_input):
        return "I was created by the R&D Team of DigitalTrack Solutions. 🤖"

    context = ""
    graph_facts = ""

    if is_structured_query(user_input):
        graph_facts = run_graph_query(user_input)
        if "No relevant graph data" not in graph_facts:
            context = ""
        elif retriever:
            docs = retriever.get_relevant_documents(user_input)
            context = "\n\n".join(doc.page_content for doc in docs)
    else:
        if retriever:
            docs = retriever.get_relevant_documents(user_input)
            context = "\n\n".join(doc.page_content for doc in docs)

    instruction = "Answer concisely in 1 or 2 sentences." if mode == "concise" else "Give a detailed and complete answer."

    augmented_input = {
        "context": context,
        "graph_facts": graph_facts,
        "instruction": instruction,
        "question": user_input
    }

    session_conversations[session_id].append({"role": "user", "content": user_input})

    try:
        with ollama_lock:
            full_response = chain.invoke(augmented_input)
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return "Error contacting the bot."

    session_conversations[session_id].append({"role": "assistant", "content": full_response})

    return full_response.strip()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    mode = data.get("mode", "concise").lower()
    session_id = data.get("session_id", "default")
    response = chat_response(user_input, mode, session_id)
    return jsonify({"response": response})

@app.route("/clear_session", methods=["POST"])
def clear_session():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    if session_id in session_conversations:
        del session_conversations[session_id]
    return jsonify({"message": f"Session {session_id} cleared."}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
