"""Hybrid agent: routes a question to the right tool, then answers.

Two tools:
  - docs  : RAG over the knowledge docs (concepts, definitions, methodology)
  - sql   : Text-to-SQL over DuckDB (precise numbers, counts, lists)

The agent loop is deliberately simple and robust for a small local model:
    route(question) -> pick a tool -> run it -> return a natural-language answer.

Everything runs locally via Ollama. Build the RAG index first
(`python src/rag_assistant.py build`).
"""
import re
from pathlib import Path

import duckdb
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag_assistant import CHAT_MODEL, rag_answer

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "lego.db"

# Multi-model design: a code-specialised model writes the SQL (it's far more
# accurate at it), while the general chat model handles routing, RAG, and
# phrasing results. Right model for each job.
SQL_MODEL = "qwen2.5-coder:1.5b"

# Only these tables are exposed to the SQL tool (keeps the prompt small + safe).
SQL_TABLES = [
    "raw_sets", "raw_themes", "theme_features", "popularity_scored", "raw_brickset",
]
TABLE_NOTES = {
    "raw_sets": "one row per LEGO set (set_num, name, year, theme_id, num_parts)",
    "raw_themes": "theme tree (id, parent_id, name); join raw_sets.theme_id = raw_themes.id",
    "theme_features": "per top-level theme survival data (name, is_active 1/0, pred_active_proba, n_sets, first_year, avg_parts)",
    "popularity_scored": "per set popularity (set_num, theme, year, num_parts, minifigs, owned_by actual, pred_owned_by)",
    "raw_brickset": "Brickset data per set (set_num, year, theme, pieces, minifigs, rating, review_count, owned_by, wanted_by)",
}

# Any of these words in a generated query means it is NOT a read-only SELECT.
FORBIDDEN = ["insert", "update", "delete", "drop", "create", "alter",
             "attach", "copy", "pragma", "replace", "truncate"]


def _llm() -> ChatOllama:
    """General chat model — routing, RAG, explaining results."""
    return ChatOllama(model=CHAT_MODEL, temperature=0)


def _sql_llm() -> ChatOllama:
    """Code-specialised model — writes the SQL only."""
    return ChatOllama(model=SQL_MODEL, temperature=0)


def get_schema_text() -> str:
    """Build a compact schema description for the SQL prompt (introspected live)."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    lines = []
    for t in SQL_TABLES:
        cols = con.execute(f"DESCRIBE {t}").df()["column_name"].tolist()
        lines.append(f"- {t} ({', '.join(cols)})  -- {TABLE_NOTES[t]}")
    con.close()
    return "\n".join(lines)


# Strong "this needs the database" signals — route straight to SQL.
DATA_HINTS = [
    "how many", "how much", "count", "number of", "average", "avg", "mean",
    "median", "list", "top ", "most ", "least ", "highest", "lowest", "total",
    "sum ", "rank", "compare", "per year", "which theme", "which set",
    "how old", "in 19", "in 20",
]


def route(question: str) -> str:
    """Decide which tool to use: 'sql' for data questions, 'docs' for concepts.

    A keyword heuristic catches obvious data questions (reliable), and a
    few-shot LLM call handles the rest (small models need the examples).
    """
    q = question.lower()
    if any(h in q for h in DATA_HINTS):
        return "sql"

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Route the question to a tool. Answer with ONLY 'sql' or 'docs'.\n"
         "'sql' = needs the dataset (counts, numbers, specific sets/themes/years).\n"
         "'docs' = a concept, definition, or how something works.\n\n"
         "Examples:\n"
         "Q: How many sets are in the Star Wars theme? -> sql\n"
         "Q: What is a flagship set? -> docs\n"
         "Q: Which year released the most sets? -> sql\n"
         "Q: How does the popularity model avoid leakage? -> docs\n"
         "Q: What is the average part count in 2023? -> sql"),
        ("human", "Q: {q} ->"),
    ])
    out = (prompt | _llm()).invoke({"q": question}).content.lower()
    return "sql" if "sql" in out else "docs"


def _extract_sql(text: str) -> str:
    """Pull a single SQL statement out of the model's reply."""
    text = re.sub(r"```sql|```", "", text, flags=re.IGNORECASE).strip()
    m = re.search(r"(with|select)\b.*", text, flags=re.IGNORECASE | re.DOTALL)
    sql = (m.group(0) if m else text).strip().rstrip(";")
    return sql


def _is_safe(sql: str) -> bool:
    low = sql.lower()
    return low.lstrip().startswith(("select", "with")) and not any(w in low for w in FORBIDDEN)


_SQL_SYSTEM = (
    "You are a DuckDB SQL expert. Given the schema, write ONE read-only SELECT "
    "query that answers the question. Return ONLY the SQL — no prose, no markdown. "
    "Use only the listed tables/columns.\n\n"
    "Guidance:\n"
    "- Set-level facts (num_parts, year, name) -> query raw_sets directly; don't "
    "join unless you need the theme name.\n"
    "- raw_sets has NO popularity columns. owned_by / wanted_by / rating live in "
    "popularity_scored and raw_brickset. To show a set NAME alongside owned_by, "
    "join that table to raw_sets ON set_num.\n"
    "- Theme Active/Retired status -> theme_features.\n"
    "- Avoid unnecessary joins.\n\nSchema:\n{schema}"
)


def _generate_sql(question: str, schema: str, correction: str) -> str:
    gen = ChatPromptTemplate.from_messages([
        ("system", _SQL_SYSTEM),
        ("human", "{q}{correction}"),
    ])
    raw = (gen | _sql_llm()).invoke(
        {"schema": schema, "q": question, "correction": correction}
    ).content
    return _extract_sql(raw)


def run_sql_tool(question: str, max_tries: int = 3) -> tuple[str, dict]:
    """Text-to-SQL with self-correction: generate -> run -> on error, fix and retry."""
    schema = get_schema_text()
    correction = ""
    last_sql, last_err = "", ""

    for attempt in range(1, max_tries + 1):
        sql = _generate_sql(question, schema, correction)

        if not _is_safe(sql):
            last_sql, last_err = sql, "not a read-only SELECT"
            correction = (f"\n\nYour previous query was unsafe or invalid:\n{sql}\n"
                          "Write a single read-only SELECT instead.")
            continue
        try:
            con = duckdb.connect(str(DB_PATH), read_only=True)
            df = con.execute(sql).df().head(50)
            con.close()
        except Exception as e:
            last_sql, last_err = sql, str(e)
            # Feed the error back so the model can fix its own query (agent loop).
            correction = (f"\n\nYour previous query failed with this error:\n{e}\n"
                          f"The query was:\n{sql}\nFix it and return only the corrected SQL.")
            continue

        # Success — phrase the result in plain English.
        explain = ChatPromptTemplate.from_messages([
            ("system", "Answer the question in one or two sentences using ONLY the "
                       "result table below. Copy any numbers EXACTLY as they appear — "
                       "do not round, reformat, or change a single digit. If the result "
                       "doesn't contain what's needed, say you couldn't determine it "
                       "from the data — never invent a number."),
            ("human", "Question: {q}\n\nResult:\n{table}"),
        ])
        answer = (explain | _llm()).invoke(
            {"q": question, "table": df.to_string(index=False)}
        ).content
        return answer, {"tool": "sql", "sql": sql, "rows": df, "attempts": attempt}

    return (f"I couldn't produce a working query after {max_tries} tries (last error: "
            f"{last_err}).", {"tool": "sql", "sql": last_sql, "error": last_err})


def agent_answer(question: str) -> tuple[str, dict]:
    """Route to a tool and return (answer, meta). meta['tool'] is 'sql' or 'docs'."""
    tool = route(question)
    if tool == "sql":
        return run_sql_tool(question)
    answer, sources = rag_answer(question)
    return answer, {"tool": "docs", "sources": sources}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How many themes are currently active?"
    ans, meta = agent_answer(q)
    print(f"[tool: {meta['tool']}]")
    if meta.get("sql"):
        print(f"[sql] {meta['sql']}")
    print(ans)
