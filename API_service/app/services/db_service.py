
import psycopg2
from psycopg2.extras import Json
from app.config import settings

def get_conn():
    try:
        return psycopg2.connect(
            dbname=settings.PG_DB,
            user=settings.PG_USER,
            password=settings.PG_PASSWORD,
            host=settings.PG_HOST,
            port=settings.PG_PORT,
        )
    except:
        return None

def log_interaction(query, response, status, safety, sources, timings):
    conn = get_conn()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO interaction_logs (
                user_query, model_response, status, safety_check_result,
                sources, model_id_llm, model_id_embed,
                total_duration_sec, input_guardrail_sec,
                embedding_sec, retrieval_sec, llm_generation_sec,
                output_guardrail_sec, client_latency_sec, config_snapshot
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                query, response, status, safety, sources,
                "", "", timings["total_sec"],
                timings["input_guardrail_sec"],
                timings["embedding_sec"], timings["retrieval_sec"],
                timings["llm_generation_sec"], timings["output_guardrail_sec"],
                0.0, Json({})
            )
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()
