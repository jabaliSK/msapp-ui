
import time
from fastapi import BackgroundTasks
from app.services.llm_service import llm_generate, embed_text
from app.services.safety_service import safety_check
from app.services.milvus_service import (
    search_cache, search_collection, insert_cache
)
from app.services.db_service import log_interaction
from app.config import settings
from app.models.schemas import ChatRequest, ChatResponse, ProcessTimings

def rag_chat(req: ChatRequest, bg: BackgroundTasks):
    t0 = time.perf_counter()

    # 1 Safety
    t_start = time.perf_counter()
    if not safety_check(req.query):
        timings = _timings(t0, input_guard=time.perf_counter()-t_start)
        return ChatResponse(
            response="I cannot process this request due to safety guidelines.",
            sources=[],
            safety_check="FAILED_INPUT",
            timings=timings,
            llm_model="",
            embed_model="",
            guard_model="",
        )

    t_input = time.perf_counter() - t_start

    # 2 Embedding
    t_start = time.perf_counter()
    vec = embed_text(req.query)
    t_embed = time.perf_counter() - t_start

    # 3 Cache
    cache_hit = search_cache(vec)
    if cache_hit:
        timings = _timings(t0, t_input, t_embed, retrieval=0)
        return ChatResponse(
            response=cache_hit["response"],
            sources=["SEMANTIC_CACHE"],
            safety_check="PASSED_CACHE",
            timings=timings,
            llm_model="",
            embed_model="",
            guard_model="",
        )

    # 4 Retrieval
    t_start = time.perf_counter()
    chunks, sources = search_collection(vec)
    t_retrieval = time.perf_counter() - t_start

    # 5 LLM
    t_start = time.perf_counter()
    response = llm_generate(req.query, chunks)
    t_llm = time.perf_counter() - t_start

    # 6 Output safety
    t_start = time.perf_counter()
    safe = safety_check(response, output=True)
    t_out = time.perf_counter() - t_start

    final = response if safe else "Response withheld due to safety."

    # 7 Cache
    if safe:
        insert_cache(vec, final, req.query)

    timings = _timings(t0, t_input, t_embed, t_retrieval, t_llm, t_out)

    bg.add_task(log_interaction,
        req.query, final, "Success", "PASSED" if safe else "FAILED_OUTPUT",
        sources, timings.dict()
    )

    return ChatResponse(
        response=final,
        sources=sources,
        safety_check="PASSED" if safe else "FAILED_OUTPUT",
        timings=timings,
        llm_model="model",
        embed_model="embed",
        guard_model="guard",
    )

def _timings(start, input_guard=0, embed=0, retrieval=0, llm=0, out=0):
    return ProcessTimings(
        input_guardrail_sec=round(input_guard, 4),
        embedding_sec=round(embed, 4),
        retrieval_sec=round(retrieval, 4),
        llm_generation_sec=round(llm, 4),
        output_guardrail_sec=round(out, 4),
        total_sec=round(time.perf_counter() - start, 4),
    )
