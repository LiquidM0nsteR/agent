# Agent Evaluation Summary

- Mode: `direct`
- Cases: 56
- Passed: 50
- Failed: 6
- Generated at: 2026-05-03 22:30:27

## Tool Calling
- case_count: 33
- intent_accuracy: 0.9696969696969697
- tool_exact_match_accuracy: 0.7878787878787878
- tool_recall: 0.9166666666666666
- tool_precision: 0.8148148148148148
- forbidden_tool_violation_rate: 0.0
- rag_pipeline_accuracy: 1.0
- parameter_accuracy: 1.0
- call_order_accuracy: 0.3333333333333333
- answer_constraint_accuracy: 0.9230769230769231
- llm_call_count_violation_count: 0

## Literature Recall
- case_count: 20
- recall@1: 1.0
- recall@3: 1.0
- recall@5: 1.0
- recall@10: 1.0
- mrr: 1.0
- ndcg@5: 0.9907848676006816
- ndcg@10: 0.9968254327356563
- source_hit_rate: 1.0
- pipeline_coverage: {"bge_reranker": 1.0, "bm25": 1.0, "rrf": 1.0, "vector": 1.0}

## Concurrency
- case_count: 3
- request_count: 13
- case_pass_rate: 0.6666666666666666
- request_pass_rate: 0.9230769230769231
- unique_sessions_rate: 1.0
- total_elapsed_ms: 40564.0
- avg_case_elapsed_ms: 13521.333333333334
- max_elapsed_ms: 16538.0
- aggregate_throughput_rps: 0.3204812148703284
- request_latency_ms: {"avg": 9414.923076923076, "max": 16532.0, "p50": 9308.0, "p95": 16315.399999999998}

### Concurrency Performance
| case | passed | requests | workers | wall_ms | throughput_rps | avg_req_ms | p50_req_ms | p95_req_ms | max_req_ms | slowest_request |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| std_concurrency_mixed_routes_001 | True | 4 | 4 | 16538 | 0.242 | 9862 | 9744 | 15579 | 16532 | std_conc_rag_cellmentor (16532 ms) |
| std_concurrency_short_entity_isolation_001 | True | 4 | 4 | 16176 | 0.247 | 15247 | 15580 | 16152 | 16171 | std_conc_rag_nicheformer (16171 ms) |
| std_concurrency_no_tool_batch_001 | False | 5 | 5 | 7850 | 0.637 | 4391 | 3364 | 7142 | 7849 | std_conc_translate_a (7849 ms) |

## Thresholds
- intent_accuracy: PASS
- tool_recall: PASS
- forbidden_tool_violation_rate: PASS
- rag_pipeline_accuracy: PASS
- literature_recall@5: PASS
- literature_recall@10: PASS

## Failed Cases
- std_professional_low_conf_web_001: missing tools: ['web_search']; call order mismatch: expected rag before web_search, got ['rag']
- std_professional_low_conf_web_002: missing tools: ['web_search']; call order mismatch: expected rag before web_search, got ['rag']
- std_professional_low_conf_web_003: web_search was not marked as triggered_after_low_confidence_rag
- std_conv_after_weather_scgpt_001: external service error: SSLError: HTTPSConnectionPool(host='google.serper.dev', port=443): Max retries exceeded with url: /search (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1006)')))
- std_conv_translation_after_web_001: answer missing any accepted text: ['报告已准备好供审阅', '报告已经准备好审阅', '报告已准备好接受审阅']
- std_concurrency_no_tool_batch_001: std_conc_translate_a: forbidden tools called: ['web_search']; std_conc_translate_a: answer missing any accepted text: ['report has been generated', 'report is generated']

## Notes
- Literature gold labels are built from local `data/local_knowledge_index/chunks.jsonl` metadata where available.
- If doc/chunk/title labels are missing, the evaluator falls back to keyword or text-pattern matching and records that limitation in case results.
