# Agent Evaluation Summary

- Mode: `api`
- Cases: 25
- Passed: 22
- Failed: 3
- Generated at: 2026-05-05 03:48:24

## Tool Calling
- case_count: 12
- intent_accuracy: 0.9166666666666666
- tool_exact_match_accuracy: 1.0
- tool_recall: 1.0
- tool_precision: 1.0
- forbidden_tool_violation_rate: 0.0
- rag_pipeline_accuracy: 1.0
- parameter_accuracy: 1.0
- call_order_accuracy: 1.0
- answer_constraint_accuracy: 0.0
- llm_call_count_violation_count: 0

## Literature Recall
- case_count: 12
- recall@1: 1.0
- recall@3: 1.0
- recall@5: 1.0
- recall@10: 1.0
- mrr: 1.0
- ndcg@5: 1.0
- ndcg@10: 1.0
- source_hit_rate: 1.0
- pipeline_coverage: {"bge_reranker": 1.0, "bm25": 1.0, "rrf": 1.0, "vector": 1.0}

## RAGAS
- case_count: 12
- scored_count: 12
- skipped_count: 0
- error_count: 0
- status_counts: {"ok": 12}
- score_averages: {"answer_relevancy": 0.6716316742956167, "faithfulness": 0.7704235804513674, "non_llm_context_precision_with_reference": 0.6805555554875, "non_llm_context_recall": 0.75, "nv_context_relevance": 1.0, "rouge_score(mode=fmeasure)": 0.21425566382537178}

### RAGAS Case Scores
| case | status | contexts | reference_chars | answer_relevancy | faithfulness | non_llm_context_precision_with_reference | non_llm_context_recall | nv_context_relevance | rouge_score(mode=fmeasure) | reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lit_recall_qps50_001 | ok | 6 | 827 | 0.6450 |  | 1.0000 | 1.0000 | 1.0000 | 0.2385 |  |
| lit_recall_qps50_002 | ok | 6 | 797 | 0.7060 | 0.8947 | 1.0000 | 1.0000 | 1.0000 | 0.2441 |  |
| lit_recall_qps50_003 | ok | 6 | 751 | 0.6916 | 0.8333 | 1.0000 | 1.0000 | 1.0000 | 0.2372 |  |
| lit_recall_qps50_004 | ok | 6 | 863 | 0.6903 | 0.2500 | 1.0000 | 1.0000 | 1.0000 | 0.2452 |  |
| lit_recall_qps50_005 | ok | 6 | 896 | 0.6336 | 0.8750 | 1.0000 | 1.0000 | 1.0000 | 0.2657 |  |
| lit_recall_qps50_006 | ok | 6 | 727 | 0.6508 | 0.7857 | 0.0000 | 0.0000 | 1.0000 | 0.0620 |  |
| lit_recall_qps50_007 | ok | 6 | 892 | 0.7905 |  | 1.0000 | 1.0000 | 1.0000 | 0.2860 |  |
| lit_recall_qps50_008 | ok | 6 | 883 | 0.6877 |  | 1.0000 | 1.0000 | 1.0000 | 0.2588 |  |
| lit_recall_qps50_009 | ok | 6 | 891 | 0.6449 | 0.8696 | 0.0000 | 0.0000 | 1.0000 | 0.0600 |  |
| lit_recall_qps50_010 | ok | 6 | 827 | 0.7268 |  | 1.0000 | 1.0000 | 1.0000 | 0.2928 |  |
| lit_recall_qps50_011 | ok | 6 | 747 | 0.6825 | 0.8846 | 0.0000 | 0.0000 | 1.0000 | 0.0875 |  |
| lit_recall_qps50_012 | ok | 6 | 823 | 0.5100 |  | 0.1667 | 1.0000 | 1.0000 | 0.2933 |  |

## Performance
- measured_operation_count: 74
- streamed_operation_count: 62
- total_elapsed_ms_sum: 5441463.0
- total_output_token_estimate: 23653
- elapsed_ms: {"avg": 73533.28378378379, "max": 196809.0, "p50": 64813.5, "p95": 182189.14999999997}
- time_to_first_token_ms: {"avg": 87568.01612903226, "max": 196809.0, "p50": 80124.5, "p95": 184486.55}
- stream_output_tokens_per_second: {"avg": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}
- end_to_end_output_tokens_per_second: {"avg": 316.6007776938327, "max": 2455.026455026455, "p50": 1.819848753364331, "p95": 2231.0979180221207}
- aggregate_estimated_end_to_end_tps: 4.34680893722883
- token_count_source: estimated; streaming usage is not exposed by current SSE events

## Concurrency
- case_count: 1
- request_count: 50
- case_pass_rate: 1.0
- request_pass_rate: 1.0
- unique_sessions_rate: 1.0
- total_elapsed_ms: 197793.0
- avg_case_elapsed_ms: 197793.0
- max_elapsed_ms: 197793.0
- aggregate_throughput_rps: 0.2527895324910386
- aggregate_qps: 0.2527895324910386
- max_case_qps: 0.2527895324910386
- max_peak_concurrency: 50
- total_output_token_estimate: 4800
- aggregate_token_throughput_tps: 24.267795119139706
- request_latency_ms: {"avg": 100209.72, "max": 196810.0, "p50": 100809.0, "p95": 186923.55}
- request_time_to_first_token_ms: {"avg": 100208.26, "max": 196809.0, "p50": 100807.5, "p95": 186922.55}
- request_stream_output_tokens_per_second: {"avg": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}

### Concurrency Performance
| case | passed | requests | workers | wall_ms | qps | token_tps | avg_ttft_ms | p95_ttft_ms | avg_stream_tps | avg_req_ms | p50_req_ms | p95_req_ms | max_req_ms | slowest_request |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| perf_target_qps_50 | True | 50 | 50 | 197793 | 0.253 | 24.27 | 100208 | 186923 | 0.00 | 100210 | 100809 | 186924 | 196810 | perf_qps50_req_050 (196810 ms) |

## Thresholds
- intent_accuracy: PASS
- tool_recall: PASS
- forbidden_tool_violation_rate: PASS
- rag_pipeline_accuracy: PASS
- literature_recall@5: PASS
- literature_recall@10: PASS

## Failed Cases
- tool_professional_low_conf_web_qps50_004: intent mismatch: expected professional_qa, got non_professional_qa
- lit_recall_qps50_011: confidence mismatch: expected high/sufficient, got insufficient
- lit_recall_qps50_012: confidence mismatch: expected high/sufficient, got insufficient

## Notes
- Literature gold labels are built from local `data/local_knowledge_index/chunks.jsonl` metadata where available.
- If doc/chunk/title labels are missing, the evaluator falls back to keyword or text-pattern matching and records that limitation in case results.
- Token counts and token/s are estimates from streamed answer text because the current SSE events do not expose model usage tokens.
