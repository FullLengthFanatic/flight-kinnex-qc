# Golden numbers (report v10)

Frozen on 2026-11-01 from the Supplementary Note v10 built by the analysis code before it was ported to this package.
- `v10_tables/`: the 36 source tables of the Note's figures and tables.
- `v10_inputs/`: the aggregate outputs of the pipeline and side analyses that the Note was built from (no per-read data).
- `v10_text_numbers.txt`: every number in the Note's visible text, in reading order.
- `allowed_changes.txt`: the numbers of the rebuilt Note's text that differ from `v10_text_numbers.txt`, each with its reason.
- `v10_numbers.json`: the numbers.json that `study/report/compute.py` writes from the v10 inputs: every number the Note's text quotes and every value its tables derive, unformatted. The render step's golden test renders from it and `v10_tables/`. Its software versions and commits are those of the checkout that froze it. Its keys are lowercase words, digits and underscores (the error rates of the benchmark read `bench_complete_0_001`).

`tests/test_golden.py` compares a rebuilt Note with these files. A difference is allowed only when CHANGELOG.md lists it with its reason.
