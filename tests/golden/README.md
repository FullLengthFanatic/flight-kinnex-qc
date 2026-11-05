# Golden numbers (report v10)

Frozen on 2026-11-01 from the Supplementary Note v10 built by the analysis code before it was ported to this package.
- `v10_tables/`: the 36 source tables of the Note's figures and tables.
- `v10_inputs/`: the aggregate outputs of the pipeline and side analyses that the Note was built from (no per-read data).
- `v10_text_numbers.txt`: every number in the Note's visible text, in reading order.
- `v10_numbers.json`: the numbers.json that `study/report/compute.py` writes from the v10 inputs: every number the Note's text quotes and every value its tables derive, unformatted. The render step's golden test renders from it and `v10_tables/`. Its software versions and commits are those of the checkout that froze it.

`tests/test_golden.py` compares a rebuilt Note with these files. A difference is allowed only when CHANGELOG.md lists it with its reason.
