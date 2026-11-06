# Golden numbers (report v10)

Frozen on 2026-11-01 from the Supplementary Note v10 built by the analysis code before it was ported to this package.
- `v10_tables/`: the 36 source tables of the Note's figures and tables.
- `v10_inputs/`: the aggregate outputs of the pipeline and side analyses that the Note was built from (no per-read data).
- `v10_text_numbers.txt`: every number in the Note's visible text, in reading order.
- `allowed_changes.txt`: the numbers of the rebuilt Note's text that differ from `v10_text_numbers.txt`, each with its reason.
- `v10_numbers.json`: the numbers.json that `study/report/compute.py` writes from the v10 inputs: every number the Note's text quotes and every value its tables derive, unformatted. The render step's golden test renders from it and `v10_tables/`. Its software versions and commits are those of the checkout that froze it. Its keys are lowercase words and digits joined by `_` (the error rates of the benchmark read `bench_complete_0_001`).
- `approved_number_changes.json`: not part of v10. The approved changes since v10 that move a number of the Note built from the full rerun (`report/`): per entry a numbers.json key, a table cell or a text number, its v10 value, its new value and words from the CHANGELOG.md line that lists it. Only the tests of `report/` in `tests/test_golden.py`, and the real downstream test, apply it.

`tests/test_golden.py` compares a rebuilt Note with these files. A difference is allowed only when CHANGELOG.md lists it with its reason.
