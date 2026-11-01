"""PyArrow schemas for all parquet tables."""

import pyarrow as pa

SEGMENTS = pa.schema([
    ("zm", pa.uint32()),
    ("di", pa.int16()),
    ("dl", pa.int8()),
    ("dr", pa.int8()),
    ("pos", pa.int8()),
    ("direction", pa.int8()),
    ("qs", pa.int32()),
    ("qe", pa.int32()),
    ("sread_len", pa.int32()),
    ("np", pa.int16()),
    ("rq", pa.float32()),
    ("end_class", pa.string()),
    ("tso_side", pa.string()),
    ("tso_toward_lower", pa.bool_()),
    ("internal", pa.bool_()),
    ("n_internal", pa.int16()),
    ("internal_hits", pa.string()),
    ("head_hits", pa.string()),
    ("tail_hits", pa.string()),
    ("bead_ok", pa.bool_()),
    ("bead_shift", pa.int8()),
    ("polyT_len", pa.int16()),
    ("polyT_anchored", pa.bool_()),
    ("insert_len", pa.int32()),
])

READ_INDEX = pa.schema([
    ("zm", pa.uint32()),
    ("read_len", pa.int32()),
    ("np", pa.int16()),
    ("rq", pa.float32()),
])

READ_FEATURES = pa.schema([
    ("zm", pa.uint32()),
    ("source", pa.string()),
    ("name", pa.string()),
    ("kind", pa.string()),
    ("strand", pa.string()),
    ("start", pa.int32()),
    ("end", pa.int32()),
    ("edits", pa.int8()),
])

READ_SUMMARY = pa.schema([
    ("zm", pa.uint32()),
    ("source", pa.string()),
    ("read_len", pa.int32()),
    ("np", pa.int16()),
    ("rq", pa.float32()),
    ("chain", pa.string()),
    ("chain_flipped", pa.bool_()),
    ("flags", pa.string()),
    ("primary_pattern", pa.string()),
    ("n_mas", pa.int16()),
    ("skera_passing", pa.bool_()),
])

ANNOT_SEGMENTS = pa.schema([
    ("zm", pa.uint32()),
    ("source", pa.string()),
    ("seg_index", pa.int16()),
    ("start", pa.int32()),
    ("end", pa.int32()),
    ("left_token", pa.string()),
    ("right_token", pa.string()),
    ("pos", pa.int8()),
    ("sread_len", pa.int32()),
    ("end_class", pa.string()),
    ("tso_side", pa.string()),
    ("tso_toward_lower", pa.bool_()),
    ("internal", pa.bool_()),
    ("n_internal", pa.int16()),
    ("bead_ok", pa.bool_()),
    ("polyT_len", pa.int16()),
    ("polyT_anchored", pa.bool_()),
    ("insert_len", pa.int32()),
])

READS = pa.schema([
    ("zm", pa.uint32()),
    ("read_len", pa.int32()),
    ("np", pa.int16()),
    ("rq", pa.float32()),
    ("passing", pa.bool_()),
    ("n_segments", pa.int8()),
    ("direction", pa.int8()),
    ("positions", pa.string()),
    ("full_array", pa.bool_()),
    ("total_sread_len", pa.int32()),
    ("max_sread_len", pa.int32()),
    ("n_proper", pa.int8()),
    ("unsegmented_bp", pa.int32()),
])

ZMW = pa.schema([
    ("zm", pa.uint32()),
    ("read_length", pa.int32()),
    ("pol_length", pa.int32()),
    ("read_qual", pa.float32()),
    ("hifi", pa.int8()),
    ("num_passes", pa.int32()),
])

FATE = pa.schema([
    ("zm", pa.uint32()),
    ("di", pa.int16()),
    ("pos", pa.int8()),
    ("end_class", pa.string()),
    ("sread_len", pa.int32()),
    ("skera_passing", pa.bool_()),
    ("lima_pair", pa.string()),
    ("lima_pass", pa.bool_()),
    ("refine_asrun", pa.bool_()),
    ("refine_polyA_5", pa.bool_()),
    ("refine_polyA_8", pa.bool_()),
    ("refine_polyA_12", pa.bool_()),
    ("refine_polyA_20", pa.bool_()),
    ("mapped_fltnc", pa.bool_()),
])
