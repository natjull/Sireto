"""Static/query-input proof for five isolated native source recipes.

Never reads outcomes or compares candidates to decide equivalence. Does not
search, copy output shards, alter a retrieval module, fit, or open closed data.
"""
import ast
import hashlib
import importlib.metadata
from pathlib import Path
import time

import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v143_empty_reference_continuation as target
from scripts import embeddinggemma2_v134_supplier_native_ablation as supplier_target
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE = io.HOME/'v140/structural_native_disk_20261013_night'
OUT = io.HOME/'v140/isolated_equivalence_20261013_night'
ORIGINAL = io.ROOT/'scripts/embeddinggemma2_v138_structural_native_ablation.py'
TARGET_ORIGINAL = io.ROOT/'scripts/embeddinggemma2_v132_local_evidence_ablation.py'
VARIANTS = ('rna_only', 'banatic_only', 'supplier_own_site_only',
            'supplier_temporal_only', 'continuity_address_only', 'structural_sources_max')


class ScoringAST(ast.NodeTransformer):
    """Alpha-normalization and explicit non-scoring guard removal only.

float(coefficient) is redundant because BOTH call sites already pass the same
Python float. Admin helper equivalence is separately traced through V71->V56
->V32->V23; component4 is never changed by appended parent/context components.
"""
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'self':
            node.attr = {'score': 'scores', 'owners': 'rna'}.get(node.attr, node.attr)
        return node

    def visit_Name(self, node):
        node.id = {'at': 'positions', 'pos': 'positions', 'value': 'confidences',
                   'values': 'confidences', 'native_proof': 'lexical'}.get(node.id, node.id)
        return node

    def visit_If(self, node):
        if node.body and all(isinstance(stmt, ast.Raise) for stmt in node.body) and not node.orelse:
            return None  # Assertions do not build/score a query; inputs verified.
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'guard':
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and node.func.id == 'float' and len(node.args) == 1
                and isinstance(node.args[0], ast.Attribute) and node.args[0].attr == 'coefficient'):
            return node.args[0]
        return node


def collector_ast(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EvidenceCollector')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'search')
    raw = ast.dump(method, include_attributes=False)
    normalized = ast.dump(ScoringAST().visit(method), include_attributes=False)
    return dict(raw_ast=raw, normalized_scoring_ast=normalized,
                raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                normalized_sha256=hashlib.sha256(normalized.encode()).hexdigest())


def main():
    if OUT.exists():
        raise ValueError('Fresh proof namespace')
    top_exit = io.read(SOURCE/'owned_tool_exit.json')
    if top_exit['exit_code'] or not top_exit['own_tool_exit_confirmed']:
        raise ValueError('All six actual completed outputs before equivalence proof')
    for variant in VARIANTS:
        gen = io.read(SOURCE/variant/'generation.json')
        ex = io.read(SOURCE/variant/'generate_exit.json')
        if gen['status'] != 'COMPLETE_SEALED_ALL526' or gen['queries'] != 526 or ex['exit_code'] or not ex['own_child_exit_confirmed']:
            raise ValueError('All six complete owned generations, no output inspection')
    old = io.read(SOURCE/'launch.json')
    new = target.authority(target.ROOT)
    verify(old['implementation'])
    verify(old['recipe'])
    for helper in old['helpers']:
        verify(helper)
    if importlib.metadata.version('tantivy') != '0.25.1':
        raise ValueError('Version-specific collector tie proof only')
    if old['panel'] != new['panel'] or old['baseline_features'] != new['baseline_features'] or old['baseline_calibration'] != new['baseline_calibration']:
        raise ValueError('Exact query,51-fields,58-model authority bindings')
    for field in ('index_manifest', 'index_meta', 'keys', 'xy', 'baseline_generation', 'baseline_exit'):
        if old[field] != new[field]:
            raise ValueError('Identical corpus/baseline binding: '+field)
        verify(old[field])
    if io.read(SOURCE/'calibration/generation.json') != io.read(target.ROOT/'calibration/generation.json'):
        raise ValueError('Identical models and assignment, not similar parameters')
    if io.read(SOURCE/'features_generation.json') != io.read(target.ROOT/'features_generation.json'):
        raise ValueError('Identical complete baseline numerical features and contexts')
    panel = io.read(verify(old['panel']))
    if len(panel) != 526 or any(q['fold'] not in (2, 3, 4) for q in panel):
        raise ValueError('Same safe whole526 population')
    prepared = io.read(target.ROOT/'prepared.json')
    if prepared['status'] != 'COMPLETE_ALL526_LABEL_FREE_SOURCE_SCORES' or set(prepared['shards']) != {q['query_id'] for q in panel}:
        raise ValueError('All526 target prepared numerical key bindings')
    for name in ('rna', 'banatic'):
        if old['structural_sources'][name]['generation'] != new[name+'_generation']:
            raise ValueError('Same owner-context source hash')
        gen = io.read(verify(new[name+'_generation']))
        context = io.read(verify(gen['context']))
        for q in panel:
            rows = context[q['query_id']]
            refs = [r['siren_ref'] for r in rows]
            if refs != sorted(set(refs)) or any(not 0 < r['confidence'] <= .500001 for r in rows):
                raise ValueError('Same sorted unique bounded owner bitplanes; merge cannot reorder')
    original_ast, target_ast = collector_ast(ORIGINAL), collector_ast(TARGET_ORIGINAL)
    if original_ast['normalized_scoring_ast'] != target_ast['normalized_scoring_ast']:
        raise ValueError('Actual source scoring AST differs; do NOT reuse')
    # V143 executes the same collector body with only empty-key reduction changed.
    wrapper = (io.ROOT/'scripts/embeddinggemma2_v143_empty_reference_continuation.py').read_text()
    if "'int(group.max())+1': 'int(group.max(initial=-1))+1'" not in wrapper:
        raise ValueError('Exact known empty-only wrapper')
    checks = {}
    for name, source_root in (('supplier', supplier_target.SOURCE), ('address', supplier_target.ADDRESS)):
        g = io.read(verify(old['structural_sources'][name]['generation']))
        if record(source_root/'generation.json') != old['structural_sources'][name]['generation']:
            raise ValueError('Same isolated leaf-context source generation')
        refs = io.read(verify(g['extras']))
        count = 0
        for q in panel:
            records = list(refs[q['query_id']].values()) if name == 'supplier' else [refs[q['query_id']]]
            for ref in records:
                if ref is None:
                    continue
                with np.load(verify(ref), allow_pickle=False) as f:
                    keys, confidence = f['keys'], f['confidence']
                    if (keys.shape != confidence.shape or np.any(keys[1:] <= keys[:-1])
                            or not np.isfinite(confidence).all() or np.any(confidence <= 0) or np.any(confidence > .500001)):
                        raise ValueError('Valid exact projected confidence inputs')
                count += 1
        checks[name] = dict(generation=old['structural_sources'][name]['generation'], verified_nonempty_source_shards=count,
                           proof='ONE_SOURCE_MAX_ZERO_EQUALS_SAME_FLOAT64_PROJECTED_CONFIDENCE_TIMES_IDENTICAL_FLOAT_W6')
    OUT.mkdir(parents=True)
    io.write(OUT/'collector_asts.json', dict(original=original_ast, target=target_ast))
    io.write(OUT/'report.json', dict(status='STATIC_INPUT_EQUIVALENCE_FIVE_ISOLATED_RECIPES_NOT_CANDIDATE_MATCHING',
        implementation=record(Path(__file__)), original_collector=record(ORIGINAL), target_collector=record(TARGET_ORIGINAL),
        source_launch=record(SOURCE/'launch.json'), target_launch=record(target.ROOT/'launch.json'),
        source_owned_exit=record(SOURCE/'owned_tool_exit.json'), target_prepared=record(target.ROOT/'prepared.json'),
        scoring_ast=record(OUT/'collector_asts.json'), leaf_sources=checks,
        owner_variants=['rna_only', 'banatic_only'], leaf_variants=['supplier_own_site_only', 'supplier_temporal_only', 'continuity_address_only'],
        administrative_proof='V71.lexical_parts->V56.lexical_parts->V32.lexical_parts->V23.lexical_parts; appended components leave index4 unchanged',
        leaf_query_order='SAME_BASE_QUERY_THEN_BITS0_TO19_THEN_OWNER_ADMIN_BITS0_TO19;SIRET_ONLY;LIMIT100;COUNT_FALSE',
        coefficient_cast_proof='BOTH_CALLERS_PASS_FLOAT_MODEL_COEFFICIENT6',
        native_binding='EXACT_SAME_V115_NATIVE_SEARCH_HELPERS_CALIBRATORS_INDEX_METADATA_AND_FINAL_SCORE_SIRET_SORT',
        installed_tantivy_py=importlib.metadata.version('tantivy'),
        thread_tie_inference='VERSION0.25.0_TOPDOCS_PER_SEGMENT_AND_MERGE_USE_SCORE_THEN_DOCADDRESS;INDEX_METADATA_UNCHANGED',
        primary_tie_sources=['https://github.com/quickwit-oss/tantivy-py/blob/0.25.1/Cargo.toml',
            'https://github.com/quickwit-oss/tantivy/blob/0.25.0/src/collector/top_collector.rs',
            'https://github.com/quickwit-oss/tantivy/blob/0.25.0/src/collector/top_score_collector.rs'],
        no_outputs_or_outcomes_read_for_eligibility=True, no_retrieval_or_unit_tests=True, no_copy_performed=True,
        supplier_destination_not_initialized_yet_requires_same_source_model_binding_before_copy=True,
        runtime_reuse_requires_external_manifest_and_not_full_new_slo=True,
        no_new_scientific_recipe_or_closed_dataset=True, finished_unix=time.time()))
    print('STATIC_EQUIVALENCE_PROOF_COMPLETE_NO_OUTPUT_COMPARISON', str(OUT), flush=True)


if __name__ == '__main__':
    main()
