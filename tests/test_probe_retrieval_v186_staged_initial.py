"""Trace-only code equivalence to immutable185; no national cost/recall claim."""
import ast
import inspect

import scripts.probe_retrieval_v185_staged_initial as original
import scripts.probe_retrieval_v186_staged_initial as workbench


class RemoveOnlyTrace(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == 'worker':
            node.body = [item for item in node.body if not (
                isinstance(item, ast.FunctionDef) and item.name == 'mark') and not (
                isinstance(item, ast.Assign) and len(item.targets) == 1
                and isinstance(item.targets[0], ast.Name) and item.targets[0].id == 'started')]
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'step'
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == 'mark'):
            assert len(node.value.args) == 1 and not node.value.keywords
            node.value = node.value.args[0]
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace('v186', 'v185').replace('V186', 'V185')
        return node


def test_complete_source_ast_diff_is_only_version_and_checkpoint_instrumentation():
    a = ast.parse(inspect.getsource(original))
    b = RemoveOnlyTrace().visit(ast.parse(inspect.getsource(workbench)))
    assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)


def test_every_worker_step_checkpoint_is_present_and_scores_sources_budgets_unchanged():
    tree = ast.parse(inspect.getsource(workbench.worker))
    marks = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'step']
    assert len(marks) >= 12
    assert all(isinstance(n.value, ast.Call) and n.value.func.id == 'mark' for n in marks)
    assert workbench.LIMITS == original.LIMITS
    assert workbench.FLAGS == original.FLAGS
    assert workbench.PartitionFacts is original.PartitionFacts
    assert workbench.partition_search is original.partition_search
    assert workbench.merge_partitions is original.merge_partitions
