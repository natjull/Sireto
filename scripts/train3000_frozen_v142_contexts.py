"""Non-numeric query contexts required by the unchanged V142 collector.

Original source functions; only scope/admission/I/O boilerplate is rebound.
No frozen526 per-query context is used as an input to a new query.
"""
import ast
import inspect
from pathlib import Path
import resource
import time
from types import SimpleNamespace

from scripts import train3000_frozen_v142_numeric as numeric
mini=numeric.mini


class Scope(ast.NodeTransformer):
    def visit_Constant(self,node):
        if type(node.value) is int and node.value==526:
            return ast.copy_location(ast.Call(func=ast.Name(id='len',ctx=ast.Load()),
                args=[ast.Name(id='panel',ctx=ast.Load())],keywords=[]),node)
        return node
    def visit_FunctionDef(self,node):
        if node.name=='resource_guard':
            node.body=[ast.Return(value=ast.Call(func=ast.Name(id='BRIEF10_RESOURCE_GUARD',ctx=ast.Load()),args=[],keywords=[]))]
            return node
        return self.generic_visit(node)
    def visit_BinOp(self,node):
        text=ast.unparse(node)
        if text=='80 * 1024 ** 3':return ast.copy_location(ast.Constant(value=10*1024**3),node)
        if text=='512 * 1024 ** 2':return ast.copy_location(ast.Name(id='BRIEF10_CAP',ctx=ast.Load()),node)
        return self.generic_visit(node)


def function(module,name,overrides):
    numeric.pinned(module)
    tree=ast.parse(inspect.getsource(getattr(module,name)))
    tree=ast.fix_missing_locations(Scope().visit(tree))
    namespace=dict(vars(module),**overrides)
    exec(compile(tree,str(module.__file__)+':scope_and_execution_only', 'exec'),namespace)
    numeric.BODIES[str(module.__file__)+':'+name]=numeric.hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    return namespace[name]


def prepare(panel, numeric_directory, directory):
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts import embeddinggemma2_v88_typed_context_native as expansion
    from scripts import embeddinggemma2_v98_official_initial_alignment as initials
    from scripts import embeddinggemma2_v130_structural_evidence as rna
    from scripts import embeddinggemma2_v130_banatic_memberships as banatic
    from scripts import embeddinggemma2_v133_supplier_site_context as supplier
    from scripts import embeddinggemma2_v137_continuity_address_source as address
    from scripts import embeddinggemma2_v142_bano_native_ablation as v142
    directory.mkdir(parents=True)
    original_io=numeric.install_io(directory)
    try:
        launch=mini.read(numeric_directory/'v14/launch.json')
        # Stock/index/role authorities from the frozen serving launch, with the
        # actual NEW query and numerical-source bindings instead of old526 IDs.
        launch=dict(mini.read(mini.bano.cohort.preflight.RUN/'launch.json'),panel=launch['panel'])
        numeric_gen=mini.read(numeric_directory/'v75/features_generation.json')
        source=mini.read(mini.bano.cohort.preflight.RUN/'features_generation.json')
        indirect=source['indirect_names'];mini.verified(indirect)
        fg=dict(numeric_gen,indirect_names=indirect)
        mini.save(directory/'features_generation.json',fg)
        mini.save(directory/'launch.json',launch)
        guard=lambda *args:mini.guard()
        old=mini.read(expansion.OUTPUT/'launch.json')
        config=dict(old,panel=launch['panel'],prepared_v81=mini.record(directory/'features_generation.json'))
        e=directory/'expansion';e.mkdir()
        function(expansion,'prepare',dict(authority=lambda out:config,guard=guard))(e)
        expansion_ref=mini.record(e/'context_expansion.json')
        # Historical COG bytes only. The old URL is provenance, not a request.
        gen=mini.read(initials.OUTPUT/'generation.json');cog=mini.verified(gen['cog'])
        def local_cog(url,timeout=None):
            if url!=initials.COG:raise ValueError('Only the pinned historical COG request')
            return cog.open('rb')
        function(initials,'main',dict(OUTPUT=directory/'initials',SOURCE=directory,
            urllib=SimpleNamespace(request=SimpleNamespace(urlopen=local_cog))))()
        initial_ref=mini.record(directory/'initials/initial_alignment.json')
        function(rna,'prepare',dict(BASE=directory))(directory/'rna')
        def resource_guard():
            mini.guard()
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        function(banatic,'run',dict(BASE=directory,OUTPUT=directory/'banatic',
            BRIEF10_RESOURCE_GUARD=resource_guard,BRIEF10_CAP=mini.LIMIT))()
        # V133/V137 fixed526/native-exit gates are historical orchestration, not
        # prerequisites for computing their query-only source evidence. Their
        # complete scientific bodies are extracted unchanged, not faked exits.
        supplier_root=directory/'supplier'
        build=numeric.slice_body(supplier,"    indirect=read(verify(fg['indirect_names']))",
            "    if len(extras)!=526:",'out,deadline,launch,fg,panel','\n    return extras\n',
            dict(write=numeric.engine.io.write,verify=mini.verified,duckdb=duckdb,pa=pa,pq=pq))
        supplier_extras=build(supplier_root,time.time()+8*3600,launch,fg,panel)
        mini.save(supplier_root/'query_extras.json',supplier_extras)
        family=mini.read(mini.verified(fg['family_sources']))
        prior=dict(source_links=family['compiled_links'],source_brands=indirect,
            source_legal_units=mini.record(numeric.corporate.audit_utils.STOCK),
            extras=mini.record(supplier_root/'query_extras.json'))
        mini.save(supplier_root/'generation.json',prior)
        mini.save(supplier_root/'source_preparation_complete.json',dict(
            kind='IN_PROCESS_SOURCE_PREPARATION_COMPLETE_NOT_A_CHILD_EXIT',queries=len(supplier_extras),
            extras=prior['extras'],implementation=numeric.pinned(supplier)))
        def address_record(path):
            if Path(path)==supplier_root/'owned_tool_exit.json':
                return mini.record(supplier_root/'source_preparation_complete.json')
            return mini.record(path)
        def address_write(path,value):
            if Path(path).name=='launch.json':
                value=dict(value);value['actual_prior_in_process_preparation']=value.pop('prior_owned_exit')
            return numeric.engine.io.write(path,value)
        build=numeric.slice_body(address,"    indirect=read(verify(previous['source_brands']))",
            "    if len(extras)!=526:",'out,deadline,previous,launch,fg,panel','\n    return extras\n',
            dict(PREPARED=supplier_root,write=address_write,verify=mini.verified,record=address_record,
                duckdb=duckdb,pa=pa,pq=pq))
        address_root=directory/'address'
        address_extras=build(address_root,time.time()+8*3600,prior,launch,fg,panel)
        mini.save(address_root/'query_extras.json',address_extras)
        result=dict(context=expansion_ref,initials=initial_ref,
            rna=mini.record(directory/'rna/context.json'),banatic=mini.record(directory/'banatic/context.json'),
            supplier=prior['extras'],address=mini.record(address_root/'query_extras.json'),indirect=indirect)
        mini.save(directory/'contexts.json',dict(refs=result,queries=len(panel),raw_labels_read=False,
            constructor_authorities=numeric.AUTHORITIES,bodies=numeric.BODIES,
            execution_adaptations=['scope526 -> actual query count','brief10 memory/disk guard',
                'local pinned COG bytes, no network','fresh v174 roots; no fake upstream exits']))
        return result
    finally:numeric.engine.io.write=original_io
