from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v412_review_m3_business.py"
SPEC = importlib.util.spec_from_file_location("m3b", SCRIPT)
assert SPEC and SPEC.loader
m3 = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = m3; SPEC.loader.exec_module(m3)
SNAPSHOT = "d" * 64


def plan_raw():
    return [{"query_id": f"q{i:02}", "selection_ordinal": i, "query_ordinal": j, "search_query": f"Q{i}/{j}", "crm_name": "École Alpha", "crm_address": "10 rue Victor Hugo 75001"} for i in range(1, 31) for j in range(1, 4)]


def result_url(rank):
    return "https://example.fr/1" if rank==1 else f"https://example-{rank}.fr/{rank}"


def search_raw(plan,pages=()):
    maxima={}
    for page in pages: maxima[(page["query_id"],page["query_ordinal"])]=max(maxima.get((page["query_id"],page["query_ordinal"]),0),page["result_rank"])
    rows=[]
    for r in plan:
        results=[]
        for rank in range(1,maxima.get((r.query_id,r.query_ordinal),0)+1):
                url=result_url(rank)
                safe,reason,host,domain=m3._url_identity(url)
                results.append({"rank":rank,"title":"result","snippet":"","resolved_url":url,"result_payload_sha256":m3.search_result_id(r.query_id,r.query_ordinal,rank,"result","",url),"preopen_family":"ENTITY_OFFICIAL_SITE_CANDIDATE" if safe and domain else "INADMISSIBLE","inadmissible_reason":"NONE" if safe and domain else reason,"normalized_hostname":host,"registrable_domain":domain})
        rows.append({"query_id":r.query_id,"query_ordinal":r.query_ordinal,"search_attempt_id":m3.search_attempt_id(r.query_id,r.query_ordinal,r.search_query),"status":"SUCCESS","results":results})
    return rows


def luhn_digit(prefix):
    for digit in "0123456789":
        value = prefix + digit
        if m3._luhn(value): return value
    raise AssertionError


S1 = luhn_digit("1111111110000")
S2 = luhn_digit("1111111110001")
S3 = luhn_digit("2222222220000")


def page_raw(siret=S1, *, qid="q01", page_char="a", group="PUBLIC_ADMINISTRATION", name="École Alpha", address="10 rue Victor Hugo 75001", distant=False, rank=1):
    gap = " x" * 300 if distant else " "
    text = f"{name} {address}{gap}{siret} relation établissement"
    encoded = text.encode()
    def span(value, start=0):
        left = encoded.index(value.encode(), start); return [left, left + len(value.encode())]
    name_span, address_span, siret_span = span(name), span(address), span(siret)
    relation_span = [0, len(encoded)]
    relation = encoded[relation_span[0]:relation_span[1]]
    search_attempt=m3.search_attempt_id(qid,1,f"Q{int(qid[1:])}/1"); url=result_url(rank); result_hash=m3.search_result_id(qid,1,rank,"result","",url); query_slot=rank; dossier_slot=rank; pid=m3.page_attempt_id(qid,1,rank,url,query_slot,dossier_slot); dnsid = hashlib.sha256(f"dns:{pid}".encode()).hexdigest()
    raw = text.encode()
    return {
        "page_attempt_id": pid, "dns_attempt_id": dnsid, "query_id": qid, "query_ordinal": 1, "result_rank": rank, "query_open_slot":query_slot,"dossier_open_ordinal":dossier_slot,"search_attempt_id":search_attempt,"result_payload_sha256":result_hash,"requested_url":url,"status": "SUCCESS", "independence_group": group,
        "raw_content_b64": base64.b64encode(raw).decode(), "raw_content_sha256": hashlib.sha256(raw).hexdigest(), "mime_type":"text/plain","charset":"UTF-8","text_decoder":"UTF8_STRICT","decoder_rule_id":m3.TEXT_DECODER_RULE,
        "crm": {"name": "École Alpha", "address": "10 rue Victor Hugo 75001"},
        "occurrences": [{"occurrence_id": hashlib.sha256(f"occ:{pid}:{siret}".encode()).hexdigest(), "siret_span": siret_span, "name_span": name_span, "address_span": address_span, "relation_span": relation_span, "source_excerpt_sha256": hashlib.sha256(relation).hexdigest(), "extractor_rule_id": m3.EXTRACTOR_RULE}],
    }


def dns_for(pages):
    return [{"dns_attempt_id": p["dns_attempt_id"], "parent_attempt_id": p["page_attempt_id"], "request_kind": "PAGE", "query_id": p["query_id"], "query_ordinal": p["query_ordinal"], "result_rank": p["result_rank"], "status": "SUCCESS", "addresses": ["1.1.1.1"]} for p in pages]


def broker_context(pages):
    plan=m3.validate_collection_plan(plan_raw()); attempts,results=m3.convert_search_responses(plan,search_raw(plan,pages)); decisions=m3.derive_page_decisions(results); dns=m3.convert_dns_responses(plan,dns_for(pages))
    return plan,attempts,results,decisions,dns


def lookup_raw(siret, ordinal, *, state="A", count=1, names=None, address="10 rue Victor Hugo 75001"):
    records = [] if count == 0 else [{"siret": siret, "state": state, "names": sorted(names or ["École Alpha"]), "address": address} for _ in range(count)]
    return {"siret": siret, "lookup_ordinal": ordinal, "snapshot_ref": "snapshot.parquet", "snapshot_sha256": SNAPSHOT, "records": records, "record_payload_sha256": hashlib.sha256(m3.canonical_bytes(records)).hexdigest()}


def build(pages, lookups=None):
    plan,attempts,results,decisions,dns=broker_context(pages)
    archives, occurrences, provenance = m3.convert_page_responses(plan, attempts,results,decisions,dns, pages)
    if lookups is None: lookups = [lookup_raw(row.siret, row.lookup_ordinal) for row in m3.derive_lookup_plan(occurrences)]
    return m3.seal_identity(plan, attempts,results,decisions,dns, archives, occurrences, provenance, lookups, snapshot_ref="snapshot.parquet", snapshot_sha256=SNAPSHOT)


def candidates(top1=S1, include=()):
    rows=[]
    for qi in range(1,31):
        vals=[top1 if qi==1 else f"{qi:09d}00001", *include] if qi==1 else [f"{qi:09d}00001"]
        n=2
        while len(vals)<100:
            v=f"{700000000+qi:09d}{n:05d}"; n+=1
            if v not in vals: vals.append(v)
        rows += [{"query_id":f"q{qi:02}","rank":rank,"candidate_siret":v} for rank,v in enumerate(vals,1)]
    return rows


def q1(decisions): return next(x for x in decisions if x.query_id=="q01")


def test_plan_30x3_closed_pk():
    assert len(m3.validate_collection_plan(plan_raw())) == 90
    bad=plan_raw(); bad[0]={**bad[0],"score":1}
    with pytest.raises(m3.BusinessIntegrityError,match="closed"): m3.validate_collection_plan(bad)
    dup=plan_raw(); dup[1]=dup[0].copy()
    with pytest.raises(m3.BusinessIntegrityError,match="primary key"): m3.validate_collection_plan(dup)


def test_search_attempts_are_exhaustive_bijective_90_even_without_results():
    plan=m3.validate_collection_plan(plan_raw()); attempts,results=m3.convert_search_responses(plan,search_raw(plan))
    assert len(attempts)==90 and len(results)==0
    assert all(not any(x.query_id==a.query_id and x.query_ordinal==a.query_ordinal for x in results) for a in attempts)
    replay=search_raw(plan); replay[-1]=replay[0].copy()
    with pytest.raises(m3.BusinessIntegrityError,match="PK/FK|bijective"):
        m3.convert_search_responses(plan,replay)


def test_reconstructs_luhn_name_address_triple_and_provenance():
    seal=build([page_raw()])
    assert len(seal.facts)==10 and {f.related_siren for f in seal.facts}=={S1[:9]}
    assert all(f.reconstruction_rule_id==m3.FACT_RULE for f in seal.facts)
    assert any(p.byte_start is not None for p in seal.provenance) and seal.evidence[-1].independence_group=="SIRENE_REGISTRY"


def test_no_authority_bool_or_name_address_identity_dto():
    raw=page_raw(); occurrence=raw["occurrences"][0]
    assert "qualified" not in occurrence and "site_specific" not in occurrence and "name" not in occurrence and "address" not in occurrence
    occurrence["qualified"]=True
    plan,attempts,results,decisions,dns=broker_context([raw])
    with pytest.raises(m3.BusinessIntegrityError,match="closed"): m3.convert_page_responses(plan,attempts,results,decisions,dns,[raw])


def test_orphan_page_and_dns_fk_rejected():
    raw=page_raw(); raw["query_ordinal"]=2
    valid=page_raw(); plan,attempts,results,decisions,dns=broker_context([valid])
    with pytest.raises(m3.BusinessIntegrityError,match="orphan"): m3.convert_page_responses(plan,attempts,results,decisions,dns,[raw])


def test_archive_hash_mismatch_rejected():
    raw=page_raw(); raw["raw_content_sha256"]="0"*64
    plan,attempts,results,decisions,dns=broker_context([raw])
    with pytest.raises(m3.BusinessIntegrityError,match="hash mismatch"): m3.convert_page_responses(plan,attempts,results,decisions,dns,[raw])


def test_raw_bytes_are_the_only_text_authority_and_unrelated_raw_fails():
    raw=page_raw(); unrelated=b"unrelated plain text without any CRM identity"; raw["raw_content_b64"]=base64.b64encode(unrelated).decode(); raw["raw_content_sha256"]=hashlib.sha256(unrelated).hexdigest()
    plan,attempts,results,decisions,dns=broker_context([raw])
    with pytest.raises(m3.BusinessIntegrityError,match="span|excerpt|reproduce"):
        m3.convert_page_responses(plan,attempts,results,decisions,dns,[raw])


@pytest.mark.parametrize("field,value",[("search_attempt_id","0"*64),("result_payload_sha256","0"*64),("requested_url","https://evil.invalid/"),("page_attempt_id","0"*64)])
def test_page_attempt_result_and_url_identity_mismatch(field,value):
    raw=page_raw(); raw[field]=value
    plan,attempts,results,decisions,dns=broker_context([raw])
    with pytest.raises(m3.BusinessIntegrityError,match="attempt/URL/result"):
        m3.convert_page_responses(plan,attempts,results,decisions,dns,[raw])


@pytest.mark.parametrize("mime,decoder,charset",[("text/html","UTF8_STRICT","UTF-8"),("application/pdf","UTF8_STRICT","UTF-8"),("text/plain","HTML_EXTRACTOR","UTF-8")])
def test_claim_rejects_html_pdf_and_unpinned_decoders(mime,decoder,charset):
    raw=page_raw(); raw["mime_type"],raw["text_decoder"],raw["charset"]=mime,decoder,charset
    plan,attempts,results,decisions,dns=broker_context([raw])
    with pytest.raises(m3.BusinessIntegrityError,match="unsupported PAGE text decoder"):
        m3.convert_page_responses(plan,attempts,results,decisions,dns,[raw])


def test_distant_siret_rejected():
    raw=page_raw(distant=True); plan,attempts,results,decisions,dns=broker_context([raw])
    with pytest.raises(m3.BusinessIntegrityError,match="locally bounded|too distant"): m3.convert_page_responses(plan,attempts,results,decisions,dns,[raw])


def test_falsified_name_or_address_span_rejected():
    for field in ("name_span","address_span"):
        raw=page_raw(); raw["occurrences"][0][field]=raw["occurrences"][0]["siret_span"]
        plan,attempts,results,decisions,dns=broker_context([raw])
        with pytest.raises(m3.BusinessIntegrityError,match="do not reproduce"): m3.convert_page_responses(plan,attempts,results,decisions,dns,[raw])


def test_falsified_page_crm_cannot_replace_frozen_plan_crm():
    raw=page_raw(); raw["crm"]={"name":"Forged Entity","address":"99 rue Fausse 75001"}
    plan,attempts,results,decisions,dns=broker_context([raw])
    with pytest.raises(m3.BusinessIntegrityError,match="frozen plan"): m3.convert_page_responses(plan,attempts,results,decisions,dns,[raw])


def test_missing_raw_or_span_rejected_closed():
    for mutation in ("raw_content_b64","siret_span"):
        raw=page_raw()
        if mutation in raw: del raw[mutation]
        else: del raw["occurrences"][0][mutation]
        plan,attempts,results,decisions,dns=broker_context([raw])
        with pytest.raises(m3.BusinessIntegrityError,match="closed"): m3.convert_page_responses(plan,attempts,results,decisions,dns,[raw])


def test_invalid_luhn_never_becomes_reliable():
    raw=page_raw(); bad="12345678901234"; text=base64.b64decode(raw["raw_content_b64"]).decode().replace(S1,bad); raw["raw_content_b64"]=base64.b64encode(text.encode()).decode(); raw["raw_content_sha256"]=hashlib.sha256(text.encode()).hexdigest(); raw["occurrences"][0]["source_excerpt_sha256"]=hashlib.sha256(text.encode()).hexdigest()
    plan,attempts,results,decisions,dns=broker_context([raw])
    with pytest.raises(m3.BusinessIntegrityError,match="Luhn"): m3.convert_page_responses(plan,attempts,results,decisions,dns,[raw])


def test_snapshot_and_record_payload_hash_pins():
    raw=page_raw(); seal=build([raw]); assert seal.sirene_records[0].snapshot_sha256==SNAPSHOT
    bad=lookup_raw(S1,1); bad["record_payload_sha256"]="0"*64
    with pytest.raises(m3.BusinessIntegrityError,match="payload hash"): build([raw],[bad])
    bad=lookup_raw(S1,1); bad["snapshot_sha256"]="0"*64
    with pytest.raises(m3.BusinessIntegrityError,match="snapshot pin"): build([raw],[bad])


def test_global_siret_lookup_cache_and_ordinal():
    pages=[page_raw(qid="q01",page_char="a"),page_raw(qid="q02",page_char="b")]
    seal=build(pages)
    assert len(seal.lookup_plan)==1 and seal.lookup_plan[0].query_ids==("q01","q02") and len(seal.sirene_records)==1


@pytest.mark.parametrize("state,count",[("F",1),("A",0),("A",2)])
def test_closed_missing_nonunique_no_false_reliable(state,count):
    identity=build([page_raw()],[lookup_raw(S1,1,state=state,count=count)])
    _,dec=m3.compare_after_barrier(identity,candidates())
    assert q1(dec).conditional_outcome=="NO_CONDITIONAL_SUPPORT" and q1(dec).reliable is False


def test_candidate_invariance_multisite_cross_siren_and_outside_top100():
    identity=build([page_raw()]); before=m3.identity_bytes(identity)
    _,correct=m3.compare_after_barrier(identity,candidates(top1=S1))
    _,multi=m3.compare_after_barrier(identity,candidates(top1=S2,include=(S1,)))
    _,cross=m3.compare_after_barrier(identity,candidates(top1=S3))
    assert q1(correct).conditional_outcome=="TOP1_IN_CONDITIONAL_SUPPORT" and not q1(correct).reliable
    assert q1(multi).collision_kind=="SAME_SIREN_MULTISITE" and q1(multi).conditional_alternative_in_top100 and not q1(multi).reliable
    assert q1(cross).collision_kind=="CROSS_SIREN_COLLISION" and not q1(cross).conditional_alternative_in_top100 and not q1(cross).reliable
    assert m3.identity_bytes(identity)==before


def test_same_group_collision_unresolved_and_two_groups_ambiguous():
    same=build([page_raw(S1,page_char="a"),page_raw(S3,page_char="b",rank=2)])
    _,dec=m3.compare_after_barrier(same,candidates(top1=S1,include=(S3,)))
    assert q1(dec).conditional_outcome=="NO_CONDITIONAL_SUPPORT" and not q1(dec).reliable
    pages=[page_raw(S1,page_char="a"),page_raw(S3,page_char="b",group="OFFICIAL_SECTOR_DIRECTORY",rank=2)]
    ambiguous=build(pages)
    _,dec=m3.compare_after_barrier(ambiguous,candidates(top1=S1,include=(S3,)))
    assert q1(dec).conditional_outcome=="MULTIPLE_CONDITIONAL_SUPPORT" and not q1(dec).reliable


def test_barrier_candidate_pk_and_purity():
    identity=build([page_raw()]); bad=replace(identity,identity_sha256="0"*64)
    with pytest.raises(m3.BusinessIntegrityError,match="barrier/hash"): m3.compare_after_barrier(bad,candidates())
    pool=candidates(); pool[1]=pool[0].copy()
    with pytest.raises(m3.BusinessIntegrityError,match="PK"): m3.compare_after_barrier(identity,pool)
    source=SCRIPT.read_text()
    for forbidden in ("import os","pathlib","socket","subprocess","pyarrow","pandas","open("): assert forbidden not in source


def test_seal_revalidates_tampered_occurrence_and_provenance():
    pages=[page_raw()]; plan,attempts,results,decisions,dns=broker_context(pages); archives,occurrences,provenance=m3.convert_page_responses(plan,attempts,results,decisions,dns,pages)
    forged_occ=(replace(occurrences[0],siret=S2),)
    forged_prov=(replace(provenance[0],related_siret=S2),)
    with pytest.raises(m3.BusinessIntegrityError,match="occurrence identity"):
        m3.seal_identity(plan,attempts,results,decisions,dns,archives,forged_occ,forged_prov,[lookup_raw(S2,1)],snapshot_ref="snapshot.parquet",snapshot_sha256=SNAPSHOT)


def test_self_attested_example_and_lookup_are_never_a_label_or_reliable():
    identity=build([page_raw(group="PUBLIC_ADMINISTRATION")],[lookup_raw(S1,1)])
    _,summaries=m3.compare_after_barrier(identity,candidates(top1=S1))
    summary=q1(summaries)
    assert identity.claim=="M3B_TEXT_PLAIN_IDNA311_CONDITIONAL_SUPPORT_NOT_LABEL"
    assert summary.conditional_outcome=="TOP1_IN_CONDITIONAL_SUPPORT"
    assert summary.reliable is False
    assert all(f.provenance_verified is False for f in identity.facts)
    assert all(e.provenance_verified is False for e in identity.evidence)
    assert not hasattr(m3,"BrokerStoreReceipt")
    source=SCRIPT.read_text()
    for forbidden_label in ("TOP1_CORRECT","TOP1_WRONG","AMBIGUOUS"):
        assert forbidden_label not in source


def test_page_slots_are_in_attempt_id_and_unique_within_quota():
    first=page_raw(rank=1); second=page_raw(S3,rank=2,group="OFFICIAL_SECTOR_DIRECTORY")
    duplicate=dict(second); duplicate["query_open_slot"]=1
    duplicate["page_attempt_id"]=m3.page_attempt_id("q01",1,2,duplicate["requested_url"],1,2)
    duplicate["dns_attempt_id"]=hashlib.sha256(f"dns:{duplicate['page_attempt_id']}".encode()).hexdigest()
    plan,attempts,results,decisions,dns=broker_context([first,duplicate])
    with pytest.raises(m3.BusinessIntegrityError,match="duplicate PAGE"):
        m3.convert_page_responses(plan,attempts,results,decisions,dns,[first,duplicate])


def test_page_attempt_id_exact_contract_formula_replay():
    url="https://example.fr/1"
    expected=hashlib.sha256(b"SIRETO-V412-R30-PAGE\0"+m3.canonical_bytes(["q01",1,1,url,1,1])).hexdigest()
    assert m3.page_attempt_id("q01",1,1,url,1,1)==expected
    page=page_raw(); plan,attempts,results,decisions,dns=broker_context([page])
    assert len(decisions)==1 and decisions[0].page_attempt_id==expected and decisions[0].decision=="OPEN_ATTEMPT"


def test_isolated_page_cannot_claim_slot2_or_dossier6():
    page=page_raw(); page["query_open_slot"]=2; page["dossier_open_ordinal"]=6
    page["page_attempt_id"]=m3.page_attempt_id("q01",1,1,page["requested_url"],2,6)
    page["dns_attempt_id"]=hashlib.sha256(f"dns:{page['page_attempt_id']}".encode()).hexdigest()
    plan,attempts,results,decisions,dns=broker_context([page])
    with pytest.raises(m3.BusinessIntegrityError,match="attempt/URL/result"):
        m3.convert_page_responses(plan,attempts,results,decisions,dns,[page])


@pytest.mark.parametrize("unsafe_url",["http://example.fr/x","https://127.0.0.1/x","https://example.fr:8443/x","https://example.fr/x#fragment"])
def test_unsafe_preopen_urls_are_skipped_without_slots(unsafe_url):
    plan=m3.validate_collection_plan(plan_raw()); responses=search_raw(plan,[page_raw()]); result=responses[0]["results"][0]
    _,reason,host,domain=m3._url_identity(unsafe_url)
    result.update({"resolved_url":unsafe_url,"result_payload_sha256":m3.search_result_id("q01",1,1,"result","",unsafe_url),"preopen_family":"INADMISSIBLE","inadmissible_reason":reason,"normalized_hostname":host,"registrable_domain":domain})
    attempts,results=m3.convert_search_responses(plan,responses); decisions=m3.derive_page_decisions(results)
    assert decisions[0].decision=="SKIP_INADMISSIBLE"
    assert decisions[0].query_open_slot is decisions[0].dossier_open_ordinal is decisions[0].page_attempt_id is None


def test_subdomains_deduplicate_on_recomputed_registrable_domain():
    plan=m3.validate_collection_plan(plan_raw()); pages=[page_raw(rank=1),page_raw(rank=2)]; responses=search_raw(plan,pages)
    for rank,url in enumerate(("https://a.example.fr/1","https://b.example.fr/2"),1):
        result=responses[0]["results"][rank-1]; _,_,host,domain=m3._url_identity(url)
        result.update({"resolved_url":url,"result_payload_sha256":m3.search_result_id("q01",1,rank,"result","",url),"normalized_hostname":host,"registrable_domain":domain})
    _,results=m3.convert_search_responses(plan,responses); decisions=m3.derive_page_decisions(results)
    assert [d.decision for d in decisions]==["OPEN_ATTEMPT","SKIP_DUPLICATE_DOMAIN"]
    assert [d.normalized_domain for d in decisions]==["example.fr","example.fr"]


def test_public_suffix_tuple_is_exactly_pinned_27_lines():
    raw=("\n".join(m3.PUBLIC_SUFFIXES)+"\n").encode("ascii")
    assert len(m3.PUBLIC_SUFFIXES)==27
    assert hashlib.sha256(raw).hexdigest()==m3.PUBLIC_SUFFIXES_SHA256=="10fe038631c2a3dd619370e368be3dbd9b6cb8daf2bd4203ced236cf6226c823"


def test_all_pinned_domain_vectors_replay_exactly():
    vectors=json.loads((ROOT/"config/v4_12_review_domain_vectors.json").read_text())
    assert vectors
    for vector in vectors:
        observed=m3.evaluate_domain_hostname(vector["input_hostname"])
        assert observed=={key:vector[key] for key in ("normalized_hostname","registrable_domain","matches_public_administration","matches_sirene_copy","safe")}


@pytest.mark.parametrize("url",[
    "https://%65xample.fr/x",
    "https://a..example.fr/x",
    "https://bad_name.example.fr/x",
    "https://-bad.example.fr/x",
    "https://"+("a"*64)+".example.fr/x",
])
def test_percent_encoded_and_invalid_idna_authorities_are_unsafe(url):
    safe,reason,_,_=m3._url_identity(url)
    assert safe is False and reason=="UNSAFE_URL"


def test_idna_dependency_is_exactly_version_311(monkeypatch):
    m3._assert_dependencies()
    assert m3.IDNA_VERSION=="3.11" and m3.PINNED_DEPENDENCIES==("idna==3.11",)
    monkeypatch.setattr(m3.idna,"__version__","3.10")
    with pytest.raises(m3.BusinessIntegrityError,match="version mismatch"):
        m3._url_identity("https://example.fr/")


@pytest.mark.parametrize("url",[
    "https://exa\r\nmple.fr/x",
    "https://exa\tmple.fr/x",
    "\x01https://example.fr/x",
    " https://example.fr/x",
    "https://example.fr/x\r\nInjected: yes",
    "https://example.fr/x\x00tail",
    "https:\\example.fr\\x",
    "https://example.fr\\x",
    "https://example.fr/\u200b",
])
def test_raw_url_controls_edges_formats_and_backslashes_are_unsafe(url):
    assert m3._url_identity(url)==(False,"UNSAFE_URL",None,None)


@pytest.mark.parametrize("url,host",[("https://localhost/x","localhost"),("https://service.local/x","service.local")])
def test_localhost_and_dot_local_are_explicitly_unsafe(url,host):
    assert m3._url_identity(url)==(False,"UNSAFE_URL",host,None)


@pytest.mark.parametrize("value",[None,"",123])
def test_non_string_or_empty_url_is_unsafe(value):
    assert m3._url_identity(value)==(False,"UNSAFE_URL",None,None)
