#!/bin/bash
# Plan uniquement : ce script n'a PAS été exécuté pendant l'audit.
# Sans argument : --dry-run. --apply --confirmed-by-nathan après validation et arrêt des écritures SIRETO.
set -euo pipefail
mode="${1:---dry-run}"
case "$mode" in
  --dry-run) [[ $# -le 1 ]] || { echo "Arguments invalides" >&2; exit 2; } ;;
  --apply) [[ $# -eq 2 && "${2:-}" == --confirmed-by-nathan ]] || { echo "Validation Nathan requise : --apply --confirmed-by-nathan" >&2; exit 2; } ;;
  *) echo "Usage: $0 [--dry-run | --apply --confirmed-by-nathan]" >&2; exit 2 ;;
esac
repo="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
python3 - "$mode" "$repo/docs/disk_cleanup_plan.json" <<'PY'
import hashlib,json,os,shutil,stat,subprocess,sys
from pathlib import Path
MODE,PLAN=sys.argv[1:]; VOLUME=Path('/Volumes/CATNAT_DATA')
PLAN_SHA='4d126ad17da8b531d8bdb445768f8acd9bbdfb41491658429f0bb5ae34892600'
HANDOVER_SHA='38a21244d9b0a24aa9337a3e1a180f9ec8e27cd8d8ef1d29300efa730080b529'
TARGETS=[
    ('/Volumes/CATNAT_DATA/SIRETO_PPLX/probe_20261008_a', 40),
    ('/Volumes/CATNAT_DATA/sireto-native-anchor.3u44bv62', 56),
    ('/Volumes/CATNAT_DATA/sireto-native-anchor2.1ln9o_3x', 56),
    ('/Volumes/CATNAT_DATA/sireto-native-audit.CMWbvw', 80),
    ('/Volumes/CATNAT_DATA/sireto-native-forge.i1duf0kv', 36),
    ('/Volumes/CATNAT_DATA/sireto-native-forge2.rxbp5hon', 40),
    ('/Volumes/CATNAT_DATA/sireto-native-forge3.ejphutqx', 40),
    ('/Volumes/CATNAT_DATA/sireto-native-name-audit.oFeIVK', 80),
    ('/Volumes/CATNAT_DATA/sireto-native-nouuid.tp2FFe', 72),
    ('/Volumes/CATNAT_DATA/sireto-native-rereview.dZecJ6', 128),
    ('/Volumes/CATNAT_DATA/sireto-s0-fd-selfcheck.bQKNQZ', 0),
    ('/Volumes/CATNAT_DATA/sireto-s0-final-selfcheck.NdaYs5', 0),
    ('/Volumes/CATNAT_DATA/sireto-s0-selfcheck.MMNVt3', 0),
    ('/Volumes/CATNAT_DATA/sireto-s0-stable-selfcheck.Qno6TZ', 0),
    ('/Volumes/CATNAT_DATA/sireto-s0-stop-selfcheck.HrSLUw', 0),
    ('/Volumes/CATNAT_DATA/sireto-systemvol-final.4rFcrc', 16),
    ('/Volumes/CATNAT_DATA/sireto-v412-diag.6jrpPN', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-diag.TnF6YR', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-diag.VmG6W9', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-diag.ymbnIR', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-diagtmp.4Smw9Z', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-diagtmp.a3NN4c', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-diagtmp.eUZBgJ', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-diagtmp.mFnd7r', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-parent.8pAlsx', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-parent.AsM8eD', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-parent.R1Va2b', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-parent.jEawJf', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-parent.o852O6', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-tmp.Ybor2A', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-tmp.cF4Del', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-tmp.cKedzd', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-tmp.xogeaK', 0),
    ('/Volumes/CATNAT_DATA/sireto-v412-tmp.yhvyZj', 0),
    ('/Volumes/CATNAT_DATA/sireto_m3_cli_smoke_20260731', 2364),
]
def fail(message):raise SystemExit('REFUS : '+message)
raw=Path(PLAN).read_bytes()
if hashlib.sha256(raw).hexdigest()!=PLAN_SHA:fail('plan JSON modifié, réaudit nécessaire')
plan=json.loads(raw)
repo=Path(PLAN).parent.parent
if hashlib.sha256((repo/'handover.md').read_bytes()).hexdigest()!=HANDOVER_SHA:fail('handover modifié depuis cet audit')
changed=subprocess.check_output(['git','diff','--name-only',plan['reference_git_head'],'HEAD','--','docs','scripts'],cwd=repo,text=True).splitlines()
if set(changed)-{'docs/disk_cleanup_plan.md','docs/disk_cleanup_plan.json','scripts/disk_cleanup_apply.sh'}:fail('nouveau code/document commité depuis cet audit')
for ref in plan['git_reference_audit']['protected_current_file_fingerprints']:
    p=repo/ref['path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=ref['sha256']:fail('source/doc modifié : '+str(p))
allowed={x['path']:x for x in plan['apply_targets']}
if set(allowed)!={p for p,_ in TARGETS}:fail('liste explicite différente du plan')
if set(allowed)!={x['path'] for x in plan['folders'] if x['classification']=='OBSOLÈTE'}:fail('classe différente de OBSOLÈTE')
repo=Path(PLAN).parent.parent
if hashlib.sha256((repo/plan['frozen_v142']['manifest']).read_bytes()).hexdigest()!=plan['frozen_v142']['manifest_sha256']:fail('sceau V142 modifié')
# Existing absent V142 artifact is documented, never repaired or deleted here.
for ref in plan['reference_method']['metadata_manifest_records']:
    p=Path(ref['path'])
    if not p.is_file() or p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=ref['sha256']:fail('manifeste modifié : '+str(p))
protected=[Path(a['path']) for a in plan['frozen_v142']['artifacts']]
for folder in plan['folders']:
    if folder['classification']!='OBSOLÈTE' and folder['kind']=='partition':protected.append(Path(folder['path']))
def inspect(path,expected_kib):
    p=Path(path);snapshot=allowed[path]
    if not p.is_absolute() or p.resolve()!=p or not p.is_dir() or p.is_symlink():fail('cible absente/lien/invalide : '+path)
    if not p.is_relative_to(VOLUME) or not p.relative_to(VOLUME).parts[0].lower().startswith('sireto'):fail('hors périmètre SIRETO : '+path)
    if os.path.ismount(p):fail('point de montage : '+path)
    for q in protected:
        if p==q or p.is_relative_to(q) or q.is_relative_to(p):fail('chevauchement protégé : '+str(q))
    s=p.stat()
    if (s.st_dev,s.st_ino,s.st_mtime_ns)!=(snapshot['device'],snapshot['root_inode'],snapshot['root_mtime_ns']):fail('racine modifiée : '+path)
    actual=[]
    for parent,dirs,files in os.walk(p,followlinks=False):
        for name in sorted(dirs+files):
            q=Path(parent,name);s=q.lstat()
            if '.pending.' in name or q.is_symlink() or s.st_dev!=snapshot['device'] or os.path.ismount(q):fail('pending/lien/montage : '+str(q))
            if not (stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode)):fail('entrée spéciale : '+str(q))
            row=dict(relative_path=str(q.relative_to(p)),kind='directory' if q.is_dir() else 'file',inode=s.st_ino,size_bytes=s.st_size,mtime_ns=s.st_mtime_ns)
            if q.is_file():
                h=hashlib.sha256()
                with q.open('rb') as stream:
                    for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
                row['sha256']=h.hexdigest()
            actual.append(row)
    if sorted(actual,key=lambda x:x['relative_path'])!=snapshot['entries']:fail('contenu ajouté/retiré/modifié : '+path)
    result=subprocess.check_output(['du','-sk',path],text=True)
    measured=int(result.split('\t',1)[0])
    if measured!=expected_kib:fail(f'taille modifiée {path}: {measured} != {expected_kib} KiB')
    return measured
# Full preflight before any deletion; repeated immediately before each deletion.
for path,kib in TARGETS:
    inspect(path,kib);print(f'{MODE}: {kib} KiB  {path}',flush=True)
if MODE=='--apply':
    for path,kib in TARGETS:
        inspect(path,kib)
        shutil.rmtree(path)
        print('SUPPRIMÉ : '+path,flush=True)
print(f'{len(TARGETS)} dossiers OBSOLÈTE ; {sum(k for _,k in TARGETS)} KiB ; mode={MODE}')
PY
