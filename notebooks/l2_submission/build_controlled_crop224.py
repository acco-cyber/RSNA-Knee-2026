"""Build the strict controlled challenger from the user's measured 0.956 kernel.

This is a crop224 failure/ranking audit; no new L2 model is claimed.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path('/workspace/scratch/2d32e5629d42')
HERE = Path(__file__).parent
BASE = ROOT / 'recovered_kaggle/best_base/rsna-knee-v224-r7-reader-students10.ipynb'
PUBLIC = ROOT / '.private/public_sources/heliosli9573/0-9573-latest-public-dataset-score-in-one-click.ipynb'
MANIFEST = ROOT / '.private/public_sources/heliosli9573/dataset_manifest/manifest.json'
nb = json.loads(BASE.read_text())
public = json.loads(PUBLIC.read_text())

def source(c):
    return ''.join(c['source'])

def set_source(index, code):
    nb['cells'][index]['source'] = code.splitlines(keepends=True)

nb['cells'][0]['source'] = [
    '# RSNA Knee 0.962 Target — Crop224 Audit\n',
    '\n',
    'Controlled challenger built from the account’s completed **0.956** submission '
    '(script version 358274276). **0.962 is a target, not a measured score.**\n',
    '\n',
    'The learned models, preprocessing, reader and existing 15-student coefficients are retained. '
    'This changes only crop224 raw prediction/failure accounting, tied ranking, and explicit failure checks. '
    'Existing round-1/round-2 students are identified as such; this notebook contains no new L2 checkpoint.\n',
    '\n',
    'For the three-study public preview only, R7 uses its identical eager model instead of '
    'spending several minutes compiling it. Hidden tests with more than three studies retain the '
    'manifest’s compile setting and its built-in compiled-versus-eager check.\n',
]

# Save 224 raw probabilities before ranking. The selected public graph has the
# same model/checkpoint source and this precise addition.
old224 = source(nb['cells'][9])
needle = '    for d in devs:\n        for m, _ in models[d]: del m\n'
assert old224.count(needle) == 1
old224 = old224.replace(needle,
    '    _raw = pd.DataFrame(arm_probs[0].astype(np.float32), columns=LAB)\n'
    '    _raw.insert(0, "StudyInstanceUID", test_ids)\n'
    '    _raw.to_csv("/kaggle/working/raptor224_raw.csv", index=False)\n' + needle)
set_source(9, old224)

# Require the selected crop branch; a missing/failed branch cannot quietly submit
# the older graph under this candidate's name.
set_source(10, '''# Required crop224 stage; preserve stage order and global time margin.
import os, sys, time, subprocess
_left = 9 * 3600 - (time.time() - NB_T0) - 40 * 60
if _left < 30 * 60:
    raise RuntimeError("Crop224 Audit lacks enough remaining time for its required 224 stage")
_t224 = time.time()
_rc = subprocess.run([sys.executable, '-u', '/tmp/raptor224_run.py'],
                     timeout=min(_left, 2.5 * 3600)).returncode
if _rc != 0:
    raise RuntimeError(f"Required crop224 stage failed: exit {_rc}")
for _fname in ('raptor224_sub.csv', 'raptor224_raw.csv'):
    if not os.path.isfile('/kaggle/working/' + _fname):
        raise RuntimeError(f"Required crop224 stage missing {_fname}")
R224_OK = True
RAPTOR224_SECONDS = time.time() - _t224
print('RAPTOR224_STAGE_DONE seconds %.0f' % RAPTOR224_SECONDS, flush=True)
''')

# Use each model's own raw probabilities and own failed-study mask, with average
# ties. The numeric coefficients remain Raptor384 .325, Raptor224 .175, R7 .5.
set_source(12, source(public['cells'][8]))

# Pin the R7 manifest, not merely the mutable dataset slug. Runtime flags are
# passed through the manifest-owned worker interface; its source bytes remain
# unchanged and their existing SHA256 checks still run.
code = source(nb['cells'][6])
digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
marker = "manifest = json.loads((M_SRC / 'manifest.json').read_text())"
assert code.count(marker) == 1
code = code.replace(marker,
    f"_r7_manifest_hash = hashlib.sha256((M_SRC / 'manifest.json').read_bytes()).hexdigest()\n"
    f"assert _r7_manifest_hash == '{digest}', 'R7 manifest differs from the audited model graph'\n" + marker)
marker = "if manifest.get('compile'):"
assert code.count(marker) == 1
code = code.replace(marker,
    "import csv as _preview_csv\n"
    "with (DATA / 'test.csv').open(newline='', encoding='utf-8') as _fh:\n"
    "    _preview_studies = sum(1 for _ in _preview_csv.DictReader(_fh))\n"
    "PREVIEW_EAGER_ONLY = _preview_studies <= 3\n"
    "print('R7_COMPILE_POLICY', {'studies': _preview_studies, 'preview_eager_only': PREVIEW_EAGER_ONLY,\n"
    "                            'manifest_compile': bool(manifest.get('compile'))}, flush=True)\n"
    "if manifest.get('compile') and not PREVIEW_EAGER_ONLY:")
set_source(6, code)

# Do not let the required reader and required existing-student stages silently
# fall back to a different recipe after a stage exception.
code = source(nb['cells'][18])
code = code.replace("    print('Part B skipped: time budget used, the stage-3 submission is kept', flush=True)",
                    "    raise RuntimeError('Required reader stage cannot finish within the candidate budget')")
code = code.replace('        print("own leg FAILED, public stack submission kept:", repr(_o_e))',
                    '        raise RuntimeError("Required public reader stage failed") from _o_e')
set_source(18, code)
code = source(nb['cells'][19])
code = code.replace("    print('student stage skipped: time budget', flush=True)",
                    "    raise RuntimeError('Required existing-student stage cannot finish within the candidate budget')")
marker = 'print("student checkpoints:", _ckpts, flush=True)'
code = code.replace(marker, marker + '\nassert len(_ckpts) == 15, "Expected the exact existing 15-student graph; attach all four specified training kernels"')
code = code.replace('    print("student failed studies:", nfail, flush=True)',
                    '    print("student failed studies:", nfail, flush=True)\n'
                    '    if nfail > 0.03 * len(_ids):\n'
                    '        raise RuntimeError("More than 3% of studies failed in the required existing-student stage")')
code = code.replace('    print("STUDENT_STAGE_FAILED -> kept V20 submission:", type(e).__name__, e, flush=True)',
                    '    raise RuntimeError("Required existing-student stage failed; no alternate graph is submitted") from e')
set_source(19, code)

receipt = '''# Final selected CSV validation and receipt, after every transformation.
import hashlib, json
_final_path = WORK / 'submission.csv'
_final = pd.read_csv(_final_path, dtype={'StudyInstanceUID': str})
_official_ids = pd.read_csv(DATA / 'test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
_official_cols = ['StudyInstanceUID', 'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
                  'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
assert _final.columns.tolist() == _official_cols
assert _final.StudyInstanceUID.tolist() == _official_ids and not _final.StudyInstanceUID.duplicated().any()
_values = _final[_official_cols[1:]].to_numpy(dtype=np.float64)
assert np.isfinite(_values).all() and ((_values >= 0) & (_values <= 1)).all()
_receipt = {'candidate': 'RSNA Knee 0.962 Target - Crop224 Audit', 'actual_score': None,
            'incumbent_score': 0.956, 'incumbent_version': 358274276,
            'new_l2_checkpoints': 0, 'existing_student_checkpoints': len(_ckpts),
            'existing_student_weight': STUDENT_W, 'crop224_weight_within_raptor': A224,
            'r7_manifest_sha256': _r7_manifest_hash, 'preview_eager_only': PREVIEW_EAGER_ONLY,
            'selected_csv_sha256': hashlib.sha256(_final_path.read_bytes()).hexdigest(),
            'rows': len(_final), 'student_failed_studies': nfail}
(WORK / 'candidate_receipt.json').write_text(json.dumps(_receipt, indent=2))
print('CROP224_AUDIT_CANDIDATE_COMPLETE', json.dumps(_receipt), flush=True)
'''
nb['cells'].append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
                    'source':receipt.splitlines(keepends=True)})
for c in nb['cells']:
    if c['cell_type']=='code':
        c['outputs']=[]
        c['execution_count']=None
        text=source(c)
        # Validate the real Python body for %%writefile cells as well.
        if text.lstrip().startswith('%%writefile'):
            text=text.split('\n',1)[1]
        ast.parse(text)
meta=json.loads((BASE.parent/'kernel-metadata.json').read_text())
meta.update(id='gwey1025/rsna-knee-0-962-target-crop224-audit',
            title='RSNA Knee 0.962 Target - Crop224 Audit',
            code_file='rsna-knee-0962-target-crop224-audit.ipynb',
            is_private=True, enable_gpu=True, enable_internet=False)
meta.pop('id_no',None)
path=HERE/meta['code_file']
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
(HERE/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
(HERE/'crop224_build_receipt.json').write_text(json.dumps({
    'base_path':str(BASE), 'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
    'source_version':358274276,'candidate_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'r7_manifest_sha256':digest, 'parse_checks':'all Python cell and writefile bodies passed',
    'real_inference_completed':False,'actual_candidate_score':None},indent=2))
print(path)
