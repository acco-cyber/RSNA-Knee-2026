"""Reuse only the exact validated three-study base; always run real new L2 MRI inference."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
base=json.loads((HERE/'rsna-knee-0962-target-crop224-audit.ipynb').read_text())
normal=json.loads((HERE/'actual_l2_fold0/rsna-knee-l2-0-962-target-fold0.ipynb').read_text())
pin=json.loads((HERE/'actual_l2_fold0/l2_source_pin.json').read_text())
assert pin['dataset_version']==1 and pin['annotations_complete'] is True
builder=ast.parse((HERE/'build_actual_l2.py').read_text())
parts={n.targets[0].id:ast.literal_eval(n.value) for n in builder.body
       if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
       and n.targets[0].id in {'setup','sidecar','stage'}}
cache=ROOT/'private_data/candidate_a'
cache_files=['submission.csv','candidate_receipt.json','blend_receipt.json','submission_receipt.json']
hashes={name:hashlib.sha256((cache/name).read_bytes()).hexdigest() for name in cache_files}
import pandas as pd
uids=pd.read_csv(cache/'submission.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist()
assert len(uids)==3
source_sha=hashlib.sha256((HERE/'rsna-knee-0962-target-crop224-audit.ipynb').read_bytes()).hexdigest()
prelude=r'''# Executable configuration: eligibility is exact, cached base has no hidden-test use.
import hashlib,json,os,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
NB_T0=time.time()
WORK=Path('/kaggle/working');WORK.mkdir(exist_ok=True)
_fast_comp=next(Path(p) for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                                 '/kaggle/input/rsna-knee-abnormality-detection') if (Path(p)/'test.csv').is_file())
_fast_test=pd.read_csv(_fast_comp/'test.csv',dtype={'StudyInstanceUID':str})
_fast_ids=_fast_test.StudyInstanceUID.tolist()
_official_cols=['StudyInstanceUID']+L2_EXPECTED_SOURCE['targets']
_official_ids=_fast_ids
FAST_PREVIEW_CACHE=len(_fast_ids)<=3 and _fast_ids==CACHE_BASE_UIDS
if FAST_PREVIEW_CACHE:
    _cache_candidates=[Path('/kaggle/input/notebooks/gwey1025/rsna-knee-0-962-target-crop224-audit'),
                       Path('/kaggle/input/rsna-knee-0-962-target-crop224-audit')]
    _cache_root=next((p for p in _cache_candidates if (p/'candidate_receipt.json').is_file()),None)
    if _cache_root is None:raise RuntimeError('The exact validated first-preview cache is required')
    for _name,_sha in CACHE_BASE_FILE_HASHES.items():
        if hashlib.sha256((_cache_root/_name).read_bytes()).hexdigest()!=_sha:
            raise RuntimeError('First-preview cache digest differs: '+_name)
    _cache_receipt=json.loads((_cache_root/'candidate_receipt.json').read_text())
    assert _cache_receipt['existing_student_checkpoints']==15 and _cache_receipt['new_l2_checkpoints']==0
    assert _cache_receipt['existing_student_weight']==.10 and _cache_receipt['crop224_weight_within_raptor']==.35
    assert _cache_receipt['incumbent_version']==358274276 and _cache_receipt['student_failed_studies']==0
    assert _cache_receipt['r7_manifest_sha256']=='c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6'
    _cache_blend=json.loads((_cache_root/'blend_receipt.json').read_text())
    assert all(_cache_blend[k]==0 for k in ('raptor384_failed','raptor224_failed','ours_failed','both_failed'))
    _cache_core=json.loads((_cache_root/'submission_receipt.json').read_text())
    assert _cache_core['status']=='PASS' and _cache_core['rows']==_cache_core['scored']==3
    assert _cache_core['fallback_studies']==0 and _cache_core['bundle_id']=='8c7b666e27dc0e8f'
    _cache_frame=pd.read_csv(_cache_root/'submission.csv',dtype={'StudyInstanceUID':str})
    assert _cache_frame.columns.tolist()==_official_cols and _cache_frame.StudyInstanceUID.tolist()==_official_ids
    assert not _cache_frame.StudyInstanceUID.duplicated().any()
    _cache_values=_cache_frame[_official_cols[1:]].to_numpy()
    assert np.isfinite(_cache_values).all() and ((_cache_values>=0)&(_cache_values<=1)).all()
    shutil.copyfile(_cache_root/'submission.csv',WORK/'submission.csv')
    (WORK/'preview_base_cache_receipt.json').write_text(json.dumps({
        'exact_visible_preview_only':True,'uids':CACHE_BASE_UIDS,'file_hashes':CACHE_BASE_FILE_HASHES,
        'first_candidate_source_sha256':CACHE_BASE_SOURCE_SHA256,
        'first_candidate':'gwey1025/rsna-knee-0-962-target-crop224-audit','version':1,
        'actual_new_l2_prediction_reused':False,'hidden_test_cache_enabled':False},indent=2))
print('L2_PREVIEW_CACHE_POLICY',{'enabled':FAST_PREVIEW_CACHE,'studies':len(_fast_ids),
                               'exact_first_preview_uids':_fast_ids==CACHE_BASE_UIDS},flush=True)
'''
configuration=('L2_EXPECTED_SOURCE = '+repr(pin)+'\nL2_CONTRACT_SOURCE = '+repr((HERE/'l2_provenance.py').read_text())+
               '\nL2_KNEE_SOURCE = '+repr((HERE.parent/'l2_train/knee_r2.py').read_text())+
               '\nCACHE_BASE_UIDS = '+repr(uids)+'\nCACHE_BASE_FILE_HASHES = '+repr(hashes)+
               '\nCACHE_BASE_SOURCE_SHA256 = '+repr(source_sha)+'\n'+prelude)
ast.parse(configuration)
assert normal['cells'][1]['cell_type']=='code'
student_index=next(i for i,c in enumerate(normal['cells']) if c['cell_type']=='code'
                   and ''.join(c['source']).startswith('import time as _st'))
tail_index=next(i for i,c in enumerate(normal['cells']) if c['cell_type']=='code'
                and ''.join(c['source']).startswith('# Apply the separate actual-L2 family'))
cells=[normal['cells'][0],{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
                         'source':configuration.splitlines(keepends=True)}]
fast_prefix=''.join(normal['cells'][student_index]['source']).split('\ntry:\n',1)[0]+'\n'
bootstrap=r'''# Prefer the original two wheels; retain installed trainer-compatible versions if absent.
import importlib.metadata,subprocess,sys,zipfile
_dep_slug='rsna-orthofoundation-fold0-epoch7-20261004'
_dep_input=Path('/kaggle/input')
_dep_candidates=[_dep_input/_dep_slug,_dep_input/'datasets/heliosli'/_dep_slug]
_dep_candidates+=list(_dep_input.glob('datasets/*/'+_dep_slug))+list(_dep_input.glob('datasets/*/'+_dep_slug+'/*'))
_dep_candidates+=list(_dep_input.glob('*/'+_dep_slug))+list(_dep_input.glob(_dep_slug+'/*'))
_dep_roots=list(dict.fromkeys(p for p in _dep_candidates if (p/'receipt.json').is_file()))
_dep_wheels=WORK/'preview_wheels';_dep_wheels.mkdir(exist_ok=True)
_dep_names=['timm-1.0.25-py3-none-any.whl','pydicom-3.0.1-py3-none-any.whl']
_dep_found=False
if len(_dep_roots)==1:
    _dep_root=_dep_roots[0]
    if (_dep_root/'dependencies.zip').is_file():
        with zipfile.ZipFile(_dep_root/'dependencies.zip') as _dep_zip:
            _dep_members={n:[p for p in _dep_zip.namelist() if Path(p).name==n] for n in _dep_names}
            if all(len(v)==1 for v in _dep_members.values()):
                for _dep_name in _dep_names:
                    (_dep_wheels/_dep_name).write_bytes(_dep_zip.read(_dep_members[_dep_name][0]))
                _dep_found=True
    else:
        _dep_members={n:list(_dep_root.rglob(n)) for n in _dep_names}
        if all(len(v)==1 for v in _dep_members.values()):
            for _dep_name in _dep_names:
                shutil.copyfile(_dep_members[_dep_name][0],_dep_wheels/_dep_name)
            _dep_found=True
if _dep_found:
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',
                    *[str(_dep_wheels/n) for n in _dep_names]],check=True)
_dep_versions={n:importlib.metadata.version(n) for n in ('timm','pydicom')}
_dep_parity=_dep_versions=={'timm':'1.0.25','pydicom':'3.0.1'}
_dep_receipt={'versions':_dep_versions,'same_first_base_versions':_dep_parity,
              'offline_wheels_used':_dep_found,'source_dataset':'heliosli/'+_dep_slug,
              'parity_limit':None if _dep_parity else 'Installed image versions retained for real L2-only preview; full hidden base keeps its original dependency setup',
              'wheel_sha256':{n:hashlib.sha256((_dep_wheels/n).read_bytes()).hexdigest() for n in _dep_names} if _dep_found else {}}
(WORK/'preview_dependency_receipt.json').write_text(json.dumps(_dep_receipt,indent=2))
print('FAST_PREVIEW_DEPENDENCIES',_dep_receipt,flush=True)
'''
ast.parse(bootstrap)
fast_inference=(bootstrap+'\n'+fast_prefix+'\n'+parts['setup']+
                "\nwith Pool(3) as _fast_pool, torch.no_grad():\n"
                "    for i,(st,x,mask,pos) in enumerate(_fast_pool.imap(_prep,_ids,chunksize=1)):\n"+
                ''.join('        '+line+'\n' for line in parts['sidecar'].splitlines())+
                "print('ACTUAL_L2_FAST_MRI_COMPLETE',len(_l2_predictions),'seconds',time.time()-NB_T0,flush=True)\n")
ast.parse(fast_inference)
normal_student=''.join(normal['cells'][student_index]['source'])
normal_tail=''.join(normal['cells'][tail_index]['source'])
normal_tail=normal_tail.replace('del models,m,ck,_l2_models,_model',
    'if FAST_PREVIEW_CACHE:\n    del _l2_models,_model\nelse:\n    del models,m,ck,_l2_models,_model')
for index,cell in enumerate(normal['cells'][2:],2):
    if cell['cell_type']!='code':cells.append(cell);continue
    source=''.join(cell['source'])
    if index==tail_index:
        wrapped=normal_tail
    elif index==student_index:
        wrapped=('if FAST_PREVIEW_CACHE:\n    exec(compile('+repr(fast_inference)+",'<actual-l2-fast-mri>','exec'),globals())\n"
                 'else:\n    exec(compile('+repr(source)+",'<unchanged-students-and-l2>','exec'),globals())\n")
    elif source.startswith('%%writefile '):
        header,body=source.split('\n',1)
        wrapped='if not FAST_PREVIEW_CACHE:\n    get_ipython().run_cell_magic(\'writefile\','+repr(header[len('%%writefile '):])+','+repr(body)+')\n'
    else:
        wrapped='if not FAST_PREVIEW_CACHE:\n    exec(compile('+repr(source)+",'<unchanged-base-cell-"+str(index)+">','exec'),globals())\n"
    ast.parse(wrapped)
    copied=dict(cell);copied.update(source=wrapped.splitlines(keepends=True),execution_count=None,outputs=[]);cells.append(copied)
normal['cells']=cells
normal['cells'][0]['source']+=['\nFor the exact three already-validated public-preview UIDs only, the base prediction '
    'and four receipts are reused by SHA256. Every actual new L2 probability is computed from '
    'the current MRI using the real trained checkpoint. Other UIDs, including hidden tests, '
    'execute the full unchanged base, reader, original 15 students and shared-tensor L2 path.\n']
out=HERE/'actual_l2_fold0_fastpreview';out.mkdir(exist_ok=True)
name='rsna-knee-l2-0-962-target-fold0-fastpreview.ipynb'
(out/name).write_text(json.dumps(normal,ensure_ascii=False,indent=1))
metadata=json.loads((HERE/'actual_l2_fold0/kernel-metadata.json').read_text())
metadata['code_file']=name
metadata['kernel_sources']+=['gwey1025/rsna-knee-0-962-target-crop224-audit']
(out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
(out/'l2_source_pin.json').write_text(json.dumps(pin,indent=2))
(out/'preview_cache_pin.json').write_text(json.dumps({'uids':uids,'file_hashes':hashes,'source_sha256':source_sha},indent=2))
print(out/name)
print('SHA256',hashlib.sha256((out/name).read_bytes()).hexdigest())
