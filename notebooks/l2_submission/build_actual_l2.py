"""Prepare an explicit real-L2 family; do not replace it with old students."""
import argparse, ast, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--label-manifest', type=Path, default=HERE.parents[2]/'private_data/l2_dataset/manifest.json')
parser.add_argument('--dataset-version', type=int)
args=parser.parse_args()
KNEE_SOURCE=(HERE.parent/'l2_train/knee_r2.py').read_text()
CONTRACT_SOURCE=(HERE/'l2_provenance.py').read_text()
TRAINER_HASH=hashlib.sha256((HERE.parent/'l2_train/train_student_l2.py').read_bytes()).hexdigest()
EXPECTED=None
if args.label_manifest.is_file():
    manifest=json.loads(args.label_manifest.read_text())
    if manifest.get('annotations_complete') is True:
        assert manifest['annotated_studies']==manifest['report_only_studies']==4349
        expected_hashes={name:manifest['outputs'][name] for name in ('labels_l2.csv','weights_l2.csv')}
        for name, sha256 in expected_hashes.items():
            assert hashlib.sha256((args.label_manifest.parent/name).read_bytes()).hexdigest()==sha256
        expected_hashes.update({'knee.py':hashlib.sha256(KNEE_SOURCE.encode()).hexdigest(),'trainer':TRAINER_HASH})
        a1=HERE.parents[2]/'private_data/a1'
        expected_hashes['labels_cons.csv']=hashlib.sha256((a1/'labels_cons.csv').read_bytes()).hexdigest()
        EXPECTED={'dataset':'gwey1025/rsna-knee-l2-labels-20261015','dataset_version':args.dataset_version,
                  'annotations_complete':True,'annotated_studies':manifest['annotated_studies'],
                  'targets':list(manifest['report_share']),
                  'source_hashes':expected_hashes}
        # Match the trainer's official ordering, cache availability and zero-confidence exclusion.
        import pandas as pd
        official=pd.read_csv(HERE.parents[2]/'private_data/competition/train.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
        lab=pd.read_csv(a1/'labels_cons.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[official.index]
        weights=pd.read_csv(args.label_manifest.parent/'weights_l2.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
        cache=pd.read_csv(a1/'cache256_meta.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
        labels=manifest['report_share'].keys()
        gold=official[list(labels)].notna().all(axis=1)
        weak=lab[~gold & lab.index.isin(cache.index[cache.n_slots>0])]
        tr=weak[(weak.fold!=0)&(weights.loc[weak.index,list(labels)].sum(axis=1)>0)]
        va=weak[weak.fold==0]
        EXPECTED['train_ids_sha256']=hashlib.sha256('\n'.join(tr.index).encode()).hexdigest()
        EXPECTED['val_ids_sha256']=hashlib.sha256('\n'.join(va.index).encode()).hexdigest()
nb=json.loads((HERE/'rsna-knee-0962-target-crop224-audit.ipynb').read_text())
# The incumbent's broad student glob would also see the newly attached L2 file.
# Resolve the existing 15 students by explicit training-kernel/fold names instead.
_old_source=''.join(nb['cells'][19]['source'])
_old_line='_ckpts = sorted(set(glob.glob("/kaggle/input/notebooks/*/*/student*.pt") + glob.glob("/kaggle/input/*/student*.pt")))'
assert _old_source.count(_old_line)==1
_explicit_old=r'''_existing_specs = [
    ('rsna-knee-b-student-f0', ['student_f0.pt']),
    ('rsna-knee-b-student-f1234', [f'student_f{f}.pt' for f in range(1,5)]),
    ('rsna-knee-b-student-f01234-r2-convnext', [f'student_r2_convnext_tiny_f{f}.pt' for f in range(5)]),
    ('rsna-knee-b-student-f01234-r2-effv2', [f'student_r2_tf_efficientnetv2_s_f{f}.pt' for f in range(5)]),
]
_ckpts = []
for _slug, _names in _existing_specs:
    _mounts = [f'/kaggle/input/notebooks/gwey1025/{_slug}',f'/kaggle/input/{_slug}']
    _mount = next((p for p in _mounts if all(os.path.isfile(p+'/'+f) for f in _names)),None)
    if _mount is None:
        raise RuntimeError('Missing exact existing student family '+_slug)
    _ckpts.extend(_mount+'/'+f for f in _names)
_ckpts = sorted(_ckpts)'''
nb['cells'][19]['source']=_old_source.replace(_old_line,_explicit_old).splitlines(keepends=True)
nb['cells'][0]['source']=[
    '# RSNA Knee L2 0.962 Target Fold0\n\n',
    '**0.962 is an unverified target.** This candidate requires the new, actual L2 '
    'ConvNeXt-T fold-0 checkpoint from `gwey1025/rsna-knee-l2-smoke-20261015`. '
    'It keeps the audited incumbent family and blends the L2 family at an apriori 0.05 '
    'rank weight. Existing round-1/round-2 students retain their original identities. '
    'No missing L2 checkpoint may be substituted with an older student.\n\n',
    'The one-fold budgeted L2 training run is a smoke experiment, not the proposed '
    'complete five-fold student ensemble. Gold58 labels were excluded from gradients '
    'but their prior use in choosing label mixtures prevents an independent accuracy claim.\n',
    '\nThe quota-limited smoke checkpoint uses **four training windows per slot**, '
    'which differs from the original eight-window training recipe. Inference retains '
    'the original twelve windows, 256-pixel JPEG-quality-95 preprocessing and actual '
    'six acquisition slots. The final completed report-label generation, training '
    'source, completion receipt and checkpoint digest must all match their pins.\n',
    '\nL2 predictions use the exact same prepared tensors as the original 15-student '
    'loop, with a separate checkpoint list, raw prediction table and failure mask. '
    'The original 15-model mean is unchanged. A second GPU, when available, holds '
    'the sole L2 model; no second DICOM preprocessing pass is performed.\n',
]
preflight=('L2_EXPECTED_SOURCE = '+repr(EXPECTED)+'\nL2_CONTRACT_SOURCE = '+repr(CONTRACT_SOURCE)+
           '\nL2_KNEE_SOURCE = '+repr(KNEE_SOURCE)+'\nL2_PIN_IS_READY = '+repr(EXPECTED is not None)+"\n"
           "if not L2_PIN_IS_READY:\n"
           "    raise RuntimeError('Final, completely annotated L2 sources must be pinned by regenerating this notebook before execution')\n")
# Preserve the incumbent's prose cell. Insert executable configuration after all
# original-index transformations below; never place Python in a markdown cell.
setup=r'''# Separate actual L2 family; validate before any L2 model execution.
import gc, hashlib, importlib.util, json, os, time
exec(L2_CONTRACT_SOURCE, globals())
if L2_EXPECTED_SOURCE is None:
    raise RuntimeError('Final complete L2 label hashes must be pinned before this candidate may run')

def _l2_digest(path):
    digest = hashlib.sha256()
    with open(path,'rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

L2_W = 0.05
L2_FOLDS = [0]  # intentionally one budgeted smoke fold; never call this a five-fold result
L2_TRAINING_KERNEL = 'gwey1025/rsna-knee-l2-smoke-20261015'
_l2_mounts = ['/kaggle/input/notebooks/' + L2_TRAINING_KERNEL,
              '/kaggle/input/rsna-knee-l2-smoke-20261015']
_l2_root = next((p for p in _l2_mounts if os.path.isfile(p + '/student_l2_convnext_tiny_f0.pt')), None)
if _l2_root is None:
    raise RuntimeError('Actual L2 training output is required; no old-student substitution is permitted')
if time.time() - NB_T0 > 8.5 * 3600:
    raise RuntimeError('Insufficient remaining notebook budget for the selected L2 family')
_l2_checkpoints = [f'{_l2_root}/student_l2_convnext_tiny_f{f}.pt' for f in L2_FOLDS]
assert all(os.path.isfile(p) for p in _l2_checkpoints), 'The explicitly requested L2 folds are incomplete'
assert sorted(os.path.basename(p) for p in glob.glob(_l2_root+'/student_l2_convnext_tiny_f*.pt')) == ['student_l2_convnext_tiny_f0.pt']
_l2_completion = json.loads(open(_l2_root+'/completion_l2_f0.json').read())
_l2_manifest = json.loads(open(_l2_root+'/l2_training_manifest.json').read())
_l2_warm_start = next(p for p in _ckpts if os.path.basename(p)=='student_r2_convnext_tiny_f0.pt')
_l2_warm_start_hash = _l2_digest(_l2_warm_start)
_l2_knee_path = WORK/'l2_exact_knee.py'
_l2_knee_path.write_text(L2_KNEE_SOURCE)
assert _l2_digest(_l2_knee_path)==L2_EXPECTED_SOURCE['source_hashes']['knee.py']
_l2_spec = importlib.util.spec_from_file_location('l2_exact_knee',_l2_knee_path)
_l2_knee = importlib.util.module_from_spec(_l2_spec)
_l2_spec.loader.exec_module(_l2_knee)
assert _l2_knee.LABELS==list(_L)==L2_EXPECTED_SOURCE['targets'] and _l2_knee.N_SLOTS==_NS==6
assert (_RES,_K,_pp.MAX_SLICES)==(256,12,32)
for _n in range(3,33):
    assert np.array_equal(_wc(_n,12,False),_l2_knee.window_centres(_n,12,False)), 'Evaluation window selection differs from the trained source'
# Preserve the old family's device and keep just one L2 model on the second GPU.
# A single-GPU environment uses the same device without adding it to `models`.
_l2_device = torch.device('cuda:1' if torch.cuda.device_count()>1 else 'cuda:0')
_l2_models, _l2_sources = [], []
for _fold, _path in zip(L2_FOLDS, _l2_checkpoints):
    _ck = torch.load(_path, map_location='cpu', weights_only=False)
    _args, _prov = _ck['args'], _ck['provenance']
    _l2_checkpoint_hash = _l2_digest(_path)
    _l2_verified = verify_l2_smoke(_ck,_l2_completion,_l2_manifest,L2_EXPECTED_SOURCE,
                                 _l2_checkpoint_hash,_l2_warm_start_hash,list(_L))
    _model = _l2_knee.KneeNet(_args['backbone'], pretrained=False, dim=int(_args.get('dim',384)),
                  depth=int(_args.get('depth',2)), drop=float(_args.get('drop',.2)),
                  drop_path=float(_args.get('drop_path',.1)))
    _model.load_state_dict(_ck['model'], strict=True)
    _l2_models.append(_model.to(_l2_device,memory_format=torch.channels_last).eval())
    _l2_sources.append({'fold':_fold, 'file':os.path.basename(_path),
                        'sha256':_l2_checkpoint_hash,'verified':_l2_verified,
                        'args':_args, 'provenance':_prov})
    del _ck
_l2_predictions = {}
_l2_sidecar_seconds = 0.0
assert len(_l2_models)==1 and len(_ckpts)==15
print('ACTUAL_L2_VERIFIED',json.dumps(_l2_sources[0]['verified']),
      'sidecar_device',str(_l2_device),'old_student_count',len(_ckpts),flush=True)
'''
sidecar=r'''# Compute separate L2 probabilities from the same x/mask/pos, without decoding again.
if mask.any():
    _l2_step_started = time.perf_counter()
    _l2_x = torch.from_numpy(x)[None].to(_l2_device)
    _l2_m = torch.from_numpy(mask)[None].to(_l2_device)
    _l2_p = torch.from_numpy(pos)[None].to(_l2_device)
    with torch.autocast('cuda',dtype=torch.float16):
        _l2_prob = _l2_models[0](_l2_x,_l2_m,_l2_p,res=256).float().sigmoid()
    assert torch.isfinite(_l2_prob).all(), 'Nonfinite actual L2 sidecar predictions'
    _l2_predictions[st] = _l2_prob[0].cpu().numpy()
    _l2_sidecar_seconds += time.perf_counter()-_l2_step_started
    del _l2_x,_l2_m,_l2_p,_l2_prob
else:
    _l2_predictions[st] = np.full(len(_L),np.nan)
if (i+1)%100==0:
    print('ACTUAL_L2_SIDECAR_PROGRESS',i+1,len(_ids),round(_l2_sidecar_seconds,1),flush=True)
'''
stage=r'''# Apply the separate actual-L2 family only after the unchanged incumbent blend.
_l2_base = pd.read_csv(WORK/'submission.csv', dtype={'StudyInstanceUID':str})
_l2_base.to_csv(WORK/'submission_before_actual_l2.csv', index=False)
_l2_ids = _l2_base.StudyInstanceUID.tolist()
assert _l2_ids == _ids and list(_L) == _official_cols[1:]
assert set(_l2_predictions)==set(_l2_ids), 'Actual L2 sidecar study coverage differs'
assert all(np.asarray(v).shape==(len(_L),) for v in _l2_predictions.values())
del models,m,ck,_l2_models,_model
gc.collect()
for _device_index in range(torch.cuda.device_count()):
    with torch.cuda.device(_device_index):
        torch.cuda.empty_cache()
_l2_raw = pd.DataFrame([_l2_predictions[u] for u in _l2_ids],columns=_L,index=_l2_ids)
_l2_raw.index.name = 'StudyInstanceUID'
_l2_raw.to_csv(WORK/'actual_l2_test_raw.csv')
_l2_failed = _l2_raw.isna().any(axis=1).to_numpy()
assert _l2_failed.sum() <= .03*len(_l2_ids), 'More than 3% of studies failed in the actual L2 family'
_l2_out = _l2_base.copy()
for _c in _L:
    _rb = _l2_base[_c].rank(method='average',pct=True).to_numpy()
    _rs = _l2_raw[_c].rank(method='average',pct=True).to_numpy()
    _blend = np.where(_l2_failed,_rb,(1-L2_W)*_rb+L2_W*np.nan_to_num(_rs,nan=.5))
    _l2_out[_c] = pd.Series(_blend).rank(method='average',pct=True).to_numpy(dtype=np.float32)
assert _l2_out.columns.tolist()==_official_cols and _l2_out.StudyInstanceUID.tolist()==_official_ids
assert not _l2_out.StudyInstanceUID.duplicated().any()
_p = _l2_out[_L].to_numpy(dtype=np.float64)
assert np.isfinite(_p).all() and ((_p>=0)&(_p<=1)).all()
_tmp = WORK/'submission.actual_l2.tmp.csv'
_l2_out.to_csv(_tmp,index=False); os.replace(_tmp,WORK/'submission.csv')
_l2_receipt = {'candidate':'RSNA Knee L2 0.962 Target Fold0','actual_score':None,
               'incumbent_score':.956,'new_l2_folds':L2_FOLDS,'new_l2_weight':L2_W,
               'new_l2_sources':_l2_sources,'rows':len(_l2_ids),'l2_failed':int(_l2_failed.sum()),
               'existing_students_are_l2':False,'training_kernel':L2_TRAINING_KERNEL,
               'expected_l2_source':L2_EXPECTED_SOURCE,'l2_train_windows':4,'l2_eval_windows':12,
               'shared_prepared_tensors':True,'second_preprocessing_pass':False,
               'sidecar_device':str(_l2_device),'existing_student_checkpoints':15,
               'selected_csv_sha256':hashlib.sha256((WORK/'submission.csv').read_bytes()).hexdigest(),
               'seconds_l2':_l2_sidecar_seconds,'seconds_total':time.time()-NB_T0}
(WORK/'candidate_receipt.json').write_text(json.dumps(_l2_receipt,indent=2))
print('ACTUAL_L2_CANDIDATE_COMPLETE',json.dumps({k:v for k,v in _l2_receipt.items() if k!='new_l2_sources'}),flush=True)
'''
for source in (setup,sidecar,stage):
    ast.parse(source)
student_source=''.join(nb['cells'][19]['source'])
old_student_source=student_source
model_anchor='    models = []\n'
prediction_anchor='            preds[st] = pr[0].cpu().numpy() if mask.any() else np.full(len(_L), np.nan)\n'
assert student_source.count(model_anchor)==1 and student_source.count(prediction_anchor)==1
student_source=student_source.replace(model_anchor,''.join('    '+line+'\n' for line in setup.splitlines())+model_anchor)
student_source=student_source.replace(prediction_anchor,prediction_anchor+''.join('            '+line+'\n' for line in sidecar.splitlines()))
student_source=student_source.replace('Required existing-student stage failed; no alternate graph is submitted',
                                     'Required existing-student or actual-L2 sidecar stage failed; no alternate graph is submitted')
ast.parse(student_source)
# The old family's image preprocessing, mean probability and rank blend stay identical.
def relevant_nodes(source):
    tree=ast.parse(source)
    prep=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_prep')
    assignments={n.targets[0].id:ast.dump(n,include_attributes=False) for n in ast.walk(tree)
                 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
                 and n.targets[0].id in {'pr','rb','rs','bl'}}
    return ast.dump(prep,include_attributes=False),assignments
assert relevant_nodes(old_student_source)==relevant_nodes(student_source)
nb['cells'][19]['source']=student_source.splitlines(keepends=True)
nb['cells'].append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
                    'source':stage.splitlines(keepends=True)})
nb['cells'].insert(1,{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
                      'source':preflight.splitlines(keepends=True)})
out=HERE/'actual_l2_fold0';out.mkdir(exist_ok=True)
name='rsna-knee-l2-0-962-target-fold0.ipynb'
(out/name).write_text(json.dumps(nb,ensure_ascii=False,indent=1))
meta=json.loads((HERE/'kernel-metadata.json').read_text())
meta.update(id='gwey1025/rsna-knee-l2-0-962-target-fold0',title='RSNA Knee L2 0.962 Target Fold0',code_file=name)
meta['kernel_sources'] += ['gwey1025/rsna-knee-l2-smoke-20261015']
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
(out/'l2_source_pin.json').write_text(json.dumps(EXPECTED,indent=2))
(out/'l2_sidecar_build_receipt.json').write_text(json.dumps({
    'original_15_student_preprocessing_mean_rank_blend_unchanged':True,
    'new_l2_folds':[0],'new_l2_weight':.05,'shared_prepared_tensors':True,
    'second_preprocessing_pass':False,'l2_model_count':1,
    'setup_sha256':hashlib.sha256(setup.encode()).hexdigest(),
    'sidecar_sha256':hashlib.sha256(sidecar.encode()).hexdigest(),
    'separate_blend_sha256':hashlib.sha256(stage.encode()).hexdigest(),
},indent=2))
print('FINAL_L2_SOURCE_PIN', 'ready' if EXPECTED is not None else 'not yet complete; runtime intentionally blocked')
print(out/name)
