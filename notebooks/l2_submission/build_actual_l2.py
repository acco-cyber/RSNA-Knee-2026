"""Prepare an explicit real-L2 family; do not replace it with old students."""
import ast, copy, json
from pathlib import Path
HERE=Path(__file__).parent
nb=json.loads((HERE/'rsna-knee-0962-target-crop224-audit.ipynb').read_text())
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
]
stage=r'''# Separate actual L2 family; explicit fold names and training provenance.
import gc, hashlib, json, os, time
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
_l2_base = pd.read_csv(WORK/'submission.csv', dtype={'StudyInstanceUID':str})
_l2_base.to_csv(WORK/'submission_before_actual_l2.csv', index=False)
_l2_ids = _l2_base.StudyInstanceUID.tolist()
assert _l2_ids == _ids and list(_L) == _official_cols[1:]
del models
gc.collect(); torch.cuda.empty_cache()
_l2_models, _l2_sources = [], []
for _fold, _path in zip(L2_FOLDS, _l2_checkpoints):
    _ck = torch.load(_path, map_location='cpu', weights_only=False)
    _args, _prov = _ck['args'], _ck['provenance']
    assert _args['mode'] == 'l2' and int(_args['fold']) == _fold, 'Incorrect L2 checkpoint mode or fold'
    assert _args['backbone'] == 'convnext_tiny' and int(_args['res']) == 256 and int(_args['k_eval']) == 12
    assert _prov['targets'] == list(_L), 'L2 target order differs'
    assert _prov['gold_excluded_from_gradient_training'] is True, 'Gold studies leaked into L2 gradient training'
    assert int(_prov.get('completed_epochs',0)) >= 1, 'L2 checkpoint has no completed training epoch'
    assert 'labels_l2' in _prov['source_hashes'] or 'labels_l2.csv' in _prov['source_hashes'], 'L2 label source hash missing'
    _model = _Net(_args['backbone'], pretrained=False, dim=int(_args.get('dim',384)),
                  depth=int(_args.get('depth',2)), drop=float(_args.get('drop',.2)),
                  drop_path=float(_args.get('drop_path',.1)))
    _model.load_state_dict(_ck['model'], strict=True)
    _l2_models.append(_model.cuda().to(memory_format=torch.channels_last).eval())
    _l2_sources.append({'fold':_fold, 'file':os.path.basename(_path),
                        'sha256':hashlib.sha256(open(_path,'rb').read()).hexdigest(),
                        'args':_args, 'provenance':_prov})
    del _ck
_l2_t0 = time.time()
_l2_predictions = {}
with Pool(4) as _pool, torch.no_grad():
    for _i, (_sid, _x, _mask, _pos) in enumerate(_pool.imap(_prep,_l2_ids,chunksize=1)):
        if not _mask.any():
            _l2_predictions[_sid] = np.full(len(_L),np.nan)
            continue
        _xt = torch.from_numpy(_x)[None].cuda()
        _mt = torch.from_numpy(_mask)[None].cuda(); _pt = torch.from_numpy(_pos)[None].cuda()
        with torch.autocast('cuda',dtype=torch.float16):
            _probs = torch.stack([m(_xt,_mt,_pt,res=256).float().sigmoid() for m in _l2_models]).mean(0)
        assert torch.isfinite(_probs).all(), 'Nonfinite actual L2 predictions'
        _l2_predictions[_sid] = _probs[0].cpu().numpy()
        if (_i+1)%100==0:
            print('ACTUAL_L2_PROGRESS',_i+1,len(_l2_ids),round(time.time()-_l2_t0),flush=True)
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
               'selected_csv_sha256':hashlib.sha256((WORK/'submission.csv').read_bytes()).hexdigest(),
               'seconds_l2':time.time()-_l2_t0,'seconds_total':time.time()-NB_T0}
(WORK/'candidate_receipt.json').write_text(json.dumps(_l2_receipt,indent=2))
print('ACTUAL_L2_CANDIDATE_COMPLETE',json.dumps({k:v for k,v in _l2_receipt.items() if k!='new_l2_sources'}),flush=True)
'''
ast.parse(stage)
nb['cells'].append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
                    'source':stage.splitlines(keepends=True)})
out=HERE/'actual_l2_fold0';out.mkdir(exist_ok=True)
name='rsna-knee-l2-0-962-target-fold0.ipynb'
(out/name).write_text(json.dumps(nb,ensure_ascii=False,indent=1))
meta=json.loads((HERE/'kernel-metadata.json').read_text())
meta.update(id='gwey1025/rsna-knee-l2-0-962-target-fold0',title='RSNA Knee L2 0.962 Target Fold0',code_file=name)
meta['kernel_sources'] += ['gwey1025/rsna-knee-l2-smoke-20261015']
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
print(out/name)
