"""Inspect torch.save ZIP metadata and finite weights without importing PyTorch.

Only explicit data constructors and inert tensor/storage placeholders are
accepted by the pickle reader. This does not execute a model or establish
state_dict compatibility; real GPU inference remains required.
"""
import argparse
from collections import OrderedDict
import hashlib
import io
import json
import pickle
from pathlib import Path
import zipfile

import numpy as np


def digest(path):
    result=hashlib.sha256()
    with open(path,'rb') as source:
        for block in iter(lambda:source.read(8*1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def rebuild_tensor(storage,offset,shape,stride,*unused):
    return {'storage':storage,'offset':int(offset),'shape':tuple(shape),'stride':tuple(stride)}


class MetadataReader(pickle.Unpickler):
    def __init__(self,data):
        super().__init__(io.BytesIO(data));self.storages={}

    def find_class(self,module,name):
        if module=='collections' and name=='OrderedDict':return OrderedDict
        if module=='builtins' and name=='set':return set
        if module=='torch' and name=='Size':return tuple
        if module=='torch' and (name.endswith('Storage') or name in
                              {'float32','float64','float16','bfloat16','int64','int32','int16','int8','uint8','bool'}):
            return name
        if module=='torch._utils' and name in {'_rebuild_tensor','_rebuild_tensor_v2','_rebuild_tensor_v3'}:
            return rebuild_tensor
        raise pickle.UnpicklingError(f'Unapproved constructor in checkpoint metadata: {module}.{name}')

    def persistent_load(self,value):
        if not isinstance(value,tuple) or len(value)!=5 or value[0]!='storage':
            raise pickle.UnpicklingError('Unsupported persistent tensor storage descriptor')
        _,kind,key,location,numel=value
        storage={'type':kind,'key':str(key),'location':str(location),'numel':int(numel)}
        old=self.storages.setdefault(storage['key'],storage)
        if old!=storage:raise pickle.UnpicklingError('Inconsistent storage metadata')
        return storage


def inspect(path):
    with zipfile.ZipFile(path) as archive:
        choices=[name for name in archive.namelist() if name.endswith('/data.pkl')]
        if len(choices)!=1:raise ValueError('Expected one torch.save metadata entry')
        name=choices[0];prefix=name.rsplit('/',1)[0]
        reader=MetadataReader(archive.read(name));checkpoint=reader.load()
        if not isinstance(checkpoint,dict) or set(checkpoint)!={'model','args','provenance'}:
            raise ValueError('Unexpected L2 checkpoint top-level fields')
        order=archive.read(prefix+'/byteorder').decode() if prefix+'/byteorder' in archive.namelist() else 'little'
        if order not in {'little','big'}:raise ValueError('Invalid tensor byte order')
        endian='<' if order=='little' else '>'
        types={'FloatStorage':'f4','DoubleStorage':'f8','HalfStorage':'f2',
               'LongStorage':'i8','IntStorage':'i4','ShortStorage':'i2',
               'CharStorage':'i1','ByteStorage':'u1','BoolStorage':'?','BFloat16Storage':'u2'}
        storage_bytes=0;floating_elements=0
        for key,storage in reader.storages.items():
            kind=storage['type']
            if kind not in types:raise ValueError('Unsupported storage type: '+str(kind))
            dtype=np.dtype(endian+types[kind]);entry=prefix+'/data/'+key
            expected_bytes=storage['numel']*dtype.itemsize
            if archive.getinfo(entry).file_size!=expected_bytes:
                raise ValueError('Tensor storage length differs: '+key)
            with archive.open(entry) as source:
                for block in iter(lambda:source.read(1024*1024),b''):
                    values=np.frombuffer(block,dtype=dtype)
                    if kind=='BFloat16Storage':
                        if ((values&0x7f80)==0x7f80).any():raise ValueError('Nonfinite bfloat16 tensor storage: '+key)
                    elif dtype.kind=='f' and not np.isfinite(values).all():
                        raise ValueError('Nonfinite floating tensor storage: '+key)
            storage_bytes+=expected_bytes
            if dtype.kind=='f' or kind=='BFloat16Storage':floating_elements+=storage['numel']
        shapes={key:list(value['shape']) for key,value in checkpoint['model'].items()}
        return checkpoint,{'file':Path(path).name,'checkpoint_sha256':digest(path),
                           'args':checkpoint['args'],'provenance':checkpoint['provenance'],
                           'tensor_entries':len(shapes),'tensor_storage_bytes':storage_bytes,
                           'floating_storage_elements':floating_elements,'all_stored_weights_finite':True,
                           'tensor_shapes_sha256':hashlib.sha256(json.dumps(shapes,sort_keys=True).encode()).hexdigest(),
                           'gpu_state_dict_load_or_inference_verified':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    _,report=inspect(args.checkpoint)
    if args.output:args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
