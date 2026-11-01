"""Direct attribute metadata only: no tensor payload, recursion into weights/configs, or CUDA work."""
def describe_tensor(value):
    if not all(hasattr(value,n) for n in ('shape','dtype','device','data_ptr','stride','numel')):return None
    return dict(class_name=type(value).__name__,shape=list(value.shape),dtype=str(value.dtype),device=str(value.device),
        pointer=value.data_ptr(),stride=list(value.stride()),numel=value.numel(),element_size=value.element_size(),contiguous=value.is_contiguous())

def direct_metadata(obj):
    result=dict(class_name=type(obj).__name__,tensors={},paired_buffers={},scalars={},errors=[])
    for name,value in vars(obj).items():
        try:
            tensor=describe_tensor(value)
            if tensor is not None:result['tensors'][name]=tensor;continue
            # Scalar geometry, never arbitrary config/URL/string contents.
            if value is None or type(value) in (bool,int,float):result['scalars'][name]=value;continue
            if isinstance(value,tuple) and all(type(x) in (bool,int) for x in value):result['scalars'][name]=list(value);continue
            fields=vars(value) if hasattr(value,'__dict__') else {}
            if 'gpu' in fields or 'cpu' in fields:
                pair={side:describe_tensor(fields[side]) for side in ('cpu','gpu') if side in fields}
                if any(x is not None for x in pair.values()):result['paired_buffers'][name]=pair
        except Exception as exc:result['errors'].append(dict(attribute=name,error=repr(exc)))
    return result
