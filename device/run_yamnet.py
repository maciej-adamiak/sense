import argparse
import csv
import ctypes
import sys
from pathlib import Path

import numpy as np

LIB_CANDIDATES = [
    '/mnt/system/lib/libcviruntime.so',
    '/mnt/system/usr/lib/libcviruntime.so',
    '/opt/cvitek/lib/libcviruntime.so',
    '/mnt/tpu/tpu-sdk/lib/libcviruntime.so',
]
LIB_PATH = next((p for p in LIB_CANDIDATES if Path(p).exists()), None)
if not LIB_PATH:
    sys.exit('libcviruntime.so not found in known locations; check the SDK install')

rt = ctypes.CDLL(LIB_PATH)


class CviTensor(ctypes.Structure):
    _fields_ = [
        ('name', ctypes.c_char_p),
        ('shape', ctypes.c_int32 * 4),
        ('fmt', ctypes.c_int32),
        ('count', ctypes.c_size_t),
        ('mem_size', ctypes.c_size_t),
        ('sys_mem', ctypes.c_void_p),
    ]


MODEL_HANDLE = ctypes.c_void_p

rt.CVI_NN_RegisterModel.argtypes = [ctypes.c_char_p, ctypes.POINTER(MODEL_HANDLE)]
rt.CVI_NN_RegisterModel.restype = ctypes.c_int

rt.CVI_NN_GetInputOutputTensors.argtypes = [
    MODEL_HANDLE,
    ctypes.POINTER(ctypes.POINTER(CviTensor)),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.POINTER(CviTensor)),
    ctypes.POINTER(ctypes.c_int32),
]
rt.CVI_NN_GetInputOutputTensors.restype = ctypes.c_int

rt.CVI_NN_Forward.argtypes = [
    MODEL_HANDLE,
    ctypes.POINTER(CviTensor),
    ctypes.c_int32,
    ctypes.POINTER(CviTensor),
    ctypes.c_int32,
]
rt.CVI_NN_Forward.restype = ctypes.c_int

rt.CVI_NN_TensorPtr.argtypes = [ctypes.POINTER(CviTensor)]
rt.CVI_NN_TensorPtr.restype = ctypes.c_void_p
rt.CVI_NN_TensorSize.argtypes = [ctypes.POINTER(CviTensor)]
rt.CVI_NN_TensorSize.restype = ctypes.c_size_t
rt.CVI_NN_TensorCount.argtypes = [ctypes.POINTER(CviTensor)]
rt.CVI_NN_TensorCount.restype = ctypes.c_size_t
rt.CVI_NN_CleanupModel.argtypes = [MODEL_HANDLE]


def load_input(path: Path, expected_bytes: int) -> np.ndarray:
    if path.suffix == '.npz':
        npz = np.load(path)
        arr = npz[npz.files[0]].astype(np.float32, copy=False)
    elif path.suffix in ('.bin', '.raw'):
        arr = np.fromfile(path, dtype=np.float32)
    else:
        sys.exit(f'unsupported input format: {path.suffix}')

    if arr.nbytes != expected_bytes:
        sys.exit(f'input size {arr.nbytes} != model expects {expected_bytes}')

    return np.ascontiguousarray(arr)


def load_labels(path: Path) -> list[str]:
    labels = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            labels.append(row[2] if len(row) >= 3 else row[-1])

    return labels


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/root/yamnet/yamnet.cvimodel')
    ap.add_argument('--labels', default='/root/yamnet/yamnet_class_map.csv')
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('input', help='path to .bin (raw float32) or .npz')
    args = ap.parse_args()

    handle = MODEL_HANDLE()
    if rt.CVI_NN_RegisterModel(args.model.encode(), ctypes.byref(handle)) != 0:
        sys.exit('CVI_NN_RegisterModel failed')

    in_ptr = ctypes.POINTER(CviTensor)()
    out_ptr = ctypes.POINTER(CviTensor)()
    in_num = ctypes.c_int32()
    out_num = ctypes.c_int32()
    rt.CVI_NN_GetInputOutputTensors(
        handle,
        ctypes.byref(in_ptr),
        ctypes.byref(in_num),
        ctypes.byref(out_ptr),
        ctypes.byref(out_num),
    )

    in_bytes = rt.CVI_NN_TensorSize(ctypes.byref(in_ptr[0]))
    arr = load_input(Path(args.input), in_bytes)

    dst = rt.CVI_NN_TensorPtr(ctypes.byref(in_ptr[0]))
    ctypes.memmove(dst, arr.ctypes.data, in_bytes)

    if rt.CVI_NN_Forward(handle, in_ptr, in_num, out_ptr, out_num) != 0:
        sys.exit('CVI_NN_Forward failed')

    out_count = rt.CVI_NN_TensorCount(ctypes.byref(out_ptr[0]))
    out_ptr_raw = rt.CVI_NN_TensorPtr(ctypes.byref(out_ptr[0]))
    logits = np.ctypeslib.as_array((ctypes.c_float * out_count).from_address(out_ptr_raw)).copy()

    probs = softmax(logits)
    top = probs.argsort()[-args.topk :][::-1]
    labels = load_labels(Path(args.labels))

    print(f'model:  {Path(args.model).name}')
    print(f'input:  {Path(args.input).name}')
    print(f'top-{args.topk}:')

    for i in top:
        print(f'  {i:3d}  {probs[i]:.4f}  {labels[i]}')

    rt.CVI_NN_CleanupModel(handle)


if __name__ == '__main__':
    main()
