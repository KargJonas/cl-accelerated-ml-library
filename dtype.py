from enum import Enum

class DType(Enum):
    F16 = "f16"; F32 = "f32"

type_sizes = {
    DType.F16: 2,
    DType.F32: 4,
}
