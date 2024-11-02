from enum import Enum, IntEnum, auto
from typing import Union


class FastEnum(IntEnum):
  def __str__(self): return Enum.__str__(self)

class UnaryOps(FastEnum):
  """A -> A (elementwise)"""
  EXP2 = auto(); LOG2 = auto(); CAST = auto(); BITCAST = auto(); SIN = auto(); SQRT = auto(); RECIP = auto(); NEG = auto() # noqa: E702

class BinaryOps(FastEnum):
  """A + A -> A (elementwise)"""
  ADD = auto(); MUL = auto(); IDIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto(); CMPNE = auto(); XOR = auto() # noqa: E702
  SHL = auto(); SHR = auto(); OR = auto(); AND = auto(); THREEFRY = auto(); SUB = auto() # noqa: E702

class TernaryOps(FastEnum):
  """A + A + A -> A (elementwise)"""
  WHERE = auto(); MULACC = auto() # noqa: E702

class ReduceOps(FastEnum):
  """A -> B (reduce)"""
  SUM = auto(); PROD = auto(); MAX = auto() # noqa: E702

class MetaOps(FastEnum):
  EMPTY = auto(); CONST = auto(); COPY = auto(); CONTIGUOUS = auto(); ASSIGN = auto(); VIEW = auto() # noqa: E702

Op = Union[UnaryOps, BinaryOps, ReduceOps, MetaOps, TernaryOps]
