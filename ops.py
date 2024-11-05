from enum import IntEnum


class UnaryOps(IntEnum):
    EXP2 = 100; LOG2 = 101; CAST = 102; BITCAST = 103; SIN = 104; SQRT = 105; RECIP = 106; NEG = 107

class BinaryOps(IntEnum):
    ADD = 200; MUL = 201; IDIV = 202; MAX = 203; MOD = 204; CMPLT = 205; CMPNE = 206; XOR = 207
    SHL = 208; SHR = 209; OR = 210; AND = 211; THREEFRY = 212; SUB = 213

class TernaryOps(IntEnum):
    WHERE = 300; MULACC = 301

class ReduceOps(IntEnum):
    SUM = 400; PROD = 401; MAX = 402

class MetaOps(IntEnum):
    EMPTY = 500; CONST = 501; COPY = 502; CONTIGUOUS = 503; ASSIGN = 504; VIEW = 505

Op = (*UnaryOps, *BinaryOps, *TernaryOps, *ReduceOps, *MetaOps)