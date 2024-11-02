export enum UnaryOps { EXP2 = "EXP2", LOG2 = "LOG2", CAST = "CAST", BITCAST = "BITCAST", SIN = "SIN", SQRT = "SQRT", RECIP = "RECIP", NEG = "NEG" };
export enum BinaryOps { ADD = "ADD", MUL = "MUL", IDIV = "IDIV", MAX = "MAX", MOD = "MOD", CMPLT = "CMPLT", CMPNE = "CMPNE", XOR = "XOR", SHL = "SHL", SHR = "SHR", OR = "OR", AND = "AND", THREEFRY = "THREEFRY", SUB = "SUB" };
export enum TernaryOps { WHERE = "WHERE", MULACC = "MULACC" };
export enum ReduceOps { SUM = "SUM", PROD = "PROD", MAX = "MAX" };
export enum MetaOps { EMPTY = "EMPTY", CONST = "CONST", COPY = "COPY", CONTIGUOUS = "CONTIGUOUS", ASSIGN = "ASSIGN", VIEW = "VIEW" };
export type Ops = UnaryOps | BinaryOps | TernaryOps | ReduceOps | MetaOps;