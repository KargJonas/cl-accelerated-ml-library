import { BinaryOps, MetaOps, ReduceOps, TernaryOps, UnaryOps, type Ops } from "./ops";

// enum UnaryOps { EXP2 = "EXP2", LOG2 = "LOG2", CAST = "CAST", BITCAST = "BITCAST", SIN = "SIN", SQRT = "SQRT", RECIP = "RECIP", NEG = "NEG" };
// enum BinaryOps { ADD = "ADD", MUL = "MUL", IDIV = "IDIV", MAX = "MAX", MOD = "MOD", CMPLT = "CMPLT", CMPNE = "CMPNE", XOR = "XOR", SHL = "SHL", SHR = "SHR", OR = "OR", AND = "AND", THREEFRY = "THREEFRY", SUB = "SUB" };
// enum TernaryOps { WHERE = "WHERE", MULACC = "MULACC" };
// enum ReduceOps { SUM = "SUM", PROD = "PROD", MAX = "MAX" };
// enum MetaOps { EMPTY = "EMPTY", CONST = "CONST", COPY = "COPY", CONTIGUOUS = "CONTIGUOUS", ASSIGN = "ASSIGN", VIEW = "VIEW" };

class Tensor {
    shape?: number[]; // todo: NOTE shape calculation will be deferred until execution
    op: Ops;
    parents: Tensor[];

    constructor(op: Ops = MetaOps.CONST, parents: Tensor[] = []) {
        this.op = op;
        this.parents = parents;
    }

    unary = (op: UnaryOps) => new Tensor(op, [this]);
    binary = (op: BinaryOps, other: Tensor) => new Tensor(op, [this, other]);
    ternary = (op: TernaryOps, a: Tensor, b: Tensor) => new Tensor(op, [this, a, b]);
    reduce = (op: ReduceOps) => new Tensor(op, [this]);

    add = (other: Tensor) => this.binary(BinaryOps.ADD, other);
    mul = (other: Tensor) => this.binary(BinaryOps.MUL, other);
    idiv = (other: Tensor) => this.binary(BinaryOps.IDIV, other);
    max = (other: Tensor) => this.binary(BinaryOps.MAX, other);

    exp2 = () => this.unary(UnaryOps.EXP2);
    log2 = () => this.unary(UnaryOps.LOG2);
    cast = () => this.unary(UnaryOps.CAST); // this op needs an additional type argument
}
