// optimizer.hpp - AST-level optimizations: constant folding, algebraic simplification,
// short-circuit simplification and dead-branch elimination.
//
// All expressions in the language are side-effect free, so these rewrites are always safe.
#pragma once
#include <climits>
#include <cmath>
#include <cstdint>

#include "ast.hpp"

struct OptStats {
    int folded = 0;            // constant expressions evaluated at compile time
    int simplified = 0;        // algebraic identities removed (x+0, x*1, true && c ...)
    int branches_removed = 0;  // if/while statements removed because the condition is constant
    int total() const { return folded + simplified + branches_removed; }
};

class Optimizer {
public:
    OptStats stats;

    void run(Kernel& k) { k.body = opt_stmt(std::move(k.body)); }

private:
    static int32_t wrap(long long v) { return static_cast<int32_t>(static_cast<uint32_t>(static_cast<uint64_t>(v))); }

    ExprPtr int_lit(const Expr& at, long long v) {
        auto e = make_expr(Expr::IntLit, at.line, at.col);
        e->type = Type::Int;
        e->ival = wrap(v);
        return e;
    }
    ExprPtr float_lit(const Expr& at, float v) {
        auto e = make_expr(Expr::FloatLit, at.line, at.col);
        e->type = Type::Float;
        e->fval = v;
        return e;
    }
    ExprPtr bool_lit(const Expr& at, bool v) {
        auto e = make_expr(Expr::BoolLit, at.line, at.col);
        e->type = Type::Bool;
        e->bval = v;
        return e;
    }

    static bool is_int(const Expr& e, long long v) { return e.kind == Expr::IntLit && e.ival == v; }
    static bool is_flt(const Expr& e, float v) { return e.kind == Expr::FloatLit && e.fval == v; }

    ExprPtr fold(ExprPtr e) {
        if (e->lhs) e->lhs = fold(std::move(e->lhs));
        if (e->rhs) e->rhs = fold(std::move(e->rhs));

        switch (e->kind) {
            case Expr::Unary: return fold_unary(std::move(e));
            case Expr::Cast: return fold_cast(std::move(e));
            case Expr::Binary: return fold_binary(std::move(e));
            default: return e;
        }
    }

    ExprPtr fold_unary(ExprPtr e) {
        const Expr& x = *e->lhs;
        if (e->op == "-") {
            if (x.kind == Expr::IntLit) {
                stats.folded++;
                return int_lit(*e, -x.ival);
            }
            if (x.kind == Expr::FloatLit) {
                stats.folded++;
                return float_lit(*e, -x.fval);
            }
        } else if (e->op == "!" && x.kind == Expr::BoolLit) {
            stats.folded++;
            return bool_lit(*e, !x.bval);
        }
        return e;
    }

    ExprPtr fold_cast(ExprPtr e) {
        const Expr& x = *e->lhs;
        if (e->type == Type::Float && x.kind == Expr::IntLit) {
            stats.folded++;
            return float_lit(*e, static_cast<float>(x.ival));
        }
        if (e->type == Type::Int && x.kind == Expr::FloatLit) {
            // Only fold when the conversion is well defined in C++ (no NaN / overflow).
            if (std::isfinite(x.fval) && x.fval > -2147483648.0f && x.fval < 2147483648.0f) {
                stats.folded++;
                return int_lit(*e, static_cast<long long>(x.fval));  // truncates toward zero, like cvt.rzi
            }
        }
        if (e->type == x.type) {  // redundant cast
            stats.simplified++;
            return std::move(e->lhs);
        }
        return e;
    }

    ExprPtr fold_binary(ExprPtr e) {
        const std::string& op = e->op;
        Expr& l = *e->lhs;
        Expr& r = *e->rhs;

        // ---- logical operators (short-circuit simplification)
        if (op == "&&" || op == "||") {
            bool isAnd = op == "&&";
            if (l.kind == Expr::BoolLit) {
                stats.simplified++;
                if (isAnd) return l.bval ? std::move(e->rhs) : bool_lit(*e, false);
                return l.bval ? bool_lit(*e, true) : std::move(e->rhs);
            }
            if (r.kind == Expr::BoolLit) {
                stats.simplified++;
                if (isAnd) return r.bval ? std::move(e->lhs) : bool_lit(*e, false);
                return r.bval ? bool_lit(*e, true) : std::move(e->lhs);
            }
            return e;
        }

        // ---- int op int
        if (l.kind == Expr::IntLit && r.kind == Expr::IntLit) {
            long long a = l.ival, b = r.ival;
            if (op == "+") { stats.folded++; return int_lit(*e, a + b); }
            if (op == "-") { stats.folded++; return int_lit(*e, a - b); }
            if (op == "*") { stats.folded++; return int_lit(*e, a * b); }
            if ((op == "/" || op == "%") && b != 0 && !(a == INT_MIN && b == -1)) {
                stats.folded++;
                return int_lit(*e, op == "/" ? a / b : a % b);  // C semantics: truncate toward zero, like div.s32
            }
            if (op == "<") { stats.folded++; return bool_lit(*e, a < b); }
            if (op == "<=") { stats.folded++; return bool_lit(*e, a <= b); }
            if (op == ">") { stats.folded++; return bool_lit(*e, a > b); }
            if (op == ">=") { stats.folded++; return bool_lit(*e, a >= b); }
            if (op == "==") { stats.folded++; return bool_lit(*e, a == b); }
            if (op == "!=") { stats.folded++; return bool_lit(*e, a != b); }
            return e;
        }

        // ---- float op float
        if (l.kind == Expr::FloatLit && r.kind == Expr::FloatLit) {
            float a = l.fval, b = r.fval;
            float v = 0;
            bool arith = true;
            if (op == "+") v = a + b;
            else if (op == "-") v = a - b;
            else if (op == "*") v = a * b;
            else if (op == "/") v = a / b;
            else arith = false;
            if (arith) {
                if (std::isfinite(v)) {  // never bake inf/NaN into the output
                    stats.folded++;
                    return float_lit(*e, v);
                }
                return e;
            }
            if (op == "<") { stats.folded++; return bool_lit(*e, a < b); }
            if (op == "<=") { stats.folded++; return bool_lit(*e, a <= b); }
            if (op == ">") { stats.folded++; return bool_lit(*e, a > b); }
            if (op == ">=") { stats.folded++; return bool_lit(*e, a >= b); }
            if (op == "==") { stats.folded++; return bool_lit(*e, a == b); }
            if (op == "!=") { stats.folded++; return bool_lit(*e, a != b); }
            return e;
        }

        // ---- algebraic identities
        if (e->type == Type::Int) {
            if (op == "+" && is_int(r, 0)) { stats.simplified++; return std::move(e->lhs); }
            if (op == "+" && is_int(l, 0)) { stats.simplified++; return std::move(e->rhs); }
            if (op == "-" && is_int(r, 0)) { stats.simplified++; return std::move(e->lhs); }
            if (op == "*" && is_int(r, 1)) { stats.simplified++; return std::move(e->lhs); }
            if (op == "*" && is_int(l, 1)) { stats.simplified++; return std::move(e->rhs); }
            if (op == "/" && is_int(r, 1)) { stats.simplified++; return std::move(e->lhs); }
            if (op == "*" && (is_int(l, 0) || is_int(r, 0))) { stats.simplified++; return int_lit(*e, 0); }
        } else if (e->type == Type::Float) {
            // Only identities that are exact for every float (including -0.0 and NaN).
            if (op == "*" && is_flt(r, 1.0f)) { stats.simplified++; return std::move(e->lhs); }
            if (op == "*" && is_flt(l, 1.0f)) { stats.simplified++; return std::move(e->rhs); }
            if (op == "/" && is_flt(r, 1.0f)) { stats.simplified++; return std::move(e->lhs); }
        }
        return e;
    }

    StmtPtr empty_block(const Stmt& at) {
        auto b = std::make_unique<Stmt>();
        b->kind = Stmt::Block;
        b->line = at.line;
        b->col = at.col;
        return b;
    }

    StmtPtr opt_stmt(StmtPtr s) {
        switch (s->kind) {
            case Stmt::Decl:
                if (s->expr) s->expr = fold(std::move(s->expr));
                return s;
            case Stmt::Assign:
                s->expr = fold(std::move(s->expr));
                return s;
            case Stmt::Store:
                s->index = fold(std::move(s->index));
                s->expr = fold(std::move(s->expr));
                return s;
            case Stmt::Block:
                for (auto& c : s->body) c = opt_stmt(std::move(c));
                return s;
            case Stmt::Return: return s;
            case Stmt::If: {
                s->expr = fold(std::move(s->expr));
                if (s->expr->kind == Expr::BoolLit) {  // condition known at compile time
                    stats.branches_removed++;
                    if (s->expr->bval) return opt_stmt(std::move(s->thenS));
                    if (s->elseS) return opt_stmt(std::move(s->elseS));
                    return empty_block(*s);
                }
                s->thenS = opt_stmt(std::move(s->thenS));
                if (s->elseS) s->elseS = opt_stmt(std::move(s->elseS));
                return s;
            }
            case Stmt::While: {
                s->expr = fold(std::move(s->expr));
                if (s->expr->kind == Expr::BoolLit && !s->expr->bval) {  // loop never runs
                    stats.branches_removed++;
                    return empty_block(*s);
                }
                s->thenS = opt_stmt(std::move(s->thenS));
                return s;
            }
        }
        return s;
    }
};
