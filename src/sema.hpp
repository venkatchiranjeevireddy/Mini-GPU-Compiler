// sema.hpp - semantic analysis: scoping, type checking, implicit int->float conversion.
//
// After this pass every Expr has a valid `type`, every Var/Index/Decl/Assign/Store has a symbol id,
// and implicit int->float conversions have been made explicit as Cast nodes.
#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ast.hpp"
#include "common.hpp"

class Sema {
public:
    void check_program(Program& p) {
        std::unordered_set<std::string> seen;
        for (auto& k : p.kernels) {
            if (!seen.insert(k.name).second)
                throw CompileError(k.line, k.col, "kernel '" + k.name + "' is defined more than once");
            check_kernel(k);
        }
    }

private:
    Kernel* k_ = nullptr;
    std::vector<std::unordered_map<std::string, int>> scopes_;

    int declare(const std::string& name, Type t, int line, int col, bool isParam = false, int paramIndex = -1) {
        if (scopes_.back().count(name))
            throw CompileError(line, col, "'" + name + "' is already declared in this scope");
        int id = static_cast<int>(k_->symbols.size());
        k_->symbols.push_back({name, t, isParam, paramIndex});
        scopes_.back()[name] = id;
        return id;
    }

    int lookup(const std::string& name, int line, int col) {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto f = it->find(name);
            if (f != it->end()) return f->second;
        }
        throw CompileError(line, col, "use of undeclared variable '" + name + "'");
    }

    void check_kernel(Kernel& k) {
        k_ = &k;
        scopes_.clear();
        scopes_.emplace_back();
        for (size_t i = 0; i < k.params.size(); i++) {
            auto& p = k.params[i];
            p.sym = declare(p.name, p.type, p.line, p.col, true, static_cast<int>(i));
        }
        check_stmt(*k.body);
        scopes_.clear();
    }

    // Wrap `e` in an implicit conversion to `to`, or fail with a helpful message.
    ExprPtr coerce(ExprPtr e, Type to, const char* what) {
        if (e->type == to) return e;
        if (e->type == Type::Int && to == Type::Float) {
            auto c = make_expr(Expr::Cast, e->line, e->col);
            c->type = Type::Float;
            c->lhs = std::move(e);
            return c;
        }
        std::string msg = std::string("cannot use ") + type_name(e->type) + " value as " + type_name(to) + " in " + what;
        if (e->type == Type::Float && to == Type::Int) msg += " (use an explicit cast: (int) expr)";
        throw CompileError(e->line, e->col, msg);
    }

    void check_expr(ExprPtr& e) {
        switch (e->kind) {
            case Expr::IntLit: e->type = Type::Int; return;
            case Expr::FloatLit: e->type = Type::Float; return;
            case Expr::BoolLit: e->type = Type::Bool; return;
            case Expr::Special: e->type = Type::Int; return;
            case Expr::Var: {
                e->sym = lookup(e->name, e->line, e->col);
                Type t = k_->symbols[static_cast<size_t>(e->sym)].type;
                if (is_ptr(t))
                    throw CompileError(e->line, e->col, "pointer '" + e->name + "' can only be used with an index, like " + e->name + "[i]");
                e->type = t;
                return;
            }
            case Expr::Index: {
                e->sym = lookup(e->name, e->line, e->col);
                Type t = k_->symbols[static_cast<size_t>(e->sym)].type;
                if (!is_ptr(t)) throw CompileError(e->line, e->col, "'" + e->name + "' is not an array/pointer, cannot index it");
                check_expr(e->rhs);
                if (e->rhs->type != Type::Int)
                    throw CompileError(e->rhs->line, e->rhs->col, std::string("array index must be int, found ") + type_name(e->rhs->type));
                e->type = elem_type(t);
                return;
            }
            case Expr::Cast: {
                check_expr(e->lhs);
                if (!is_numeric(e->lhs->type))
                    throw CompileError(e->line, e->col, std::string("cannot cast ") + type_name(e->lhs->type));
                return;  // e->type already holds the target type
            }
            case Expr::Unary: {
                check_expr(e->lhs);
                if (e->op == "-") {
                    if (!is_numeric(e->lhs->type)) throw CompileError(e->line, e->col, "unary '-' needs an int or float operand");
                    e->type = e->lhs->type;
                } else {  // '!'
                    if (e->lhs->type != Type::Bool)
                        throw CompileError(e->line, e->col, "'!' needs a condition (comparison), e.g. !(x < 3)");
                    e->type = Type::Bool;
                }
                return;
            }
            case Expr::Binary: check_binary(e); return;
        }
    }

    void check_binary(ExprPtr& e) {
        check_expr(e->lhs);
        check_expr(e->rhs);
        const std::string& op = e->op;
        if (op == "&&" || op == "||") {
            if (e->lhs->type != Type::Bool || e->rhs->type != Type::Bool)
                throw CompileError(e->line, e->col, "'" + op + "' needs conditions on both sides (comparisons)");
            e->type = Type::Bool;
            return;
        }
        if (!is_numeric(e->lhs->type) || !is_numeric(e->rhs->type))
            throw CompileError(e->line, e->col, "operator '" + op + "' needs int or float operands, found " +
                                                     type_name(e->lhs->type) + " and " + type_name(e->rhs->type));
        Type common = (e->lhs->type == Type::Float || e->rhs->type == Type::Float) ? Type::Float : Type::Int;
        if (op == "%" && common != Type::Int) throw CompileError(e->line, e->col, "'%' only works on int operands");
        e->lhs = coerce(std::move(e->lhs), common, "expression");
        e->rhs = coerce(std::move(e->rhs), common, "expression");
        bool isCmp = op == "<" || op == "<=" || op == ">" || op == ">=" || op == "==" || op == "!=";
        e->type = isCmp ? Type::Bool : common;
    }

    void check_stmt(Stmt& s) {
        switch (s.kind) {
            case Stmt::Block:
                scopes_.emplace_back();
                for (auto& c : s.body) check_stmt(*c);
                scopes_.pop_back();
                return;
            case Stmt::Decl:
                if (s.expr) {
                    check_expr(s.expr);
                    s.expr = coerce(std::move(s.expr), s.declType, "initializer");
                }
                s.sym = declare(s.name, s.declType, s.line, s.col);  // declared after init: `int x = x;` is an error
                return;
            case Stmt::Assign: {
                s.sym = lookup(s.name, s.line, s.col);
                Type t = k_->symbols[static_cast<size_t>(s.sym)].type;
                if (is_ptr(t)) throw CompileError(s.line, s.col, "cannot assign to pointer '" + s.name + "'; assign to an element: " + s.name + "[i] = ...");
                check_expr(s.expr);
                s.expr = coerce(std::move(s.expr), t, "assignment");
                return;
            }
            case Stmt::Store: {
                s.sym = lookup(s.name, s.line, s.col);
                Type t = k_->symbols[static_cast<size_t>(s.sym)].type;
                if (!is_ptr(t)) throw CompileError(s.line, s.col, "'" + s.name + "' is not an array/pointer, cannot index it");
                check_expr(s.index);
                if (s.index->type != Type::Int)
                    throw CompileError(s.index->line, s.index->col, std::string("array index must be int, found ") + type_name(s.index->type));
                check_expr(s.expr);
                s.expr = coerce(std::move(s.expr), elem_type(t), "array store");
                return;
            }
            case Stmt::If:
            case Stmt::While:
                check_expr(s.expr);
                if (s.expr->type != Type::Bool)
                    throw CompileError(s.expr->line, s.expr->col,
                                       std::string("condition must be a comparison (like i < n), found ") + type_name(s.expr->type));
                check_stmt(*s.thenS);
                if (s.elseS) check_stmt(*s.elseS);
                return;
            case Stmt::Return: return;
        }
    }
};
