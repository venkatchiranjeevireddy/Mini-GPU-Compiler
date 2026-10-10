// ast.hpp - abstract syntax tree, symbol table entries and an AST printer.
#pragma once
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

enum class Type { Void, Int, Float, Bool, PtrInt, PtrFloat };

inline const char* type_name(Type t) {
    switch (t) {
        case Type::Void: return "void";
        case Type::Int: return "int";
        case Type::Float: return "float";
        case Type::Bool: return "bool";
        case Type::PtrInt: return "int*";
        case Type::PtrFloat: return "float*";
    }
    return "?";
}
inline bool is_ptr(Type t) { return t == Type::PtrInt || t == Type::PtrFloat; }
inline bool is_numeric(Type t) { return t == Type::Int || t == Type::Float; }
inline Type elem_type(Type t) { return t == Type::PtrInt ? Type::Int : Type::Float; }

struct Expr;
using ExprPtr = std::unique_ptr<Expr>;

struct Expr {
    enum Kind { IntLit, FloatLit, BoolLit, Var, Special, Unary, Binary, Index, Cast } kind = IntLit;
    Type type = Type::Void;  // filled in by Sema
    int line = 0, col = 0;
    long long ival = 0;      // IntLit
    float fval = 0;          // FloatLit
    bool bval = false;       // BoolLit
    std::string name;        // Var / Index: variable name.  Special: "threadIdx.x" etc.
    std::string op;          // Unary / Binary operator text
    int sym = -1;            // Var / Index: symbol id (filled in by Sema)
    ExprPtr lhs, rhs;        // Unary/Cast: lhs.  Binary: lhs, rhs.  Index: rhs is the index expression.
};

struct Stmt;
using StmtPtr = std::unique_ptr<Stmt>;

struct Stmt {
    enum Kind { Decl, Assign, Store, If, While, Block, Return } kind = Block;
    int line = 0, col = 0;
    Type declType = Type::Void;  // Decl
    std::string name;            // Decl / Assign / Store: target variable
    int sym = -1;                // symbol id (filled in by Sema)
    ExprPtr index;               // Store: element index
    ExprPtr expr;                // Decl init (may be null), Assign rhs, Store value, If/While condition
    std::vector<StmtPtr> body;   // Block statements
    StmtPtr thenS, elseS;        // If branches; While body is thenS
};

struct Param {
    Type type = Type::Int;
    std::string name;
    int sym = -1;
    int line = 0, col = 0;
};

struct Symbol {
    std::string name;
    Type type = Type::Int;
    bool isParam = false;
    int paramIndex = -1;
};

struct Kernel {
    std::string name;
    std::vector<Param> params;
    StmtPtr body;
    std::vector<Symbol> symbols;  // filled in by Sema
    int line = 0, col = 0;
};

struct Program {
    std::vector<Kernel> kernels;
};

// ---------------------------------------------------------------- helpers

inline ExprPtr make_expr(Expr::Kind k, int line, int col) {
    auto e = std::make_unique<Expr>();
    e->kind = k;
    e->line = line;
    e->col = col;
    return e;
}

inline ExprPtr clone_expr(const Expr& e) {
    auto c = std::make_unique<Expr>();
    c->kind = e.kind;
    c->type = e.type;
    c->line = e.line;
    c->col = e.col;
    c->ival = e.ival;
    c->fval = e.fval;
    c->bval = e.bval;
    c->name = e.name;
    c->op = e.op;
    c->sym = e.sym;
    if (e.lhs) c->lhs = clone_expr(*e.lhs);
    if (e.rhs) c->rhs = clone_expr(*e.rhs);
    return c;
}

// ---------------------------------------------------------------- printer

inline void dump_expr(const Expr& e, std::ostream& os) {
    switch (e.kind) {
        case Expr::IntLit: os << e.ival; break;
        case Expr::FloatLit: os << e.fval << "f"; break;
        case Expr::BoolLit: os << (e.bval ? "true" : "false"); break;
        case Expr::Var: os << e.name; break;
        case Expr::Special: os << e.name; break;
        case Expr::Unary:
            os << "(" << e.op << " ";
            dump_expr(*e.lhs, os);
            os << ")";
            break;
        case Expr::Binary:
            os << "(" << e.op << " ";
            dump_expr(*e.lhs, os);
            os << " ";
            dump_expr(*e.rhs, os);
            os << ")";
            break;
        case Expr::Index:
            os << e.name << "[";
            dump_expr(*e.rhs, os);
            os << "]";
            break;
        case Expr::Cast:
            os << "(cast " << type_name(e.type) << " ";
            dump_expr(*e.lhs, os);
            os << ")";
            break;
    }
}

inline void dump_stmt(const Stmt& s, std::ostream& os, int ind) {
    std::string pad(static_cast<size_t>(ind) * 2, ' ');
    switch (s.kind) {
        case Stmt::Decl:
            os << pad << "Decl " << type_name(s.declType) << " " << s.name;
            if (s.expr) {
                os << " = ";
                dump_expr(*s.expr, os);
            }
            os << "\n";
            break;
        case Stmt::Assign:
            os << pad << "Assign " << s.name << " = ";
            dump_expr(*s.expr, os);
            os << "\n";
            break;
        case Stmt::Store:
            os << pad << "Store " << s.name << "[";
            dump_expr(*s.index, os);
            os << "] = ";
            dump_expr(*s.expr, os);
            os << "\n";
            break;
        case Stmt::If:
            os << pad << "If ";
            dump_expr(*s.expr, os);
            os << "\n";
            dump_stmt(*s.thenS, os, ind + 1);
            if (s.elseS) {
                os << pad << "Else\n";
                dump_stmt(*s.elseS, os, ind + 1);
            }
            break;
        case Stmt::While:
            os << pad << "While ";
            dump_expr(*s.expr, os);
            os << "\n";
            dump_stmt(*s.thenS, os, ind + 1);
            break;
        case Stmt::Block:
            os << pad << "Block\n";
            for (const auto& c : s.body) dump_stmt(*c, os, ind + 1);
            break;
        case Stmt::Return: os << pad << "Return\n"; break;
    }
}

inline void dump_kernel(const Kernel& k, std::ostream& os) {
    os << "Kernel " << k.name << "(";
    for (size_t i = 0; i < k.params.size(); i++) {
        if (i) os << ", ";
        os << type_name(k.params[i].type) << " " << k.params[i].name;
    }
    os << ")\n";
    dump_stmt(*k.body, os, 1);
}
