// parser.hpp - recursive-descent parser producing the AST.
//
// Grammar (informal):
//   program   := kernel*
//   kernel    := 'kernel' IDENT '(' [param (',' param)*] ')' block
//   param     := ('int' | 'float') ['*'] IDENT
//   block     := '{' stmt* '}'
//   stmt      := block | decl ';' | simple ';' | 'if' '(' expr ')' stmt ['else' stmt]
//              | 'while' '(' expr ')' stmt
//              | 'for' '(' [decl|simple] ';' [expr] ';' [simple] ')' stmt
//              | 'return' ';'
//   decl      := ('int' | 'float') IDENT ['=' expr]
//   simple    := IDENT ('=' | '+=' | '-=' | '*=' | '/=') expr
//              | IDENT '[' expr ']' ('=' | '+=' | '-=' | '*=' | '/=') expr
//              | IDENT ('++' | '--')
//   expr      := precedence climbing: || , && , == != , < <= > >= , + - , * / % , unary - ! , cast, primary
#pragma once
#include <algorithm>
#include <string>
#include <vector>

#include "ast.hpp"
#include "common.hpp"
#include "lexer.hpp"

class Parser {
public:
    explicit Parser(std::vector<Token> toks) : t_(std::move(toks)) {}

    Program parse_program() {
        Program p;
        while (cur().kind != Tok::Eof) p.kernels.push_back(parse_kernel());
        if (p.kernels.empty()) throw CompileError(1, 1, "no kernel found in input");
        return p;
    }

private:
    std::vector<Token> t_;
    size_t p_ = 0;

    // ------------------------------------------------------------ token helpers
    const Token& cur() const { return t_[p_]; }
    const Token& peek(size_t n = 1) const { return t_[std::min(p_ + n, t_.size() - 1)]; }
    bool at(Tok k) const { return cur().kind == k; }
    Token next() {
        Token t = t_[p_];
        if (p_ + 1 < t_.size()) p_++;
        return t;
    }
    bool accept(Tok k) {
        if (at(k)) {
            next();
            return true;
        }
        return false;
    }
    Token expect(Tok k, const char* ctx = "") {
        if (!at(k)) {
            std::string msg = "expected " + tok_name(k);
            if (*ctx) msg += std::string(" ") + ctx;
            msg += ", found " + (at(Tok::Eof) ? std::string("end of file") : "'" + cur().text + "'");
            throw CompileError(cur().line, cur().col, msg);
        }
        return next();
    }
    [[noreturn]] void fail(const std::string& msg) { throw CompileError(cur().line, cur().col, msg); }

    // ------------------------------------------------------------ declarations
    Kernel parse_kernel() {
        Kernel k;
        Token kw = expect(Tok::KwKernel, "at start of a kernel definition");
        k.line = kw.line;
        k.col = kw.col;
        k.name = expect(Tok::Ident, "(kernel name)").text;
        expect(Tok::LParen);
        if (!at(Tok::RParen)) {
            do {
                k.params.push_back(parse_param());
            } while (accept(Tok::Comma));
        }
        expect(Tok::RParen);
        k.body = parse_block();
        return k;
    }

    Param parse_param() {
        Param p;
        p.line = cur().line;
        p.col = cur().col;
        bool isInt;
        if (accept(Tok::KwInt)) isInt = true;
        else if (accept(Tok::KwFloat)) isInt = false;
        else fail("expected parameter type 'int' or 'float'");
        bool ptr = accept(Tok::Star);
        p.type = ptr ? (isInt ? Type::PtrInt : Type::PtrFloat) : (isInt ? Type::Int : Type::Float);
        p.name = expect(Tok::Ident, "(parameter name)").text;
        return p;
    }

    // ------------------------------------------------------------ statements
    StmtPtr new_stmt(Stmt::Kind k, const Token& at_tok) {
        auto s = std::make_unique<Stmt>();
        s->kind = k;
        s->line = at_tok.line;
        s->col = at_tok.col;
        return s;
    }

    StmtPtr parse_block() {
        Token lb = expect(Tok::LBrace);
        auto b = new_stmt(Stmt::Block, lb);
        while (!at(Tok::RBrace)) {
            if (at(Tok::Eof)) throw CompileError(lb.line, lb.col, "unterminated '{' (missing '}')");
            b->body.push_back(parse_stmt());
        }
        expect(Tok::RBrace);
        return b;
    }

    // A statement used as the body of if/while/for is always wrapped in a Block (gives it its own scope).
    StmtPtr parse_body() {
        if (at(Tok::LBrace)) return parse_block();
        Token start = cur();
        auto b = new_stmt(Stmt::Block, start);
        b->body.push_back(parse_stmt());
        return b;
    }

    StmtPtr parse_stmt() {
        const Token start = cur();
        switch (start.kind) {
            case Tok::LBrace: return parse_block();
            case Tok::KwInt:
            case Tok::KwFloat: {
                auto s = parse_decl();
                expect(Tok::Semi, "after declaration");
                return s;
            }
            case Tok::KwIf: {
                next();
                auto s = new_stmt(Stmt::If, start);
                expect(Tok::LParen, "after 'if'");
                s->expr = parse_expr();
                expect(Tok::RParen);
                s->thenS = parse_body();
                if (accept(Tok::KwElse)) s->elseS = parse_body();
                return s;
            }
            case Tok::KwWhile: {
                next();
                auto s = new_stmt(Stmt::While, start);
                expect(Tok::LParen, "after 'while'");
                s->expr = parse_expr();
                expect(Tok::RParen);
                s->thenS = parse_body();
                return s;
            }
            case Tok::KwFor: return parse_for();
            case Tok::KwReturn: {
                next();
                auto s = new_stmt(Stmt::Return, start);
                expect(Tok::Semi, "after 'return'");
                return s;
            }
            case Tok::Ident: {
                auto s = parse_simple();
                expect(Tok::Semi, "after statement");
                return s;
            }
            default: fail("unexpected " + (at(Tok::Eof) ? std::string("end of file") : "'" + cur().text + "'") + " at start of statement");
        }
    }

    StmtPtr parse_decl() {
        Token ty = next();  // 'int' or 'float'
        auto s = new_stmt(Stmt::Decl, ty);
        s->declType = ty.kind == Tok::KwInt ? Type::Int : Type::Float;
        if (at(Tok::Star)) fail("local pointer variables are not supported (pointers can only be kernel parameters)");
        Token name = expect(Tok::Ident, "(variable name)");
        s->name = name.text;
        s->line = name.line;
        s->col = name.col;
        if (accept(Tok::Assign)) s->expr = parse_expr();
        return s;
    }

    static bool is_compound(Tok k) { return k == Tok::PlusEq || k == Tok::MinusEq || k == Tok::StarEq || k == Tok::SlashEq; }
    static const char* compound_op(Tok k) {
        switch (k) {
            case Tok::PlusEq: return "+";
            case Tok::MinusEq: return "-";
            case Tok::StarEq: return "*";
            default: return "/";
        }
    }

    ExprPtr make_binary(const std::string& op, ExprPtr l, ExprPtr r, const Token& at_tok) {
        auto e = make_expr(Expr::Binary, at_tok.line, at_tok.col);
        e->op = op;
        e->lhs = std::move(l);
        e->rhs = std::move(r);
        return e;
    }

    // IDENT = e | IDENT op= e | IDENT[e] = e | IDENT[e] op= e | IDENT++ | IDENT--   (no trailing ';')
    StmtPtr parse_simple() {
        Token id = expect(Tok::Ident);
        if (at(Tok::LBracket)) {  // array store
            next();
            auto s = new_stmt(Stmt::Store, id);
            s->name = id.text;
            s->index = parse_expr();
            expect(Tok::RBracket);
            Token opTok = cur();
            if (accept(Tok::Assign)) {
                s->expr = parse_expr();
            } else if (is_compound(opTok.kind)) {
                next();
                auto load = make_expr(Expr::Index, id.line, id.col);
                load->name = id.text;
                load->rhs = clone_expr(*s->index);
                s->expr = make_binary(compound_op(opTok.kind), std::move(load), parse_expr(), opTok);
            } else {
                fail("expected '=' or compound assignment after array element");
            }
            return s;
        }
        auto s = new_stmt(Stmt::Assign, id);
        s->name = id.text;
        Token opTok = cur();
        auto var = [&]() {
            auto v = make_expr(Expr::Var, id.line, id.col);
            v->name = id.text;
            return v;
        };
        if (accept(Tok::Assign)) {
            s->expr = parse_expr();
        } else if (is_compound(opTok.kind)) {
            next();
            s->expr = make_binary(compound_op(opTok.kind), var(), parse_expr(), opTok);
        } else if (at(Tok::PlusPlus) || at(Tok::MinusMinus)) {
            bool inc = at(Tok::PlusPlus);
            next();
            auto one = make_expr(Expr::IntLit, opTok.line, opTok.col);
            one->ival = 1;
            s->expr = make_binary(inc ? "+" : "-", var(), std::move(one), opTok);
        } else {
            fail("expected '=', compound assignment, '++' or '--' after '" + id.text + "'");
        }
        return s;
    }

    // for (init; cond; step) body   ==>   { init; while (cond) { body; step; } }
    StmtPtr parse_for() {
        Token kw = next();
        expect(Tok::LParen, "after 'for'");
        auto outer = new_stmt(Stmt::Block, kw);
        if (!at(Tok::Semi)) {
            if (at(Tok::KwInt) || at(Tok::KwFloat)) outer->body.push_back(parse_decl());
            else outer->body.push_back(parse_simple());
        }
        expect(Tok::Semi, "after for-loop initializer");
        ExprPtr cond;
        if (!at(Tok::Semi)) {
            cond = parse_expr();
        } else {
            cond = make_expr(Expr::BoolLit, cur().line, cur().col);
            cond->bval = true;
        }
        expect(Tok::Semi, "after for-loop condition");
        StmtPtr step;
        if (!at(Tok::RParen)) step = parse_simple();
        expect(Tok::RParen);
        StmtPtr body = parse_body();
        if (step) body->body.push_back(std::move(step));
        auto w = new_stmt(Stmt::While, kw);
        w->expr = std::move(cond);
        w->thenS = std::move(body);
        outer->body.push_back(std::move(w));
        return outer;
    }

    // ------------------------------------------------------------ expressions
    ExprPtr parse_expr() { return parse_or(); }

    ExprPtr parse_or() {
        auto l = parse_and();
        while (at(Tok::OrOr)) {
            Token t = next();
            l = make_binary("||", std::move(l), parse_and(), t);
        }
        return l;
    }
    ExprPtr parse_and() {
        auto l = parse_eq();
        while (at(Tok::AndAnd)) {
            Token t = next();
            l = make_binary("&&", std::move(l), parse_eq(), t);
        }
        return l;
    }
    ExprPtr parse_eq() {
        auto l = parse_rel();
        while (at(Tok::EqEq) || at(Tok::Ne)) {
            Token t = next();
            l = make_binary(t.text, std::move(l), parse_rel(), t);
        }
        return l;
    }
    ExprPtr parse_rel() {
        auto l = parse_add();
        while (at(Tok::Lt) || at(Tok::Le) || at(Tok::Gt) || at(Tok::Ge)) {
            Token t = next();
            l = make_binary(t.text, std::move(l), parse_add(), t);
        }
        return l;
    }
    ExprPtr parse_add() {
        auto l = parse_mul();
        while (at(Tok::Plus) || at(Tok::Minus)) {
            Token t = next();
            l = make_binary(t.text, std::move(l), parse_mul(), t);
        }
        return l;
    }
    ExprPtr parse_mul() {
        auto l = parse_unary();
        while (at(Tok::Star) || at(Tok::Slash) || at(Tok::Percent)) {
            Token t = next();
            l = make_binary(t.text, std::move(l), parse_unary(), t);
        }
        return l;
    }
    ExprPtr parse_unary() {
        if (at(Tok::Minus) || at(Tok::Bang)) {
            Token t = next();
            auto e = make_expr(Expr::Unary, t.line, t.col);
            e->op = t.text;
            e->lhs = parse_unary();
            return e;
        }
        return parse_primary();
    }

    static bool is_special_base(const std::string& s) {
        return s == "threadIdx" || s == "blockIdx" || s == "blockDim" || s == "gridDim";
    }

    ExprPtr parse_primary() {
        Token t = cur();
        switch (t.kind) {
            case Tok::IntLit: {
                next();
                auto e = make_expr(Expr::IntLit, t.line, t.col);
                e->ival = t.ival;
                return e;
            }
            case Tok::FloatLit: {
                next();
                auto e = make_expr(Expr::FloatLit, t.line, t.col);
                e->fval = static_cast<float>(t.fval);
                return e;
            }
            case Tok::LParen: {
                if (peek().kind == Tok::KwInt || peek().kind == Tok::KwFloat) {  // cast: (float) x
                    next();
                    Token ty = next();
                    expect(Tok::RParen, "after cast type");
                    auto e = make_expr(Expr::Cast, t.line, t.col);
                    e->type = ty.kind == Tok::KwInt ? Type::Int : Type::Float;
                    e->lhs = parse_unary();
                    return e;
                }
                next();
                auto e = parse_expr();
                expect(Tok::RParen, "to close '('");
                return e;
            }
            case Tok::Ident: {
                next();
                if (is_special_base(t.text)) {
                    expect(Tok::Dot, ("after '" + t.text + "'").c_str());
                    Token comp = expect(Tok::Ident, "(x, y or z)");
                    if (comp.text != "x" && comp.text != "y" && comp.text != "z")
                        throw CompileError(comp.line, comp.col, "expected x, y or z after '" + t.text + ".'");
                    auto e = make_expr(Expr::Special, t.line, t.col);
                    e->name = t.text + "." + comp.text;
                    return e;
                }
                if (at(Tok::LBracket)) {
                    next();
                    auto e = make_expr(Expr::Index, t.line, t.col);
                    e->name = t.text;
                    e->rhs = parse_expr();
                    expect(Tok::RBracket);
                    return e;
                }
                auto e = make_expr(Expr::Var, t.line, t.col);
                e->name = t.text;
                return e;
            }
            default:
                fail("expected an expression, found " + (at(Tok::Eof) ? std::string("end of file") : "'" + cur().text + "'"));
        }
    }
};
