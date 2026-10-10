// codegen.hpp - emits PTX text for the type-checked (and optionally optimized) AST.
//
// Register classes (all virtual, ptxas does the real register allocation):
//   %r<N>  .b32  int values
//   %f<N>  .f32  float values
//   %rd<N> .b64  device addresses
//   %p<N>  .pred conditions
//
// Every source variable gets one dedicated virtual register; expression temporaries get fresh ones.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "ast.hpp"
#include "common.hpp"

struct Operand {
    std::string text;
    bool imm = false;  // true when `text` is an immediate constant rather than a register
};

class KernelGen {
public:
    explicit KernelGen(const Kernel& k) : k_(k) {}

    std::string generate() {
        sym_reg_.assign(k_.symbols.size(), "");
        emit_prologue();
        gen_stmt(*k_.body);
        line("ret;");

        std::ostringstream out;
        out << ".visible .entry " << k_.name << "(\n";
        for (size_t i = 0; i < k_.params.size(); i++) {
            const char* pt = k_.params[i].type == Type::Float ? ".f32" : (is_ptr(k_.params[i].type) ? ".u64" : ".u32");
            out << "    .param " << pt << " " << param_name(i) << (i + 1 < k_.params.size() ? ",\n" : "\n");
        }
        out << ")\n{\n";
        if (np_) out << "    .reg .pred %p<" << np_ + 1 << ">;\n";
        if (nr_) out << "    .reg .b32  %r<" << nr_ + 1 << ">;\n";
        if (nf_) out << "    .reg .f32  %f<" << nf_ + 1 << ">;\n";
        if (nrd_) out << "    .reg .b64  %rd<" << nrd_ + 1 << ">;\n";
        out << "\n" << body_.str() << "}\n";
        return out.str();
    }

private:
    const Kernel& k_;
    std::ostringstream body_;
    std::vector<std::string> sym_reg_;
    int nr_ = 0, nf_ = 0, nrd_ = 0, np_ = 0, nlabel_ = 0;

    // ------------------------------------------------------------ small helpers
    std::string param_name(size_t i) const { return k_.name + "_param_" + std::to_string(i); }
    std::string new_r() { return "%r" + std::to_string(++nr_); }
    std::string new_f() { return "%f" + std::to_string(++nf_); }
    std::string new_rd() { return "%rd" + std::to_string(++nrd_); }
    std::string new_p() { return "%p" + std::to_string(++np_); }
    std::string new_label(const char* tag) { return "$L_" + std::string(tag) + std::to_string(++nlabel_); }

    void line(const std::string& s) { body_ << "    " << s << "\n"; }
    void label(const std::string& l) { body_ << l << ":\n"; }

    static std::string float_imm(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof bits);
        char buf[16];
        std::snprintf(buf, sizeof buf, "0f%08X", bits);
        return buf;
    }

    // Make sure an int / float operand lives in a register (some PTX operand slots reject immediates).
    Operand int_reg(const Operand& o) {
        if (!o.imm) return o;
        std::string r = new_r();
        line("mov.s32 " + r + ", " + o.text + ";");
        return {r, false};
    }
    Operand float_reg(const Operand& o) {
        if (!o.imm) return o;
        std::string r = new_f();
        line("mov.f32 " + r + ", " + o.text + ";");
        return {r, false};
    }
    Operand reg_of(const Operand& o, Type t) { return t == Type::Float ? float_reg(o) : int_reg(o); }

    // ------------------------------------------------------------ kernel entry
    void emit_prologue() {
        for (size_t i = 0; i < k_.params.size(); i++) {
            const Param& p = k_.params[i];
            std::string r;
            if (is_ptr(p.type)) {
                r = new_rd();
                line("ld.param.u64 " + r + ", [" + param_name(i) + "];");
                line("cvta.to.global.u64 " + r + ", " + r + ";");
            } else if (p.type == Type::Float) {
                r = new_f();
                line("ld.param.f32 " + r + ", [" + param_name(i) + "];");
            } else {
                r = new_r();
                line("ld.param.u32 " + r + ", [" + param_name(i) + "];");
            }
            sym_reg_[static_cast<size_t>(p.sym)] = r;
        }
    }

    // ------------------------------------------------------------ expressions
    static const char* special_reg(const std::string& name) {
        static const struct { const char* src; const char* ptx; } table[] = {
            {"threadIdx.x", "%tid.x"},   {"threadIdx.y", "%tid.y"},   {"threadIdx.z", "%tid.z"},
            {"blockIdx.x", "%ctaid.x"},  {"blockIdx.y", "%ctaid.y"},  {"blockIdx.z", "%ctaid.z"},
            {"blockDim.x", "%ntid.x"},   {"blockDim.y", "%ntid.y"},   {"blockDim.z", "%ntid.z"},
            {"gridDim.x", "%nctaid.x"},  {"gridDim.y", "%nctaid.y"},  {"gridDim.z", "%nctaid.z"}};
        for (const auto& e : table)
            if (name == e.src) return e.ptx;
        return nullptr;
    }

    // Returns the address register for base[idx].
    std::string element_address(int sym, const Expr& idx) {
        const std::string& base = sym_reg_[static_cast<size_t>(sym)];
        std::string addr = new_rd();
        Operand i = gen_expr(idx);
        if (i.imm) {
            long long off = std::stoll(i.text) * 4;
            line("add.s64 " + addr + ", " + base + ", " + std::to_string(off) + ";");
        } else {
            std::string off = new_rd();
            line("mul.wide.s32 " + off + ", " + i.text + ", 4;");
            line("add.s64 " + addr + ", " + base + ", " + off + ";");
        }
        return addr;
    }

    Operand gen_expr(const Expr& e) {
        switch (e.kind) {
            case Expr::IntLit: return {std::to_string(e.ival), true};
            case Expr::FloatLit: return {float_imm(e.fval), true};
            case Expr::BoolLit: {
                std::string r = new_r(), p = new_p();
                line("mov.s32 " + r + ", " + (e.bval ? "1" : "0") + ";");
                line("setp.ne.s32 " + p + ", " + r + ", 0;");
                return {p, false};
            }
            case Expr::Var: return {sym_reg_[static_cast<size_t>(e.sym)], false};
            case Expr::Special: {
                std::string r = new_r();
                line("mov.u32 " + r + ", " + special_reg(e.name) + ";");
                return {r, false};
            }
            case Expr::Index: {
                std::string addr = element_address(e.sym, *e.rhs);
                if (e.type == Type::Float) {
                    std::string f = new_f();
                    line("ld.global.f32 " + f + ", [" + addr + "];");
                    return {f, false};
                }
                std::string r = new_r();
                line("ld.global.s32 " + r + ", [" + addr + "];");
                return {r, false};
            }
            case Expr::Cast: {
                Operand x = gen_expr(*e.lhs);
                if (e.type == Type::Float && e.lhs->type == Type::Int) {
                    x = int_reg(x);
                    std::string f = new_f();
                    line("cvt.rn.f32.s32 " + f + ", " + x.text + ";");
                    return {f, false};
                }
                if (e.type == Type::Int && e.lhs->type == Type::Float) {
                    x = float_reg(x);
                    std::string r = new_r();
                    line("cvt.rzi.s32.f32 " + r + ", " + x.text + ";");
                    return {r, false};
                }
                return x;  // same type: no-op
            }
            case Expr::Unary: {
                Operand x = gen_expr(*e.lhs);
                if (e.op == "!") {
                    std::string p = new_p();
                    line("not.pred " + p + ", " + x.text + ";");
                    return {p, false};
                }
                if (e.type == Type::Float) {
                    x = float_reg(x);
                    std::string f = new_f();
                    line("neg.f32 " + f + ", " + x.text + ";");
                    return {f, false};
                }
                x = int_reg(x);
                std::string r = new_r();
                line("neg.s32 " + r + ", " + x.text + ";");
                return {r, false};
            }
            case Expr::Binary: return gen_binary(e);
        }
        throw CompileError(e.line, e.col, "internal error: unknown expression kind");
    }

    Operand gen_binary(const Expr& e) {
        const std::string& op = e.op;

        // Short-circuit && and ||: the right side (which may load from memory) only runs when needed.
        if (op == "&&" || op == "||") {
            std::string result = new_p();
            std::string end = new_label("sc");
            Operand l = gen_expr(*e.lhs);
            line("mov.pred " + result + ", " + l.text + ";");
            line(std::string(op == "&&" ? "@!" : "@") + l.text + " bra " + end + ";");
            Operand r = gen_expr(*e.rhs);
            line("mov.pred " + result + ", " + r.text + ";");
            label(end);
            return {result, false};
        }

        Operand a = gen_expr(*e.lhs);
        Operand b = gen_expr(*e.rhs);
        Type opType = e.lhs->type;  // both operands have the same type after Sema
        bool isF = opType == Type::Float;
        a = reg_of(a, opType);  // left operand always in a register; right may stay an immediate

        if (op == "<" || op == "<=" || op == ">" || op == ">=" || op == "==" || op == "!=") {
            const char* cmp = op == "<" ? "lt" : op == "<=" ? "le" : op == ">" ? "gt" : op == ">=" ? "ge" : op == "==" ? "eq" : (isF ? "neu" : "ne");
            std::string p = new_p();
            line(std::string("setp.") + cmp + (isF ? ".f32 " : ".s32 ") + p + ", " + a.text + ", " + b.text + ";");
            return {p, false};
        }

        std::string mnem;
        if (isF) {
            // ".rn" = IEEE round-to-nearest and forbids fusing a*b+c into an fma, so results match a CPU exactly.
            mnem = op == "+" ? "add.rn.f32" : op == "-" ? "sub.rn.f32" : op == "*" ? "mul.rn.f32" : "div.rn.f32";
            std::string d = new_f();
            line(mnem + " " + d + ", " + a.text + ", " + b.text + ";");
            return {d, false};
        }
        mnem = op == "+" ? "add.s32" : op == "-" ? "sub.s32" : op == "*" ? "mul.lo.s32" : op == "/" ? "div.s32" : "rem.s32";
        if ((op == "/" || op == "%") && b.imm) b = int_reg(b);  // div/rem do not take an immediate divisor
        std::string d = new_r();
        line(mnem + " " + d + ", " + a.text + ", " + b.text + ";");
        return {d, false};
    }

    // ------------------------------------------------------------ statements
    void gen_stmt(const Stmt& s) {
        switch (s.kind) {
            case Stmt::Block:
                for (const auto& c : s.body) gen_stmt(*c);
                return;
            case Stmt::Decl: {
                bool isF = s.declType == Type::Float;
                std::string r = isF ? new_f() : new_r();
                sym_reg_[static_cast<size_t>(s.sym)] = r;
                if (s.expr) {
                    Operand v = gen_expr(*s.expr);
                    line(std::string(isF ? "mov.f32 " : "mov.s32 ") + r + ", " + v.text + ";");
                } else {  // uninitialised locals start at zero
                    line(std::string(isF ? "mov.f32 " : "mov.s32 ") + r + ", " + (isF ? "0f00000000" : "0") + ";");
                }
                return;
            }
            case Stmt::Assign: {
                Operand v = gen_expr(*s.expr);
                bool isF = s.expr->type == Type::Float;
                line(std::string(isF ? "mov.f32 " : "mov.s32 ") + sym_reg_[static_cast<size_t>(s.sym)] + ", " + v.text + ";");
                return;
            }
            case Stmt::Store: {
                std::string addr = element_address(s.sym, *s.index);
                Operand v = reg_of(gen_expr(*s.expr), s.expr->type);
                line(std::string(s.expr->type == Type::Float ? "st.global.f32 [" : "st.global.s32 [") + addr + "], " + v.text + ";");
                return;
            }
            case Stmt::If: {
                std::string elseL = new_label("else");
                std::string endL = new_label("endif");
                Operand c = gen_expr(*s.expr);
                line("@!" + c.text + " bra " + (s.elseS ? elseL : endL) + ";");
                gen_stmt(*s.thenS);
                if (s.elseS) {
                    line("bra " + endL + ";");
                    label(elseL);
                    gen_stmt(*s.elseS);
                }
                label(endL);
                return;
            }
            case Stmt::While: {
                std::string startL = new_label("loop");
                std::string endL = new_label("endloop");
                label(startL);
                Operand c = gen_expr(*s.expr);
                line("@!" + c.text + " bra " + endL + ";");
                gen_stmt(*s.thenS);
                line("bra " + startL + ";");
                label(endL);
                return;
            }
            case Stmt::Return: line("ret;"); return;
        }
    }
};

// Minimum PTX ISA version that accepts each GPU target (ptxas rejects a .target newer than its .version).
inline std::string ptx_version_for(const std::string& target) {
    int sm = 75;
    if (target.rfind("sm_", 0) == 0) sm = std::atoi(target.c_str() + 3);
    if (sm >= 100) return "8.6";
    if (sm >= 89) return "7.8";  // sm_89, sm_90
    if (sm >= 87) return "7.4";  // sm_87
    if (sm >= 86) return "7.1";  // sm_86
    if (sm >= 80) return "7.0";  // sm_80
    if (sm >= 75) return "6.4";  // sm_75 (Turing, e.g. Colab T4)
    if (sm >= 72) return "6.1";
    return "6.0";                // sm_70 and older Volta/Pascal targets
}

inline std::string generate_ptx(const Program& p, const std::string& target) {
    std::ostringstream out;
    out << "//\n// Generated by mini-ptx (kernel language -> PTX)\n//\n\n";
    out << ".version " << ptx_version_for(target) << "\n";
    out << ".target " << target << "\n";
    out << ".address_size 64\n\n";
    for (const auto& k : p.kernels) out << KernelGen(k).generate() << "\n";
    return out.str();
}
