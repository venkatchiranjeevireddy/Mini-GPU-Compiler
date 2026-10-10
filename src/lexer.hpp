// lexer.hpp - turns source text into a vector of tokens.
#pragma once
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.hpp"

enum class Tok {
    Eof, Ident, IntLit, FloatLit,
    KwKernel, KwInt, KwFloat, KwIf, KwElse, KwWhile, KwFor, KwReturn,
    LParen, RParen, LBrace, RBrace, LBracket, RBracket,
    Comma, Semi, Dot,
    Plus, Minus, Star, Slash, Percent,
    Assign, PlusEq, MinusEq, StarEq, SlashEq, PlusPlus, MinusMinus,
    Lt, Le, Gt, Ge, EqEq, Ne, AndAnd, OrOr, Bang
};

struct Token {
    Tok kind = Tok::Eof;
    std::string text;
    int line = 1, col = 1;
    long long ival = 0;
    double fval = 0;
};

inline std::string tok_name(Tok t) {
    switch (t) {
        case Tok::Eof: return "end of file";
        case Tok::Ident: return "identifier";
        case Tok::IntLit: return "integer literal";
        case Tok::FloatLit: return "float literal";
        case Tok::KwKernel: return "'kernel'";
        case Tok::KwInt: return "'int'";
        case Tok::KwFloat: return "'float'";
        case Tok::KwIf: return "'if'";
        case Tok::KwElse: return "'else'";
        case Tok::KwWhile: return "'while'";
        case Tok::KwFor: return "'for'";
        case Tok::KwReturn: return "'return'";
        case Tok::LParen: return "'('";
        case Tok::RParen: return "')'";
        case Tok::LBrace: return "'{'";
        case Tok::RBrace: return "'}'";
        case Tok::LBracket: return "'['";
        case Tok::RBracket: return "']'";
        case Tok::Comma: return "','";
        case Tok::Semi: return "';'";
        case Tok::Dot: return "'.'";
        case Tok::Plus: return "'+'";
        case Tok::Minus: return "'-'";
        case Tok::Star: return "'*'";
        case Tok::Slash: return "'/'";
        case Tok::Percent: return "'%'";
        case Tok::Assign: return "'='";
        case Tok::PlusEq: return "'+='";
        case Tok::MinusEq: return "'-='";
        case Tok::StarEq: return "'*='";
        case Tok::SlashEq: return "'/='";
        case Tok::PlusPlus: return "'++'";
        case Tok::MinusMinus: return "'--'";
        case Tok::Lt: return "'<'";
        case Tok::Le: return "'<='";
        case Tok::Gt: return "'>'";
        case Tok::Ge: return "'>='";
        case Tok::EqEq: return "'=='";
        case Tok::Ne: return "'!='";
        case Tok::AndAnd: return "'&&'";
        case Tok::OrOr: return "'||'";
        case Tok::Bang: return "'!'";
    }
    return "?";
}

class Lexer {
public:
    explicit Lexer(const std::string& src) : s_(src) {}

    std::vector<Token> tokenize() {
        std::vector<Token> out;
        for (;;) {
            skip_space_and_comments();
            Token t;
            t.line = line_;
            t.col = col_;
            if (i_ >= s_.size()) {
                t.kind = Tok::Eof;
                out.push_back(t);
                return out;
            }
            char c = s_[i_];
            if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
                lex_ident(t);
            } else if (std::isdigit(static_cast<unsigned char>(c)) ||
                       (c == '.' && i_ + 1 < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_ + 1])))) {
                lex_number(t);
            } else {
                lex_punct(t);
            }
            out.push_back(t);
        }
    }

private:
    const std::string& s_;
    size_t i_ = 0;
    int line_ = 1, col_ = 1;

    char peek(size_t off = 0) const { return i_ + off < s_.size() ? s_[i_ + off] : '\0'; }

    void advance() {
        if (s_[i_] == '\n') {
            line_++;
            col_ = 1;
        } else {
            col_++;
        }
        i_++;
    }

    void skip_space_and_comments() {
        for (;;) {
            while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) advance();
            if (peek() == '/' && peek(1) == '/') {
                while (i_ < s_.size() && s_[i_] != '\n') advance();
            } else if (peek() == '/' && peek(1) == '*') {
                int sl = line_, sc = col_;
                advance();
                advance();
                while (i_ < s_.size() && !(peek() == '*' && peek(1) == '/')) advance();
                if (i_ >= s_.size()) throw CompileError(sl, sc, "unterminated /* comment");
                advance();
                advance();
            } else {
                return;
            }
        }
    }

    void lex_ident(Token& t) {
        static const std::unordered_map<std::string, Tok> kw = {
            {"kernel", Tok::KwKernel}, {"int", Tok::KwInt},     {"float", Tok::KwFloat},
            {"if", Tok::KwIf},         {"else", Tok::KwElse},   {"while", Tok::KwWhile},
            {"for", Tok::KwFor},       {"return", Tok::KwReturn}};
        size_t start = i_;
        while (i_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[i_])) || s_[i_] == '_')) advance();
        t.text = s_.substr(start, i_ - start);
        auto it = kw.find(t.text);
        t.kind = it == kw.end() ? Tok::Ident : it->second;
    }

    void lex_number(Token& t) {
        size_t start = i_;
        bool isFloat = false;
        while (std::isdigit(static_cast<unsigned char>(peek()))) advance();
        if (peek() == '.') {
            isFloat = true;
            advance();
            while (std::isdigit(static_cast<unsigned char>(peek()))) advance();
        }
        if (peek() == 'e' || peek() == 'E') {
            size_t k = 1;
            if (peek(k) == '+' || peek(k) == '-') k++;
            if (std::isdigit(static_cast<unsigned char>(peek(k)))) {
                isFloat = true;
                for (size_t n = 0; n < k; n++) advance();
                while (std::isdigit(static_cast<unsigned char>(peek()))) advance();
            }
        }
        t.text = s_.substr(start, i_ - start);
        if (peek() == 'f' || peek() == 'F') {  // 1.5f / 2f suffix
            isFloat = true;
            advance();
        }
        if (isFloat) {
            t.kind = Tok::FloatLit;
            t.fval = std::strtod(t.text.c_str(), nullptr);
        } else {
            t.kind = Tok::IntLit;
            errno = 0;
            long long v = std::strtoll(t.text.c_str(), nullptr, 10);
            if (errno != 0 || v > INT_MAX) throw CompileError(t.line, t.col, "integer literal '" + t.text + "' is too large for 32-bit int");
            t.ival = v;
        }
    }

    void lex_punct(Token& t) {
        char c = peek();
        char n = peek(1);
        auto two = [&](Tok k, const char* txt) {
            t.kind = k;
            t.text = txt;
            advance();
            advance();
        };
        auto one = [&](Tok k) {
            t.kind = k;
            t.text = std::string(1, c);
            advance();
        };
        switch (c) {
            case '(': one(Tok::LParen); return;
            case ')': one(Tok::RParen); return;
            case '{': one(Tok::LBrace); return;
            case '}': one(Tok::RBrace); return;
            case '[': one(Tok::LBracket); return;
            case ']': one(Tok::RBracket); return;
            case ',': one(Tok::Comma); return;
            case ';': one(Tok::Semi); return;
            case '.': one(Tok::Dot); return;
            case '%': one(Tok::Percent); return;
            case '+':
                if (n == '+') two(Tok::PlusPlus, "++");
                else if (n == '=') two(Tok::PlusEq, "+=");
                else one(Tok::Plus);
                return;
            case '-':
                if (n == '-') two(Tok::MinusMinus, "--");
                else if (n == '=') two(Tok::MinusEq, "-=");
                else one(Tok::Minus);
                return;
            case '*':
                if (n == '=') two(Tok::StarEq, "*=");
                else one(Tok::Star);
                return;
            case '/':
                if (n == '=') two(Tok::SlashEq, "/=");
                else one(Tok::Slash);
                return;
            case '=':
                if (n == '=') two(Tok::EqEq, "==");
                else one(Tok::Assign);
                return;
            case '!':
                if (n == '=') two(Tok::Ne, "!=");
                else one(Tok::Bang);
                return;
            case '<':
                if (n == '=') two(Tok::Le, "<=");
                else one(Tok::Lt);
                return;
            case '>':
                if (n == '=') two(Tok::Ge, ">=");
                else one(Tok::Gt);
                return;
            case '&':
                if (n == '&') {
                    two(Tok::AndAnd, "&&");
                    return;
                }
                break;
            case '|':
                if (n == '|') {
                    two(Tok::OrOr, "||");
                    return;
                }
                break;
            default: break;
        }
        throw CompileError(t.line, t.col, std::string("unexpected character '") + c + "'");
    }
};
