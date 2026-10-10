// main.cpp - command line driver.
//
//   mini-ptx input.k                 compile, write PTX to stdout
//   mini-ptx input.k -o out.ptx      write PTX to a file
//   options: --target sm_XX   --no-opt   --dump-ast   --stats
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "ast.hpp"
#include "codegen.hpp"
#include "common.hpp"
#include "lexer.hpp"
#include "optimizer.hpp"
#include "parser.hpp"
#include "sema.hpp"

static void usage() {
    std::cerr << "usage: mini-ptx <input.k> [-o out.ptx] [--target sm_75] [--no-opt] [--dump-ast] [--stats]\n";
}

int main(int argc, char** argv) {
    std::string input, output, target = "sm_75";
    bool optimize = true, dumpAst = false, showStats = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-o" && i + 1 < argc) output = argv[++i];
        else if (a == "--target" && i + 1 < argc) target = argv[++i];
        else if (a == "--no-opt") optimize = false;
        else if (a == "--dump-ast") dumpAst = true;
        else if (a == "--stats") showStats = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (!a.empty() && a[0] == '-') { std::cerr << "unknown option: " << a << "\n"; usage(); return 2; }
        else if (input.empty()) input = a;
        else { usage(); return 2; }
    }
    if (input.empty()) { usage(); return 2; }
    if (target.rfind("sm_", 0) != 0) { std::cerr << "error: --target must look like sm_75\n"; return 2; }

    std::ifstream in(input);
    if (!in) { std::cerr << "error: cannot open '" << input << "'\n"; return 1; }
    std::stringstream buf;
    buf << in.rdbuf();
    std::string src = buf.str();

    try {
        Lexer lexer(src);
        Parser parser(lexer.tokenize());
        Program prog = parser.parse_program();
        Sema().check_program(prog);

        if (dumpAst) {
            std::cerr << "===== AST (after semantic analysis) =====\n";
            for (const auto& k : prog.kernels) dump_kernel(k, std::cerr);
        }
        if (optimize) {
            OptStats total;
            for (auto& k : prog.kernels) {
                Optimizer opt;
                opt.run(k);
                total.folded += opt.stats.folded;
                total.simplified += opt.stats.simplified;
                total.branches_removed += opt.stats.branches_removed;
            }
            if (dumpAst) {
                std::cerr << "===== AST (after optimization) =====\n";
                for (const auto& k : prog.kernels) dump_kernel(k, std::cerr);
            }
            if (showStats)
                std::cerr << "optimizer: " << total.folded << " constant folds, " << total.simplified
                          << " simplifications, " << total.branches_removed << " dead branches removed\n";
        }

        std::string ptx = generate_ptx(prog, target);
        if (output.empty()) {
            std::cout << ptx;
        } else {
            std::ofstream out(output);
            if (!out) { std::cerr << "error: cannot write '" << output << "'\n"; return 1; }
            out << ptx;
        }
    } catch (const CompileError& e) {
        std::cerr << input << ":" << e.line << ":" << e.col << ": error: " << e.what() << "\n";
        // show the offending source line with a caret
        std::istringstream ls(src);
        std::string text;
        for (int n = 1; std::getline(ls, text); n++) {
            if (n == e.line) {
                std::cerr << "  " << text << "\n  " << std::string(static_cast<size_t>(e.col > 0 ? e.col - 1 : 0), ' ') << "^\n";
                break;
            }
        }
        return 1;
    }
    return 0;
}
