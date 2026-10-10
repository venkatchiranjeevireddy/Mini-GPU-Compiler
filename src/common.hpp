// common.hpp - shared error type used by every compiler stage.
#pragma once
#include <stdexcept>
#include <string>

struct CompileError : std::runtime_error {
    int line, col;
    CompileError(int l, int c, const std::string& msg) : std::runtime_error(msg), line(l), col(c) {}
};
