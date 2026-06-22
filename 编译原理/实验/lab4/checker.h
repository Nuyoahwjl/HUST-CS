#ifndef CARROTCOMPILER_CHECKER_H
#define CARROTCOMPILER_CHECKER_H

#include "ast.h"
#include "define.h"
#include "errorReporter.h"
#include <algorithm>
#include <cstdlib>
#include <list>
#include <map>
#include <string>
#include <vector>

/**
 * @brief 语义分析器类，用于检查源代码中的语法和语义错误
 */
class Checker : public Visitor {
public:
  explicit Checker(ErrorReporter &e) : err(e) {}
  void visit(CompUnitAST &ast) override;
  void visit(DeclDefAST &ast) override;
  void visit(DeclAST &ast) override;
  void visit(DefAST &ast) override;
  void visit(InitValAST &ast) override;
  void visit(FuncDefAST &ast) override;
  void visit(FuncFParamAST &ast) override;
  void visit(BlockAST &ast) override;
  void visit(BlockItemAST &ast) override;
  void visit(StmtAST &ast) override;
  void visit(ReturnStmtAST &ast) override;
  void visit(SelectStmtAST &ast) override;
  void visit(IterationStmtAST &ast) override;
  void visit(AddExpAST &ast) override;
  void visit(MulExpAST &ast) override;
  void visit(UnaryExpAST &ast) override;
  void visit(PrimaryExpAST &ast) override;
  void visit(LValAST &ast) override;
  void visit(NumberAST &ast) override;
  void visit(CallAST &ast) override;
  void visit(RelExpAST &ast) override;
  void visit(EqExpAST &ast) override;
  void visit(LAndExpAST &ast) override;
  void visit(LOrExpAST &ast) override;

private:
  ErrorReporter &err;
  bool Expr_int{};          // 表达式的值是否为整型
  bool start_of_new_func{}; // 是否是一个新函数定义的开始
  int Expr_value{};         // 表达式的值（仅在框架支持常量求值时使用）
  TYPE current_func_return_type{}; // 当前正在检查的函数返回类型

  // 用于表示符号表中的各种类型信息的结构体
  struct Entry {
    bool is_array{};                       // 是否为数组
    bool is_func{};                        // 是否为函数
    TYPE type{};                           // 变量类型 or 返回值类型
    int array_length{};                    // 如果是数组，则为维数
    std::vector<int> arlen_value;          // 数组每一维长度；未知/指针维用 -1
    std::vector<struct Entry> func_params; // 函数参数列表
  };

  Entry current_type; // 当前表达式的类型信息
  // list 的每一项是一个作用域顶层的符号表；front 为当前最内层作用域
  std::list<std::map<std::string, Entry>> table;

  [[noreturn]] void ReportAndExit(ErrorType type, std::string name) {
    err.error(type, name);
    exit(int(type));
  }

  bool SameParamType(const Entry &formal, const Entry &actual) const {
    if (formal.type != actual.type || formal.is_array != actual.is_array) {
      return false;
    }
    if (!formal.is_array) {
      return true;
    }
    if (formal.array_length != actual.array_length) {
      return false;
    }
    // 第一维形参可能是指针维（-1），不要求与实参数值相同。
    // 由于部分框架不做常量求值，维度长度只在双方都可用且形参不是 -1 时比较。
    const auto n = std::min(formal.arlen_value.size(), actual.arlen_value.size());
    for (size_t i = 0; i < n; ++i) {
      if (formal.arlen_value[i] != -1 && actual.arlen_value[i] != 0 &&
          formal.arlen_value[i] != actual.arlen_value[i]) {
        return false;
      }
    }
    return true;
  }

  Entry *Lookup(const std::string &name) {
    for (auto it = table.begin(); it != table.end(); ++it) {
      auto entry = it->find(name);
      if (entry != it->end()) {
        return &entry->second;
      }
    }
    return nullptr;
  }

  bool InsertVar(const DeclAST &node) {
    for (auto &def : node.defList) {
      Entry tmp;
      tmp.is_func = false;
      tmp.type = node.bType;

      if (!def->arrays.empty()) {
        tmp.is_array = true;
        for (auto &exp : def->arrays) {
          exp->accept(*this);
          if (!Expr_int) {
            ReportAndExit(ErrorType::ArrayIndexNotInt, *def->id);
          }
          tmp.arlen_value.push_back(Expr_value);
        }
        tmp.array_length = static_cast<int>(def->arrays.size());
      } else {
        tmp.is_array = false;
        tmp.array_length = 0;
      }

      auto result = table.front().insert({*def->id, tmp});
      if (!result.second) {
        err.error(ErrorType::VarDuplicated, *def->id);
        return false;
      }
    }
    return true;
  }

  bool InsertFunc(const FuncDefAST &node) {
    // 函数名与当前作用域中的已有符号冲突，均视为函数重复定义。
    if (table.front().find(*node.id) != table.front().end()) {
      err.error(ErrorType::FuncDuplicated, *node.id);
      return false;
    }

    std::map<std::string, Entry> new_table;

    Entry tmp;
    tmp.is_array = false;
    tmp.is_func = true;
    tmp.type = node.funcType;

    for (auto &param : node.funcFParamList) {
      Entry p;
      p.is_func = false;
      p.type = param->bType;

      if (param->isArray) {
        p.is_array = true;
        for (auto &exp : param->arrays) {
          if (exp == nullptr) {
            p.arlen_value.push_back(-1);
          } else {
            exp->accept(*this);
            if (!Expr_int) {
              ReportAndExit(ErrorType::ArrayIndexNotInt, *param->id);
            }
            p.arlen_value.push_back(Expr_value);
          }
        }
        p.array_length = static_cast<int>(param->arrays.size());
      } else {
        p.is_array = false;
        p.array_length = 0;
      }

      tmp.func_params.push_back(p);

      auto result = new_table.insert({*param->id, p});
      if (!result.second) {
        ReportAndExit(ErrorType::VarDuplicated, *param->id);
      }
    }

    auto result = table.front().insert({*node.id, tmp});
    if (result.second) {
      table.push_front(new_table);
      return true;
    }
    return false;
  }

  void make_new_table() {
    std::map<std::string, Entry> new_table;
    table.push_front(new_table);
  }

  void delete_table() { table.pop_front(); }

  Entry *find_func() {
    for (auto &it : table) {
      for (auto &entry : it) {
        if (entry.second.is_func) {
          return &entry.second;
        }
      }
    }
    return nullptr;
  }
};

#endif // CARROTCOMPILER_CHECKER_H
