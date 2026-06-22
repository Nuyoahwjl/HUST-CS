#include "checker.h"

/* 静态语义分析
 1. Redefined Variable 变量重复定义/函数形参重复定义
 2. Redefined Function 函数重复定义
 3. Use Undefined Variable 使用未定义变量
 4. Use Undefined Function 使用未定义函数
 5. Can not Match Function Parameters 函数参数/类型不匹配
 6. Func return type not match 函数返回值类型不匹配
 7. Array index not int 数组下标不是整数
 8. Break not in loop break语句不在循环中
 9. Continue not in loop continue语句不在循环中
 10. Visit non-array variable in the form of subscript variables
*/

void Checker::visit(CompUnitAST &ast) {
  make_new_table();
  for (auto &decl : ast.declDefList) {
    decl->accept(*this);
  }
}

void Checker::visit(DeclDefAST &ast) {
  if (ast.Decl) {
    ast.Decl->accept(*this);
  }
  if (ast.funcDef) {
    ast.funcDef->accept(*this);
  }
}

void Checker::visit(DeclAST &ast) {
  for (auto &def : ast.defList) {
    def->accept(*this);
  }
  if (!InsertVar(ast)) {
    exit(int(ErrorType::VarDuplicated));
  }
}

void Checker::visit(DefAST &ast) {
  if (ast.initVal) {
    ast.initVal->accept(*this);
  }
}

void Checker::visit(InitValAST &ast) {
  if (ast.exp) {
    ast.exp->accept(*this);
  } else {
    for (auto &initVal : ast.initValList) {
      initVal->accept(*this);
    }
  }
}

void Checker::visit(FuncFParamAST &ast) {
  for (auto &exp : ast.arrays) {
    if (exp) {
      exp->accept(*this);
      if (!Expr_int) {
        ReportAndExit(ErrorType::ArrayIndexNotInt, *ast.id);
      }
    }
  }
}

void Checker::visit(ReturnStmtAST &ast) {
  TYPE ret_type = TYPE::TYPE_VOID;
  if (ast.exp) {
    ast.exp->accept(*this);
    ret_type = current_type.type;
  }

  if (ret_type != current_func_return_type) {
    ReportAndExit(ErrorType::FuncReturnTypeNotMatch, "return");
  }
}

void Checker::visit(FuncDefAST &ast) {
  if (!InsertFunc(ast)) {
    exit(int(ErrorType::FuncDuplicated));
  }

  current_func_return_type = ast.funcType;
  start_of_new_func = true;
  ast.block->accept(*this);
}

void Checker::visit(BlockAST &ast) {
  // 一个新的函数在 InsertFunc 时已经创建了形参/函数体作用域。
  if (start_of_new_func) {
    start_of_new_func = false;
  } else {
    make_new_table();
  }

  for (auto &item : ast.blockItemList) {
    item->is_inloop = ast.is_inloop;
    item->accept(*this);
  }

  delete_table();
}

void Checker::visit(BlockItemAST &ast) {
  if (ast.stmt) {
    ast.stmt->is_inloop = ast.is_inloop;
    ast.stmt->accept(*this);
  }
  if (ast.decl) {
    ast.decl->accept(*this);
  }
}

void Checker::visit(StmtAST &ast) {
  if (ast.selectStmt) {
    ast.selectStmt->is_inloop = ast.is_inloop;
    ast.selectStmt->accept(*this);
  }
  if (ast.block) {
    ast.block->is_inloop = ast.is_inloop;
    ast.block->accept(*this);
  }
  if (ast.iterationStmt) {
    ast.iterationStmt->is_inloop = true;
    ast.iterationStmt->accept(*this);
  }
  if (ast.returnStmt) {
    ast.returnStmt->accept(*this);
  }
  if (ast.lVal) {
    ast.lVal->accept(*this);
  }
  if (ast.exp) {
    ast.exp->accept(*this);
  }

  if (ast.sType == STYPE::BRE) {
    if (!ast.is_inloop) {
      ReportAndExit(ErrorType::BreakNotInLoop, "break");
    }
  } else if (ast.sType == STYPE::CONT) {
    if (!ast.is_inloop) {
      ReportAndExit(ErrorType::ContinueNotInLoop, "continue");
    }
  }
}

void Checker::visit(SelectStmtAST &ast) {
  if (ast.cond) {
    ast.cond->accept(*this);
  }
  if (ast.ifStmt) {
    ast.ifStmt->is_inloop = ast.is_inloop;
    ast.ifStmt->accept(*this);
  }
  if (ast.elseStmt) {
    ast.elseStmt->is_inloop = ast.is_inloop;
    ast.elseStmt->accept(*this);
  }
}

void Checker::visit(IterationStmtAST &ast) {
  if (ast.cond) {
    ast.cond->accept(*this);
  }
  if (ast.stmt) {
    ast.stmt->is_inloop = true;
    ast.stmt->accept(*this);
  }
}

void Checker::visit(AddExpAST &ast) {
  bool has_left = static_cast<bool>(ast.addExp);
  bool has_right = static_cast<bool>(ast.mulExp);

  Entry left_type;
  if (has_left) {
    ast.addExp->accept(*this);
    left_type = current_type;
  }

  if (has_right) {
    ast.mulExp->accept(*this);
  }

  if (has_left && has_right) {
    Entry right_type = current_type;
    current_type = Entry{};
    current_type.is_array = false;
    current_type.is_func = false;
    current_type.type = (left_type.type == TYPE::TYPE_FLOAT ||
                         right_type.type == TYPE::TYPE_FLOAT)
                            ? TYPE::TYPE_FLOAT
                            : TYPE::TYPE_INT;
    Expr_int = (current_type.type == TYPE::TYPE_INT);
  }
}

void Checker::visit(MulExpAST &ast) {
  bool has_left = static_cast<bool>(ast.mulExp);
  bool has_right = static_cast<bool>(ast.unaryExp);

  Entry left_type;
  if (has_left) {
    ast.mulExp->accept(*this);
    left_type = current_type;
  }

  if (has_right) {
    ast.unaryExp->accept(*this);
  }

  if (has_left && has_right) {
    Entry right_type = current_type;
    current_type = Entry{};
    current_type.is_array = false;
    current_type.is_func = false;
    current_type.type = (left_type.type == TYPE::TYPE_FLOAT ||
                         right_type.type == TYPE::TYPE_FLOAT)
                            ? TYPE::TYPE_FLOAT
                            : TYPE::TYPE_INT;
    Expr_int = (current_type.type == TYPE::TYPE_INT);
  }
}

void Checker::visit(UnaryExpAST &ast) {
  if (ast.primaryExp) {
    ast.primaryExp->accept(*this);
  }
  if (ast.unaryExp) {
    ast.unaryExp->accept(*this);
  }
  if (ast.call) {
    ast.call->accept(*this);
  }
}

void Checker::visit(PrimaryExpAST &ast) {
  if (ast.exp) {
    ast.exp->accept(*this);
  }
  if (ast.lval) {
    ast.lval->accept(*this);
  }
  if (ast.number) {
    ast.number->accept(*this);
  }
}

void Checker::visit(LValAST &ast) {
  auto str = ast.id.get();
  Entry *entry = Lookup(*str);
  if (entry == nullptr || entry->is_func) {
    ReportAndExit(ErrorType::VarUnknown, *ast.id);
  }

  for (auto &exp : ast.arrays) {
    if (exp) {
      exp->accept(*this);
      if (!Expr_int) {
        ReportAndExit(ErrorType::ArrayIndexNotInt, *ast.id);
      }
    }
  }

  if (!entry->is_array && !ast.arrays.empty()) {
    ReportAndExit(ErrorType::VisitVariableError, *ast.id);
  }

  if (entry->is_array && ast.arrays.size() > static_cast<size_t>(entry->array_length)) {
    ReportAndExit(ErrorType::VisitVariableError, *ast.id);
  }

  current_type = *entry;
  current_type.is_func = false;
  if (entry->is_array) {
    int used_dim = static_cast<int>(ast.arrays.size());
    int remain_dim = entry->array_length - used_dim;
    current_type.array_length = remain_dim;
    current_type.is_array = (remain_dim > 0);
    if (used_dim > 0 && used_dim <= static_cast<int>(current_type.arlen_value.size())) {
      current_type.arlen_value.erase(current_type.arlen_value.begin(),
                                     current_type.arlen_value.begin() + used_dim);
    }
  }

  Expr_int = (!current_type.is_array && current_type.type == TYPE::TYPE_INT);
}

void Checker::visit(NumberAST &ast) {
  current_type = Entry{};
  Expr_int = ast.isInt;
  current_type.type = ast.isInt ? TYPE::TYPE_INT : TYPE::TYPE_FLOAT;
}

void Checker::visit(CallAST &ast) {
  const std::string &name = *ast.id;

  auto set_call_type = [&](TYPE type) {
    current_type = Entry{};
    current_type.type = type;
    current_type.is_array = false;
    current_type.is_func = false;
    Expr_int = (type == TYPE::TYPE_INT);
  };

  // 库函数不在用户符号表中；仍然访问实参，以便发现实参里的未定义变量等错误。
  if (name == "getint" || name == "getch") {
    for (auto &exp : ast.funcCParamList) {
      exp->accept(*this);
    }
    set_call_type(TYPE::TYPE_INT);
    return;
  }
  if (name == "getfloat") {
    for (auto &exp : ast.funcCParamList) {
      exp->accept(*this);
    }
    set_call_type(TYPE::TYPE_FLOAT);
    return;
  }
  if (name == "getarray" || name == "get_float_array") {
    for (auto &exp : ast.funcCParamList) {
      exp->accept(*this);
    }
    set_call_type(TYPE::TYPE_INT);
    return;
  }
  if (name == "putint" || name == "putfloat" || name == "putch" ||
      name == "putarray" || name == "put_float_array") {
    for (auto &exp : ast.funcCParamList) {
      exp->accept(*this);
    }
    set_call_type(TYPE::TYPE_VOID);
    return;
  }

  Entry *entry = Lookup(name);
  if (entry == nullptr || !entry->is_func) {
    ReportAndExit(ErrorType::FuncUnknown, name);
  }

  if (entry->func_params.size() != ast.funcCParamList.size()) {
    ReportAndExit(ErrorType::FuncParamsNotMatch, name);
  }

  auto formal_it = entry->func_params.begin();
  for (auto &actual : ast.funcCParamList) {
    actual->accept(*this);
    if (!SameParamType(*formal_it, current_type)) {
      ReportAndExit(ErrorType::FuncParamsNotMatch, name);
    }
    ++formal_it;
  }

  set_call_type(entry->type);
}

void Checker::visit(RelExpAST &ast) {
  if (ast.relExp) {
    ast.relExp->accept(*this);
  }
  if (ast.addExp) {
    ast.addExp->accept(*this);
  }
  current_type = Entry{};
  Expr_int = false;
  current_type.type = TYPE::TYPE_BOOL;
}

void Checker::visit(EqExpAST &ast) {
  if (ast.eqExp) {
    ast.eqExp->accept(*this);
  }
  if (ast.relExp) {
    ast.relExp->accept(*this);
  }
  current_type = Entry{};
  Expr_int = false;
  current_type.type = TYPE::TYPE_BOOL;
}

void Checker::visit(LAndExpAST &ast) {
  if (ast.lAndExp) {
    ast.lAndExp->accept(*this);
  }
  if (ast.eqExp) {
    ast.eqExp->accept(*this);
  }
  current_type = Entry{};
  Expr_int = false;
  current_type.type = TYPE::TYPE_BOOL;
}

void Checker::visit(LOrExpAST &ast) {
  if (ast.lOrExp) {
    ast.lOrExp->accept(*this);
  }
  if (ast.lAndExp) {
    ast.lAndExp->accept(*this);
  }
  current_type = Entry{};
  Expr_int = false;
  current_type.type = TYPE::TYPE_BOOL;
}
