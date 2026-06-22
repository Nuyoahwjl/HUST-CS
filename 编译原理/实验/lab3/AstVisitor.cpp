#include <cassert>
#include <memory>
#include <string>

#include "utils.h"
#include "AstVisitor.h"
#include "ast.h"

using namespace frontend;

std::unique_ptr<CompileUnit> AstVisitor::compileUnit() {
    return std::move(m_compile_unit);
}

antlrcpp::Any
AstVisitor::visitCompUnit(SysyParser::CompUnitContext *const ctx) {
    std::vector<CompileUnit::Child> children;
    for (auto item : ctx->compUnitItem()) {
        if (auto decl = item->decl()) {
            auto const decls =
                    decl->accept(this).as<std::shared_ptr<std::vector<Declaration *>>>();
            for (auto d : *decls) {
                children.emplace_back(std::unique_ptr<Declaration>(d));
            }
        } else if (auto func_ = item->funcDef()) {
            auto const func = func_->accept(this).as<Function *>();
            children.emplace_back(std::unique_ptr<Function>(func));
        } else {
            assert(false);
        }
    }
    auto compile_unit = new CompileUnit(std::move(children));
    m_compile_unit.reset(compile_unit);
    return compile_unit;
}

antlrcpp::Any
AstVisitor::visitConstDecl(SysyParser::ConstDeclContext *const ctx) {
    auto const base_type_ = ctx->bType()->accept(this).as<ScalarType *>();
    std::unique_ptr<ScalarType> base_type(base_type_);
    std::vector<Declaration *> ret;
    for (auto def : ctx->constDef()) {
        Identifier ident(def->Ident()->getText());
        auto dimensions = this->visitDimensions(def->exp());
        std::unique_ptr<SysYType> type;
        if (dimensions.empty()) {
            type = std::make_unique<ScalarType>(*base_type);
        } else {
            type =
                    std::make_unique<ArrayType>(*base_type, std::move(dimensions), false);
        }
        auto const init_ = def->initVal()->accept(this).as<Initializer *>();
        std::unique_ptr<Initializer> init(init_);
        ret.push_back(new Declaration(std::move(type), std::move(ident),
                                      std::move(init), true));
    }
    return std::make_shared<std::vector<Declaration *>>(std::move(ret));
}

antlrcpp::Any AstVisitor::visitInt(SysyParser::IntContext *const ctx) {
    return new ScalarType(Int);
}

antlrcpp::Any AstVisitor::visitFloat(SysyParser::FloatContext *const ctx) {
    return new ScalarType(Float);
}

antlrcpp::Any AstVisitor::visitVarDecl(SysyParser::VarDeclContext *const ctx) {
    auto const base_type_ = ctx->bType()->accept(this).as<ScalarType *>();
    std::unique_ptr<ScalarType> base_type(base_type_);
    std::vector<Declaration *> ret;
    for (auto def : ctx->varDef()) {
        Identifier ident(def->Ident()->getText());
        auto dimensions = this->visitDimensions(def->exp());
        std::unique_ptr<SysYType> type;
        if (dimensions.empty()) {
            type = std::make_unique<ScalarType>(*base_type);
        } else {
            type =
                    std::make_unique<ArrayType>(*base_type, std::move(dimensions), false);
        }
        std::unique_ptr<Initializer> init;
        if (auto init_val = def->initVal()) {
            init.reset(init_val->accept(this).as<Initializer *>());
        }
        ret.push_back(new Declaration(std::move(type), std::move(ident),
                                      std::move(init), false));
    }
    return std::make_shared<std::vector<Declaration *>>(std::move(ret));
}

antlrcpp::Any AstVisitor::visitInit(SysyParser::InitContext *const ctx) {
    auto expr_ = ctx->exp()->accept(this).as<Expression *>();
    std::unique_ptr<Expression> expr(expr_);
    return new Initializer(std::move(expr));
}

antlrcpp::Any
AstVisitor::visitInitList(SysyParser::InitListContext *const ctx) {
    std::vector<std::unique_ptr<Initializer>> values;
    for (auto init : ctx->initVal()) {
        auto const value = init->accept(this).as<Initializer *>();
        values.emplace_back(value);
    }
    return new Initializer(std::move(values));
}

antlrcpp::Any AstVisitor::visitFuncDef(SysyParser::FuncDefContext *const ctx) {
    auto const type_ = ctx->funcType()->accept(this).as<ScalarType *>();
    std::unique_ptr<ScalarType> type(type_);
    Identifier ident(ctx->Ident()->getText(), false);
    std::vector<std::unique_ptr<Parameter>> params;
    if (auto params_ = ctx->funcFParams()) {
        for (auto param_ : params_->funcFParam()) {
            auto const param = param_->accept(this).as<Parameter *>();
            params.emplace_back(param);
        }
    }
    auto const body_ = ctx->block()->accept(this).as<Block *>();
    std::unique_ptr<Block> body(body_);
    return new Function(std::move(type), std::move(ident), std::move(params),
                        std::move(body));
}

antlrcpp::Any AstVisitor::visitVoid(SysyParser::VoidContext *const ctx) {
    return static_cast<ScalarType *>(nullptr);
}

antlrcpp::Any
AstVisitor::visitScalarParam(SysyParser::ScalarParamContext *const ctx) {
    auto const type_ = ctx->bType()->accept(this).as<ScalarType *>();
    std::unique_ptr<SysYType> type(type_);
    Identifier ident(ctx->Ident()->getText());
    return new Parameter(std::move(type), std::move(ident));
}

antlrcpp::Any AstVisitor::visitArrayParam(SysyParser::ArrayParamContext *ctx) {
    auto const basic_type_ = ctx->bType()->accept(this).as<ScalarType *>();
    std::unique_ptr<ScalarType> basic_type(basic_type_);
    Identifier ident(ctx->Ident()->getText());
    auto dimensions = this->visitDimensions(ctx->exp());
    std::unique_ptr<SysYType> type(
            new ArrayType(*basic_type, std::move(dimensions), true));
    return new Parameter(std::move(type), std::move(ident));
}

antlrcpp::Any AstVisitor::visitBlock(SysyParser::BlockContext *const ctx) {
    std::vector<Block::Child> children;
    for (auto item : ctx->blockItem()) {
        if (auto decl = item->decl()) {
            auto const decls =
                    decl->accept(this).as<std::shared_ptr<std::vector<Declaration *>>>();
            for (auto d : *decls) {
                children.emplace_back(std::unique_ptr<Declaration>(d));
            }
        } else if (auto stmt_ = item->stmt()) {
            auto const stmt = stmt_->accept(this).as<Statement *>();
            children.emplace_back(std::unique_ptr<Statement>(stmt));
        } else {
            assert(false);
        }
    }
    return new Block(std::move(children));
}

antlrcpp::Any AstVisitor::visitAssign(SysyParser::AssignContext *const ctx) {
    auto const lhs_ = ctx->lVal()->accept(this).as<LValue *>();
    std::unique_ptr<LValue> lhs(lhs_);
    auto const rhs_ = ctx->exp()->accept(this).as<Expression *>();
    std::unique_ptr<Expression> rhs(rhs_);
    auto const ret = new Assignment(std::move(lhs), std::move(rhs));
    return static_cast<Statement *>(ret);
}

antlrcpp::Any
AstVisitor::visitExprStmt(SysyParser::ExprStmtContext *const ctx) {
    std::unique_ptr<Expression> expr;
    if (auto expr_ = ctx->exp()) {
        expr.reset(expr_->accept(this).as<Expression *>());
    }
    auto const ret = new ExprStmt(std::move(expr));
    return static_cast<Statement *>(ret);
}

antlrcpp::Any
AstVisitor::visitBlockStmt(SysyParser::BlockStmtContext *const ctx) {
    auto const ret = ctx->block()->accept(this).as<Block *>();
    return static_cast<Statement *>(ret);
}

antlrcpp::Any AstVisitor::visitIfElse(SysyParser::IfElseContext *const ctx) {
    auto const cond_ = ctx->cond()->accept(this).as<Expression *>();
    std::unique_ptr<Expression> cond(cond_);
    auto const then_ = ctx->stmt(0)->accept(this).as<Statement *>();
    std::unique_ptr<Statement> then(then_);
    std::unique_ptr<Statement> else_;
    if (ctx->Else() != nullptr) {
        else_.reset(ctx->stmt(1)->accept(this).as<Statement *>());
    }
    auto const ret =
            new IfElse(std::move(cond), std::move(then), std::move(else_));
    return static_cast<Statement *>(ret);
}

// ***************  TODO:　你需要完成该方法  ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
antlrcpp::Any AstVisitor::visitWhile(SysyParser::WhileContext *const ctx) {
    auto const cond_ = ctx->cond()->accept(this).as<Expression *>();
    std::unique_ptr<Expression> cond(cond_);

    auto const body_ = ctx->stmt()->accept(this).as<Statement *>();
    std::unique_ptr<Statement> body(body_);

    auto const ret = new While(std::move(cond), std::move(body)); // 请在补充代码的同时，正确填写两个形参
    return static_cast<Statement *>(ret);
}

// **************   END  ***********************************************

antlrcpp::Any AstVisitor::visitBreak(SysyParser::BreakContext *const ctx) {
    auto const ret = new Break;
    return static_cast<Statement *>(ret);
}

antlrcpp::Any
AstVisitor::visitContinue(SysyParser::ContinueContext *const ctx) {
    auto const ret = new Continue;
    return static_cast<Statement *>(ret);
}

antlrcpp::Any AstVisitor::visitReturn(SysyParser::ReturnContext *const ctx) {
    std::unique_ptr<Expression> res;
    if (auto exp = ctx->exp()) {
        res.reset(exp->accept(this).as<Expression *>());
    }
    auto const ret = new Return(std::move(res));
    return static_cast<Statement *>(ret);
}

antlrcpp::Any AstVisitor::visitLVal(SysyParser::LValContext *const ctx) {
    Identifier ident(ctx->Ident()->getText());
    std::vector<std::unique_ptr<Expression>> indices;
    for (auto exp : ctx->exp()) {
        auto const index = exp->accept(this).as<Expression *>();
        indices.emplace_back(index);
    }
    return new LValue(std::move(ident), std::move(indices));
}

antlrcpp::Any
AstVisitor::visitPrimaryExp_(SysyParser::PrimaryExp_Context *const ctx) {
    if (ctx->number()) {
        return ctx->number()->accept(this);
    } else {
        assert(ctx->exp());
        return ctx->exp()->accept(this);
    }
}

antlrcpp::Any
AstVisitor::visitLValExpr(SysyParser::LValExprContext *const ctx) {
    auto const lval = ctx->lVal()->accept(this).as<LValue *>();
    return static_cast<Expression *>(lval);
}

antlrcpp::Any
AstVisitor::visitDecIntConst(SysyParser::DecIntConstContext *const ctx) {
    return int(std::stoll(ctx->getText(), nullptr, 10));
}

antlrcpp::Any
AstVisitor::visitOctIntConst(SysyParser::OctIntConstContext *const ctx) {
    return int(std::stoll(ctx->getText(), nullptr, 8));
}

antlrcpp::Any
AstVisitor::visitHexIntConst(SysyParser::HexIntConstContext *const ctx) {
    return int(std::stoll(ctx->getText(), nullptr, 16));
}

antlrcpp::Any
AstVisitor::visitDecFloatConst(SysyParser::DecFloatConstContext *const ctx) {
    return std::stof(ctx->getText());
}

antlrcpp::Any
AstVisitor::visitHexFloatConst(SysyParser::HexFloatConstContext *const ctx) {
    return std::stof(ctx->getText());
}

antlrcpp::Any AstVisitor::visitCall(SysyParser::CallContext *const ctx) {
    Identifier ident(ctx->Ident()->getText(), false);
    std::vector<Call::Argument> args;
    auto args_ctx = ctx->funcRParams();
    if (args_ctx) {
        for (auto arg_ : args_ctx->funcRParam()) {
            if (auto exp = arg_->exp()) {
                auto const arg = exp->accept(this).as<Expression *>();
                args.emplace_back(std::unique_ptr<Expression>(arg));
            } else if (auto str = arg_->stringConst()) {
                auto arg = str->accept(this).as<std::shared_ptr<std::string>>();
                args.emplace_back(std::move(*arg));
            } else {
                assert(false);
            }
        }
    }
    auto const ret =
            new Call(std::move(ident), std::move(args), ctx->getStart()->getLine());
    return static_cast<Expression *>(ret);
}

antlrcpp::Any
AstVisitor::visitUnaryAdd(SysyParser::UnaryAddContext *const ctx) {
    auto const operand = ctx->unaryExp()->accept(this).as<Expression *>();
    auto const ret =
            new UnaryExpr(UnaryOp::Add, std::unique_ptr<Expression>(operand));
    return static_cast<Expression *>(ret);
}

antlrcpp::Any
AstVisitor::visitUnarySub(SysyParser::UnarySubContext *const ctx) {
    auto const operand = ctx->unaryExp()->accept(this).as<Expression *>();
    auto const ret =
            new UnaryExpr(UnaryOp::Sub, std::unique_ptr<Expression>(operand));
    return static_cast<Expression *>(ret);
}

antlrcpp::Any AstVisitor::visitNot(SysyParser::NotContext *const ctx) {
    auto const operand = ctx->unaryExp()->accept(this).as<Expression *>();
    auto const ret =
            new UnaryExpr(UnaryOp::Not, std::unique_ptr<Expression>(operand));
    return static_cast<Expression *>(ret);
}

antlrcpp::Any
AstVisitor::visitStringConst(SysyParser::StringConstContext *const ctx) {
    return std::make_shared<std::string>(ctx->getText());
}

antlrcpp::Any AstVisitor::visitMul(SysyParser::MulContext *const ctx) {
    auto const lhs = ctx->mulExp()->accept(this).as<Expression *>();
    auto const rhs = ctx->unaryExp()->accept(this).as<Expression *>();
    auto const ret =
            new BinaryExpr(BinaryOp::Mul, std::unique_ptr<Expression>(lhs),
                           std::unique_ptr<Expression>(rhs));
    return static_cast<Expression *>(ret);
}

antlrcpp::Any AstVisitor::visitDiv(SysyParser::DivContext *const ctx) {
    auto const lhs = ctx->mulExp()->accept(this).as<Expression *>();
    auto const rhs = ctx->unaryExp()->accept(this).as<Expression *>();
    auto const ret =
            new BinaryExpr(BinaryOp::Div, std::unique_ptr<Expression>(lhs),
                           std::unique_ptr<Expression>(rhs));
    return static_cast<Expression *>(ret);
}

antlrcpp::Any AstVisitor::visitMod(SysyParser::ModContext *const ctx) {
    auto const lhs = ctx->mulExp()->accept(this).as<Expression *>();
    auto const rhs = ctx->unaryExp()->accept(this).as<Expression *>();
    auto const ret =
            new BinaryExpr(BinaryOp::Mod, std::unique_ptr<Expression>(lhs),
                           std::unique_ptr<Expression>(rhs));
    return static_cast<Expression *>(ret);
}

antlrcpp::Any AstVisitor::visitAdd(SysyParser::AddContext *const ctx) {
    auto const lhs = ctx->addExp()->accept(this).as<Expression *>();
    auto const rhs = ctx->mulExp()->accept(this).as<Expression *>();
    auto const ret =
            new BinaryExpr(BinaryOp::Add, std::unique_ptr<Expression>(lhs),
                           std::unique_ptr<Expression>(rhs));
    return static_cast<Expression *>(ret);
}

antlrcpp::Any AstVisitor::visitSub(SysyParser::SubContext *const ctx) {
    auto const lhs = ctx->addExp()->accept(this).as<Expression *>();
    auto const rhs = ctx->mulExp()->accept(this).as<Expression *>();
    auto const ret =
            new BinaryExpr(BinaryOp::Sub, std::unique_ptr<Expression>(lhs),
                           std::unique_ptr<Expression>(rhs));
    return static_cast<Expression *>(ret);
}

antlrcpp::Any AstVisitor::visitLt(SysyParser::LtContext *const ctx) {
    auto const lhs = ctx->relExp()->accept(this).as<Expression *>();
    auto const rhs = ctx->addExp()->accept(this).as<Expression *>();
    auto const ret =
            new BinaryExpr(BinaryOp::Lt, std::unique_ptr<Expression>(lhs),
                           std::unique_ptr<Expression>(rhs));
    return static_cast<Expression *>(ret);
}

antlrcpp::Any AstVisitor::visitGt(SysyParser::GtContext *const ctx) {
    auto const lhs = ctx->relExp()->accept(this).as<Expression *>();
    auto const rhs = ctx->addExp()->accept(this).as<Expression *>();
    auto const ret =
            new BinaryExpr(BinaryOp::Gt, std::unique_ptr<Expression>(lhs),
                           std::unique_ptr<Expression>(rhs));
    return static_cast<Expression *>(ret);
}

antlrcpp::Any AstVisitor::visitLeq(SysyParser::LeqContext *const ctx) {
    auto const lhs = ctx->relExp()->accept(this).as<Expression *>();
    auto const rhs = ctx->addExp()->accept(this).as<Expression *>();
    auto const ret =
            new BinaryExpr(BinaryOp::Leq, std::unique_ptr<Expression>(lhs),
                           std::unique_ptr<Expression>(rhs));
    return static_cast<Expression *>(ret);
}

antlrcpp::Any AstVisitor::visitGeq(SysyParser::GeqContext *const ctx) {
    auto const lhs = ctx->relExp()->accept(this).as<Expression *>();
    auto const rhs = ctx->addExp()->accept(this).as<Expression *>();
    auto const ret =
            new BinaryExpr(BinaryOp::Geq, std::unique_ptr<Expression>(lhs),
                           std::unique_ptr<Expression>(rhs));
    return static_cast<Expression *>(ret);
}

antlrcpp::Any AstVisitor::visitEq(SysyParser::EqContext *const ctx) {
    auto const lhs = ctx->eqExp()->accept(this).as<Expression *>();
    auto const rhs = ctx->relExp()->accept(this).as<Expression *>();
    auto const ret =
            new BinaryExpr(BinaryOp::Eq, std::unique_ptr<Expression>(lhs),
                           std::unique_ptr<Expression>(rhs));
    return static_cast<Expression *>(ret);
}

antlrcpp::Any AstVisitor::visitNeq(SysyParser::NeqContext *const ctx) {
    auto const lhs = ctx->eqExp()->accept(this).as<Expression *>();
    auto const rhs = ctx->relExp()->accept(this).as<Expression *>();
    auto const ret =
            new BinaryExpr(BinaryOp::Neq, std::unique_ptr<Expression>(lhs),
                           std::unique_ptr<Expression>(rhs));
    return static_cast<Expression *>(ret);
}

antlrcpp::Any AstVisitor::visitAnd(SysyParser::AndContext *const ctx) {
    auto const lhs = ctx->lAndExp()->accept(this).as<Expression *>();
    auto const rhs = ctx->eqExp()->accept(this).as<Expression *>();
    auto const ret =
            new BinaryExpr(BinaryOp::And, std::unique_ptr<Expression>(lhs),
                           std::unique_ptr<Expression>(rhs));
    return static_cast<Expression *>(ret);
}

antlrcpp::Any AstVisitor::visitOr(SysyParser::OrContext *const ctx) {
    auto const lhs = ctx->lOrExp()->accept(this).as<Expression *>();
    auto const rhs = ctx->lAndExp()->accept(this).as<Expression *>();
    auto const ret =
            new BinaryExpr(BinaryOp::Or, std::unique_ptr<Expression>(lhs),
                           std::unique_ptr<Expression>(rhs));
    return static_cast<Expression *>(ret);
}

antlrcpp::Any AstVisitor::visitNumber(SysyParser::NumberContext *const ctx) {
    if (ctx->intConst()) {
        auto val = ctx->intConst()->accept(this).as<IntLiteral::Value>();
        auto literal = new IntLiteral{val};
        return static_cast<Expression *>(literal);
    }
    if (ctx->floatConst()) {
        auto val = ctx->floatConst()->accept(this).as<FloatLiteral::Value>();
        auto literal = new FloatLiteral{val};
        return static_cast<Expression *>(literal);
    }
    assert(false);
    return static_cast<Expression *>(nullptr);
}

std::vector<std::unique_ptr<Expression>>
AstVisitor::visitDimensions(const std::vector<SysyParser::ExpContext *> &ctxs) {
    std::vector<std::unique_ptr<Expression>> ret;
    for (auto expr_ctx : ctxs) {
        auto expr_ = expr_ctx->accept(this).as<Expression *>();
        ret.emplace_back(expr_);
    }
    return ret;
}
