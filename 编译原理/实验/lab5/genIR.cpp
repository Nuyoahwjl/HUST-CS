#include "genIR.h"

#define CONST_INT(num) new ConstantInt(module->int32_ty_, num)
#define CONST_FLOAT(num) new ConstantFloat(module->float32_ty_, num)
#define VOID_T (module->void_ty_)
#define INT1_T (module->int1_ty_)
#define INT32_T  (module->int32_ty_)
#define FLOAT_T  (module->float32_ty_)
#define INT32PTR_T (module->get_pointer_type(module->int32_ty_))
#define FLOATPTR_T (module->get_pointer_type(module->float32_ty_))

// store temporary value
Type* curType;                          //当前decl类型
bool isConst;                           //当前decl是否是const
bool useConst = false;                  //计算是否使用常量
std::vector<Type *> params;             //函数形参类型表
std::vector<std::string> paramNames;    //函数形参名表
Value *retAlloca = nullptr;             //返回值
BasicBlock *retBB = nullptr;            //返回语句块
bool isNewFunc = false;                 //判断是否为新函数，用来处理函数作用域问题
bool requireLVal = false;               //告诉LVal节点不需要发射load指令
Function *currentFunction = nullptr;    //当前函数
Value *recentVal = nullptr;             //最近的表达式的value
BasicBlock *whileCondBB = nullptr;      //while语句cond分支
BasicBlock *trueBB = nullptr;           //通用true分支，即while和if为真时所跳转的基本块
BasicBlock *falseBB = nullptr;          //通用false分支，即while和if为假时所跳转的基本块
BasicBlock * whileFalseBB;              //while语句false分支，用于break跳转
int id = 1;                             //recent标号
bool has_br = false;            //一个BB中是否已经出现了br
bool is_single_exp = false;     //作为单独的exp语句出现，形如 "exp;"


//判断得到的赋值与声明类型是否一致，并做转换
void GenIR::checkInitType() const {
    if (curType == INT32_T) {
        if (dynamic_cast<ConstantFloat*>(recentVal)) {
            auto temp = dynamic_cast<ConstantFloat*>(recentVal);
            recentVal = CONST_INT((int)temp->value_);
        } else if (recentVal->type_->tid_ == Type::FloatTyID) {
            recentVal = builder->create_fptosi(recentVal, INT32_T);
        }
    } else if (curType == FLOAT_T) {
        if (dynamic_cast<ConstantInt*>(recentVal)) {
            auto temp = dynamic_cast<ConstantInt*>(recentVal);
            recentVal = CONST_FLOAT(temp->value_);
        } else if (recentVal->type_->tid_ == Type::IntegerTyID) {
            recentVal = builder->create_sitofp(recentVal, FLOAT_T);
        }
    }
}

void GenIR::visit(CompUnitAST &ast) {
    for (const auto &def : ast.declDefList) {
        def->accept(*this);
    }
}

void GenIR::visit(DeclDefAST &ast) {
    if (ast.Decl != nullptr) {
        ast.Decl->accept(*this);
    } else {
        ast.funcDef->accept(*this);
    }
}

void GenIR::visit(DeclAST &ast) {
    isConst = ast.isConst;
    curType = ast.bType == TYPE_INT ? INT32_T : FLOAT_T;
    for (auto &def : ast.defList) {
        def->accept(*this);
    }
}

void GenIR::visit(DefAST &ast) {
    string varName = *ast.id;
    //全局变量或常量
    if (scope.in_global()) {
        if (ast.arrays.empty()) {   //不是数组，即全局量
            if (ast.initVal == nullptr) { //无初始化
                if (isConst) cout << "no initVal when define const!" << endl; //无初始化全局常量报错
                //无初始化全局量一定是变量
                GlobalVariable* var;
                if (curType == INT32_T)
                    var = new GlobalVariable(varName, module.get(), curType, false, CONST_INT(0));
                else var = new GlobalVariable(varName, module.get(), curType, false, CONST_FLOAT(0));
                scope.push(varName, var);
            } else { //有初始化
                useConst = true;
                ast.initVal->accept(*this);
                useConst = false;
                checkInitType();
                if (isConst) {
                    scope.push(varName, recentVal);   //单个常量定义不用new GlobalVariable
                } else { //全局变量
                    auto initializer = static_cast<Constant *>(recentVal);
                    GlobalVariable* var;
                    var = new GlobalVariable(varName, module.get(), curType, false, initializer);
                    scope.push(varName, var);
                }
            }
        } else {        //是数组，即全局数组量
            vector<int> dimensions;  //数组各维度; [2][3][4]对应
            useConst = true;
            //获取数组各维度
            for (auto &exp : ast.arrays) {
                exp->accept(*this);
                int dimension = dynamic_cast<ConstantInt*>(recentVal)->value_;
                dimensions.push_back(dimension);
            }
            useConst = false;
            vector<ArrayType*> arrayTys(dimensions.size()); //数组类型, {[2 x [3 x [4 x i32]]], [3 x [4 x i32]], [4 x i32]}
            for (int i = dimensions.size() - 1; i >= 0; i--) {
                if (i == dimensions.size() - 1) arrayTys[i] = module->get_array_type(curType, dimensions[i]);
                else arrayTys[i] = module->get_array_type(arrayTys[i + 1], dimensions[i]);
            }
            //无初始化或者初始化仅为大括号
            if (ast.initVal == nullptr || ast.initVal->initValList.empty()) {
                auto init = new ConstantZero(arrayTys[0]);
                auto var = new GlobalVariable(varName, module.get(), arrayTys[0], isConst, init);
                scope.push(varName, var);
            } else {
                useConst = true; //全局数组量的初始值必为常量
                auto init = globalInit(dimensions, arrayTys, 0, ast.initVal->initValList);
                useConst = false;
                auto var = new GlobalVariable(varName, module.get(), arrayTys[0], isConst, init);
                scope.push(varName, var);
            }
        }
        return;
    }


    //局部变量或常量
    if (ast.arrays.empty()) {   //不是数组，即普通局部量
        if (ast.initVal == nullptr) {   //无初始化
            if (isConst) cout << "no initVal when define const!" << endl;   //无初始化局部常量报错
            else { //无初始化变量
                AllocaInst *varAlloca;
                varAlloca = builder->create_alloca(curType);
                scope.push(varName, varAlloca);
            }
        } else { //有初始化
            ast.initVal->accept(*this);
            checkInitType();
            if (isConst) {
                scope.push(varName, recentVal);  //单个常量定义不用create_alloca
            } else {
                AllocaInst *varAlloca;
                varAlloca = builder->create_alloca(curType);
                scope.push(varName, varAlloca);
                builder->create_store(recentVal, varAlloca);
            }
        }
    } else {    //局部数组量
        vector<int> dimensions(ast.arrays.size()), dimensionsCnt((ast.arrays.size()));  //数组各维度, [2][3][4]对应; 次维度数组元素个数, [24][12][4]
        int totalByte = 1; //存储总共的字节数
        useConst = true;
        //获取数组各维度
        for (int i = dimensions.size() - 1; i>= 0; i--) {
            ast.arrays[i]->accept(*this);
            int dimension = dynamic_cast<ConstantInt*>(recentVal)->value_;
            totalByte *= dimension;
            dimensions[i] = dimension;
            dimensionsCnt[i] = totalByte;
        }
        totalByte *= 4; //计算字节数
        useConst = false;
        ArrayType *arrayTy; //数组类型
        for (int i = dimensions.size() - 1; i >= 0; i--) {
            if (i == dimensions.size() - 1) arrayTy = module->get_array_type(curType, dimensions[i]);
            else arrayTy = module->get_array_type(arrayTy, dimensions[i]);
        }
        auto arrayAlloc = builder->create_alloca(arrayTy);
        scope.push(varName, arrayAlloc);
        if (ast.initVal == nullptr) { //无初始化
            if (isConst) cout << "no initVal when define const!" << endl;   //无初始化局部常量报错
            return; //无初始化变量数组无需再做处理
        }
        Value* i32P = builder->create_bitcast(arrayAlloc, INT32PTR_T);
        auto memclr = scope.find("memclr");
        builder->create_call(memclr, {i32P, CONST_INT(totalByte)}); //全部清零，但float可以清零吗
        //数组初始化时，成员exp一定是空，若initValList也是空，即是大括号，已经置零了直接返回
        if (ast.initVal->initValList.empty()) return;
        vector<Value*> idxs(dimensions.size() + 1);
        for (int i = 0; i < dimensions.size() + 1; i++) {
            idxs[i] = CONST_INT(0);
        }
        Value* ptr = builder->create_gep(arrayAlloc, idxs); //获取数组开头地址
        localInit(ptr, ast.initVal->initValList, dimensionsCnt, 1);
    }
}

//嵌套大括号数组的维度，即倒数连续0的第一个。 如[0,1,0,0]，返回2；[0,0,0,1]，返回4；
//若全是0，[0,0,0,0],返回1
int GenIR::getNextDim(vector<int> &elementsCnts, int up) {
    for (int i = elementsCnts.size() - 1; i > up; i--) {
        if (elementsCnts[i] != 0) return i + 1;
    }
    return up + 1;
}

//增加元素后，合并所有能合并的数组元素，即对齐
void GenIR::mergeElements(vector<int> &dimensions, vector<ArrayType*> &arrayTys, int up, int dimAdd, vector<Constant*> &elements, vector<int> &elementsCnts) {
    for (int i = dimAdd; i > up; i--) {
        if (elementsCnts[i] % dimensions[i] == 0) {
            vector<Constant*> temp;
            temp.assign(elements.end() - dimensions[i], elements.end());
            elements.erase(elements.end() - dimensions[i], elements.end());
            elements.push_back(new ConstantArray(arrayTys[i], temp));
            elementsCnts[i] = 0;
            elementsCnts[i - 1]++;
        } else break;
    }
}

//最后合并所有元素，不足合并则填0元素，使得elements只剩下一个arrayTys[up]类型的最终数组
void GenIR::finalMerge(vector<int> &dimensions, vector<ArrayType*> &arrayTys, int up, vector<Constant*> &elements, vector<int> &elementsCnts) const {
    for (int i = dimensions.size() - 1; i >= up; i--) {
        while (elementsCnts[i] % dimensions[i] != 0) { //补充当前数组类型所需0元素
            if (i == dimensions.size() - 1) {
                if (curType == INT32_T) {
                    elements.push_back(CONST_INT(0));
                } else {
                    elements.push_back(CONST_FLOAT(0));
                }
            } else {
                elements.push_back(new ConstantZero(arrayTys[i + 1]));
            }
            elementsCnts[i]++;
        }
        if (elementsCnts[i] != 0) {
            vector<Constant*> temp;
            temp.assign(elements.end() - dimensions[i], elements.end());
            elements.erase(elements.end() - dimensions[i], elements.end());
            elements.push_back(new ConstantArray(arrayTys[i], temp));
            elementsCnts[i] = 0;
            if (i != up) elementsCnts[i - 1]++;
        }
    }
}

//生成变量数组的初始化
ConstantArray* GenIR::globalInit(vector<int> &dimensions, vector<ArrayType*> &arrayTys, int up, vector<unique_ptr<InitValAST>> &list) {
    vector<int> elementsCnts(dimensions.size()); //对应各个维度的子数组的元素个数
    vector<Constant*> elements; //各个元素
    int dimAdd;
    for (auto &val : list) {
        if (val->exp != nullptr) {
            dimAdd = dimensions.size() - 1;
            val->exp->accept(*this);
            checkInitType();
            elements.push_back((ConstantInt*)recentVal);
        } else {
            auto nextUp = getNextDim(elementsCnts, up); //该嵌套数组的维度
            dimAdd = nextUp - 1; //比他高一维度的数组需要添加一个元素
            if (nextUp == dimensions.size()) cout << "initial invalid" << endl;//没有连续0，没对齐，不合法
            if (val->initValList.empty()) {
                elements.push_back(new ConstantZero(arrayTys[nextUp]));
            } else {
                auto temp = globalInit(dimensions, arrayTys, nextUp, val->initValList);
                elements.push_back(temp);
            }
        }
        elementsCnts[dimAdd]++;
        mergeElements(dimensions, arrayTys, up, dimAdd, elements, elementsCnts);
    }
    finalMerge(dimensions, arrayTys, up, elements, elementsCnts);
    return static_cast<ConstantArray*>(elements[0]);
}


//根据初始化的量决定嵌套数组的维度
int GenIR::getNextDim(vector<int> &dimensionsCnt, int up, int cnt) {
    for (int i = up; i < dimensionsCnt.size(); i++) {
        if (cnt % dimensionsCnt[i] == 0) return i;
    }
    return 0;
}

//根据首指针递归初始化数组,up表示子数组的最高对齐位置，比如[4][2][4]，子数组最高对齐[2][4],up为1
void GenIR::localInit(Value* ptr, vector<unique_ptr<InitValAST>> &list, vector<int> &dimensionsCnt, int up) {
    int cnt = 0;
    Value* tempPtr = ptr;
    for (auto &initVal : list) {
        if (initVal->exp) {
            if (cnt == 0) cnt++; //第一次赋值时可以少一次create_gep
            else tempPtr = builder->create_gep(ptr, {CONST_INT(cnt++)});
            initVal->exp->accept(*this);
            checkInitType();
            builder->create_store(recentVal, tempPtr);
        } else {
            auto nextUp = getNextDim(dimensionsCnt, up, cnt);
            if (nextUp == 0) cout << "initial invalid!" << endl;
            if (!initVal->initValList.empty()) {
                if (cnt != 0) tempPtr = builder->create_gep(ptr, {CONST_INT(cnt)}); //没赋值过，那tempPtr实际就是ptr
                localInit(tempPtr, initVal->initValList, dimensionsCnt, nextUp);
            }
            cnt += dimensionsCnt[nextUp]; //数组初始化量一定增加这么多
        }
    }
}

void GenIR::visit(InitValAST &ast) {
    //不是数组则求exp的值，若是数组不会进入此函数
    if (ast.exp != nullptr) {
        ast.exp->accept(*this);
    }
}

void GenIR::visit(FuncDefAST &ast) {
    isNewFunc = true;
    params.clear();
    paramNames.clear();
    Type *retType;
    if (ast.funcType == TYPE_INT) retType = INT32_T;
    else if (ast.funcType == TYPE_FLOAT) retType = FLOAT_T;
    else retType = VOID_T;

    //获取参数列表
    for (auto &funcFParam:ast.funcFParamList) {
        funcFParam->accept(*this);
    }
    //获取函数类型
    auto funTy = new FunctionType(retType, params);
    //添加函数
    auto func = new Function(funTy, *ast.id, module.get());
    currentFunction = func;
    scope.push(*ast.id, func); //在进入新的作用域之前添加到符号表中
    //进入函数(进入新的作用域)
    scope.enter();

    std::vector<Value *> args; // 获取函数的形参,通过Function中的iterator
    for (auto arg = func->arguments_.begin(); arg != func->arguments_.end(); arg++)
        args.push_back(*arg);

    auto bb = new BasicBlock(module.get(), "label_entry", func);
    builder->BB_ = bb;
    for (int i = 0; i < (int)(paramNames.size()); i++) {
        auto alloc = builder->create_alloca(params[i]); //分配形参空间
        builder->create_store(args[i], alloc);          // store 形参
        scope.push(paramNames[i], alloc);         //加入作用域
    }
    //创建统一return分支
    retBB = new BasicBlock(module.get(), "label_ret", func);
    if (retType == VOID_T) {
        // void类型无需返回值
        builder->BB_ = retBB;
        builder->create_void_ret();
    } else {
        retAlloca = builder->create_alloca(retType); // 在内存中分配返回值的位置
        builder->BB_ = retBB;
        auto retLoad = builder->create_load(retAlloca);
        builder->create_ret(retLoad);
    }
    //重新回到函数开始
    builder->BB_ = bb;
    has_br = false;
    ast.block->accept(*this);

    //处理没有return的空块
    if (!builder->BB_->get_terminator())
        builder->create_br(retBB);
}

void GenIR::visit(FuncFParamAST &ast) {
    //获取参数类型
    Type *paramType;
    if (ast.bType == TYPE_INT) paramType = INT32_T;
    else paramType = FLOAT_T;
    //是否为数组
    if (ast.isArray) {
        useConst = true; //数组维度是整型常量
        for (int i = ast.arrays.size() - 1; i >= 0; i--) {
            ast.arrays[i]->accept(*this);
            paramType = module->get_array_type(paramType, ((ConstantInt *)recentVal)->value_);
        }
        useConst = false;
        //如int a[][2]，则参数为[2 x i32]* ;  int a[],参数为i32 *
        paramType = module->get_pointer_type(paramType);
    }
    params.push_back(paramType);
    paramNames.push_back(*ast.id);
}

void GenIR::visit(BlockAST &ast) {
    //如果是一个新的函数，则不用再进入一个新的作用域
    if (isNewFunc)
        isNewFunc = false;
    //其它情况，需要进入一个新的作用域
    else {
        scope.enter();
    }
    //遍历每一个语句块
    for (auto &item : ast.blockItemList) {
        if (has_br) break;     //此BB已经出现了br，后续指令无效
        item->accept(*this);
    }

    scope.exit();
}

void GenIR::visit(BlockItemAST &ast) {
    if (ast.decl != nullptr) {
        ast.decl->accept(*this);
    } else {
        ast.stmt->accept(*this);
    }
}

//****************************************************************************
//* 你需要完成这个visit()方法
//****************************************************************************
void GenIR::visit(StmtAST &ast) {
    switch (ast.sType) {
        case SEMI:
            break;
        case ASS: {
            // 表明当前是赋值语句的左值处理
            requireLVal = true;
            ast.lVal->accept(*this);  // 访问左值，更新 recentVal 为左值的地址
            auto var = recentVal;     // 获取左值对应的变量（地址）

            is_single_exp = true;
            ast.exp->accept(*this);   // 访问右值，更新 recentVal 为右值的值
            auto expval = recentVal;  // 获取右值的值

            // 类型转换，确保左右值类型一致
            if (var->type_->tid_ == Type::FloatTyID && expval->type_->tid_ == Type::IntegerTyID) {
                // 如果左值是 float 类型，右值是 int 类型，进行 int 转 float
                expval = builder->create_sitofp(expval, FLOAT_T);
            }
            else if (var->type_->tid_ == Type::IntegerTyID && expval->type_->tid_ == Type::FloatTyID) {
                // 如果左值是 int 类型，右值是 float 类型，进行 float 转 int
                expval = builder->create_fptosi(expval, INT32_T);
            }

            // 使用 create_store 将右值存储到左值变量中
            builder->create_store(expval, var);

            break;
        }
        case EXP:
            is_single_exp = true;
            ast.exp->accept(*this);  // 处理表达式
            is_single_exp = false;
            break;
        case CONT:
            builder->create_br(whileCondBB);  // continue 语句跳转回循环条件
            has_br = true;
            break;
        case BRE:
            builder->create_br(whileFalseBB);  // break 语句跳转到循环结束部分
            has_br = true;
            break;
        case RET:
            ast.returnStmt->accept(*this);  // 返回语句
            break;
        case BLK:
            ast.block->accept(*this);  // 处理代码块
            break;
        case SEL:
            ast.selectStmt->accept(*this);  // 条件语句
            break;
        case ITER:
            ast.iterationStmt->accept(*this);  // 循环语句
            break;
    }
}

void GenIR::visit(ReturnStmtAST &ast) {
    if (ast.exp == nullptr) {
        recentVal = builder->create_br(retBB);
    } else {
        //先把返回值store在retAlloca中，再跳转到统一的返回入口
        ast.exp->accept(*this);
        //类型转换
        if (recentVal->type_ == FLOAT_T && currentFunction->get_return_type() == INT32_T) {
            auto temp = builder->create_fptosi(recentVal, INT32_T);
            builder->create_store(temp, retAlloca);
        } else if (recentVal->type_ == INT32_T && currentFunction->get_return_type() == FLOAT_T) {
            auto temp = builder->create_sitofp(recentVal, FLOAT_T);
            builder->create_store(temp, retAlloca);
        } else
            builder->create_store(recentVal, retAlloca);
        recentVal = builder->create_br(retBB);
    }
    has_br = true;
    //builder->BB_ = new BasicBlock(module.get(), to_string(id++), currentFunction); //return语句后必是新块
}

void GenIR::visit(SelectStmtAST &ast) {
    //先保存trueBB和falseBB，防止嵌套导致返回上一层后丢失块的地址
    auto tempTrue = trueBB;
    auto tempFalse = falseBB;

    trueBB = new BasicBlock(module.get(), to_string(id++), currentFunction);
    falseBB = new BasicBlock(module.get(), to_string(id++), currentFunction);
    BasicBlock* nextIf; // if语句后的基本块
    if (ast.elseStmt == nullptr) nextIf = falseBB;
    else nextIf = new BasicBlock(module.get(), to_string(id++), currentFunction);
    ast.cond->accept(*this);
    //检查是否是i1，不是则进行比较
    if (recentVal->type_ == INT32_T) {
        recentVal = builder->create_icmp_ne(recentVal, CONST_INT(0));
    } else if (recentVal->type_ == FLOAT_T) {
        recentVal = builder->create_fcmp_ne(recentVal, CONST_FLOAT(0));
    }
    builder->create_cond_br(recentVal, trueBB, falseBB);

    builder->BB_ = trueBB; //开始构建trueBB
    has_br = false;
    ast.ifStmt->accept(*this);
    if (!builder->BB_->get_terminator()) {
        builder->create_br(nextIf);
    }

    if (ast.elseStmt != nullptr) { // 开始构建falseBB
        builder->BB_ = falseBB;
        has_br = false;
        ast.elseStmt->accept(*this);
        if (!builder->BB_->get_terminator()) {
            builder->create_br(nextIf);
        }
    }

    builder->BB_ = nextIf;
    has_br = false;
    // 还原trueBB和falseBB
    trueBB = tempTrue;
    falseBB = tempFalse;
}

void GenIR::visit(IterationStmtAST &ast) {
    //先保存trueBB和falseBB，防止嵌套导致返回上一层后丢失块的地址
    auto tempTrue = trueBB;
    auto tempFalse = falseBB; //即while的next block
    auto tempCond = whileCondBB;
    auto tempWhileFalseBB = whileFalseBB; //break只跳while的false，而不跳全局false

    whileCondBB = new BasicBlock(module.get(), to_string(id++), currentFunction);
    trueBB = new BasicBlock(module.get(), to_string(id++), currentFunction);
    falseBB = new BasicBlock(module.get(), to_string(id++), currentFunction);
    whileFalseBB = falseBB;

    builder->create_br(whileCondBB);
    builder->BB_ = whileCondBB; //条件也是一个基本块
    has_br = false;
    ast.cond->accept(*this);
    if (recentVal->type_ == INT32_T) {
        recentVal = builder->create_icmp_ne(recentVal, CONST_INT(0));
    } else if (recentVal->type_ == FLOAT_T) {
        recentVal = builder->create_fcmp_ne(recentVal, CONST_FLOAT(0.0));
    }
    builder->create_cond_br(recentVal, trueBB, falseBB);

    builder->BB_ = trueBB;
    has_br = false;
    ast.stmt->accept(*this);
    //while语句体一定是跳回cond
    if (!builder->BB_->get_terminator()) {
        builder->create_br(whileCondBB);
    }

    builder->BB_ = falseBB;
    has_br = false;

    //还原trueBB，falseBB，tempCond
    trueBB = tempTrue;
    falseBB = tempFalse;
    whileCondBB = tempCond;
    whileFalseBB = tempWhileFalseBB;
}

//根据待计算的两个Constant的类型，求出对应的值赋值到intVal，floatVal中，返回计算结果是否为int
bool GenIR::checkCalType(Value* val[],int intVal[], float floatVal[]) {
    bool resultIsInt = false;
    if (dynamic_cast<ConstantInt*>(val[0]) && dynamic_cast<ConstantInt*>(val[1])) {
        resultIsInt = true;
        intVal[0] = dynamic_cast<ConstantInt*>(val[0])->value_;
        intVal[1] = dynamic_cast<ConstantInt*>(val[1])->value_;
    } else { //操作结果一定是float
        if (dynamic_cast<ConstantInt*>(val[0])) floatVal[0] = dynamic_cast<ConstantInt*>(val[0])->value_;
        else floatVal[0] = dynamic_cast<ConstantFloat*>(val[0])->value_;
        if (dynamic_cast<ConstantInt*>(val[1])) floatVal[1] = dynamic_cast<ConstantInt*>(val[1])->value_;
        else floatVal[1] = dynamic_cast<ConstantFloat*>(val[1])->value_;
    }
    return resultIsInt;
}

//根据待计算的两个寄存器数的类型，若需要转换类型输出转换指令
void GenIR::checkCalType(Value* val[]) {
    if (val[0]->type_ == INT1_T) {
        val[0] = builder->create_zext(val[0], INT32_T);
    }
    if (val[1]->type_ == INT1_T) {
        val[1] = builder->create_zext(val[1], INT32_T);
    }
    if (val[0]->type_ == INT32_T && val[1]->type_ == FLOAT_T) {
        val[0] = builder->create_sitofp(val[0], FLOAT_T);
    }
    if (val[1]->type_ == INT32_T && val[0]->type_ == FLOAT_T) {
        val[1] = builder->create_sitofp(val[1], FLOAT_T);
    }
}

void GenIR::visit(AddExpAST &ast) {
    if (ast.addExp == nullptr) {
        ast.mulExp->accept(*this);
        return;
    }

    Value* val[2]; //lVal, rVal
    ast.addExp->accept(*this);
    val[0] = recentVal;
    ast.mulExp->accept(*this);
    val[1] = recentVal;

    //若都是常量
    if (useConst) {
        int intVal[3]; //lInt, rInt, relInt;
        float floatVal[3]; // lFloat, rFloat, relFloat;
        bool resultIsInt = checkCalType(val, intVal, floatVal);
        switch (ast.op) {
            case AOP_ADD:
                intVal[2] = intVal[0] + intVal[1];
                floatVal[2] = floatVal[0] + floatVal[1];
                break;
            case AOP_MINUS:
                intVal[2] = intVal[0] - intVal[1];
                floatVal[2] = floatVal[0] - floatVal[1];
                break;
        }
        if (resultIsInt) recentVal = CONST_INT(intVal[2]);
        else recentVal = CONST_FLOAT(floatVal[2]);
        return;
    }

    //若不是常量，进行计算，输出指令
    checkCalType(val);
    if (val[0]->type_ == INT32_T) {
        switch (ast.op) {
            case AOP_ADD:
                recentVal = builder->create_iadd(val[0], val[1]);
                break;
            case AOP_MINUS:
                recentVal = builder->create_isub(val[0], val[1]);
                break;
        }
    } else {
        switch (ast.op) {
            case AOP_ADD:
                recentVal = builder->create_fadd(val[0], val[1]);
                break;
            case AOP_MINUS:
                recentVal = builder->create_fsub(val[0], val[1]);
                break;
        }
    }
}

void GenIR::visit(MulExpAST &ast) {
    if (ast.mulExp == nullptr) {
        ast.unaryExp->accept(*this);
        return;
    }

    Value* val[2]; //lVal, rVal
    ast.mulExp->accept(*this);
    val[0] = recentVal;
    ast.unaryExp->accept(*this);
    val[1] = recentVal;

    //若都是常量
    if (useConst) {
        int intVal[3]; //lInt, rInt, relInt;
        float floatVal[3]; // lFloat, rFloat, relFloat;
        bool resultIsInt = checkCalType(val, intVal, floatVal);
        switch (ast.op) {
            case MOP_MUL:
                intVal[2] = intVal[0] * intVal[1];
                floatVal[2] = floatVal[0] * floatVal[1];
                break;
            case MOP_DIV:
                intVal[2] = intVal[0] / intVal[1];
                floatVal[2] = floatVal[0] / floatVal[1];
                break;
            case MOP_MOD:
                intVal[2] = intVal[0] % intVal[1];
                break;
        }
        if (resultIsInt) recentVal = CONST_INT(intVal[2]);
        else recentVal = CONST_FLOAT(floatVal[2]);
        return;
    }

    //若不是常量，进行计算，输出指令
    checkCalType(val);
    if (val[0]->type_ == INT32_T) {
        switch (ast.op) {
            case MOP_MUL:
                recentVal = builder->create_imul(val[0], val[1]);
                break;
            case MOP_DIV:
                recentVal = builder->create_isdiv(val[0], val[1]);
                break;
            case MOP_MOD:
                recentVal = builder->create_isrem(val[0], val[1]);
                break;
        }
    }
    else {
        switch (ast.op) {
            case MOP_MUL:
                recentVal = builder->create_fmul(val[0], val[1]);
                break;
            case MOP_DIV:
                recentVal = builder->create_fdiv(val[0], val[1]);
                break;
            case MOP_MOD://never occur
                break;
        }
    }
}

void GenIR::visit(UnaryExpAST &ast) {
    // 为常量算式
    if (useConst) {
        if (ast.primaryExp) {
            ast.primaryExp->accept(*this);
        } else if (ast.unaryExp) {
            ast.unaryExp->accept(*this);
            if (ast.op == UOP_MINUS) {
                //是整型常量
                if (dynamic_cast<ConstantInt*>(recentVal)) {
                    auto temp = (ConstantInt*)recentVal;
                    temp->value_ = -temp->value_;
                    recentVal = temp;
                } else {
                    auto temp = (ConstantFloat*)recentVal;
                    temp->value_ = -temp->value_;
                    recentVal = temp;
                }
            }
        } else {
            cout << "Function call in ConstExp!" << endl;
        }
        return;
    }


    //不是常量算式
    if (ast.primaryExp != nullptr) {
        ast.primaryExp->accept(*this);
    } else if (ast.call != nullptr) {
        ast.call->accept(*this);
    } else {
        ast.unaryExp->accept(*this);
        if (recentVal->type_ == VOID_T)
            return;
        else if (recentVal->type_ == INT1_T) // INT1-->INT32
            recentVal = builder->create_zext(recentVal, INT32_T);

        if (recentVal->type_ == INT32_T) {
            switch (ast.op) {
                case UOP_MINUS:
                    recentVal = builder->create_isub(CONST_INT(0), recentVal);
                    break;
                case UOP_NOT:
                    recentVal = builder->create_icmp_eq(recentVal, CONST_INT(0));
                    break;
                case UOP_ADD:
                    break;
            }
        } else {
            switch (ast.op) {
                case UOP_MINUS:
                    recentVal = builder->create_fsub(CONST_FLOAT(0), recentVal);
                    break;
                case UOP_NOT:
                    recentVal = builder->create_fcmp_eq(recentVal, CONST_FLOAT(0));
                    break;
                case UOP_ADD:
                    break;
            }
        }
    }
}

void GenIR::visit(PrimaryExpAST &ast) {
    if (ast.exp) {
        ast.exp->accept(*this);
    } else if (ast.lval) {
        ast.lval->accept(*this);
    } else if (ast.number) {
        ast.number->accept(*this);
    }
}

void GenIR::visit(LValAST &ast) {
    bool isTrueLVal = requireLVal; //是否真是作为左值
    requireLVal = false;
    auto var = scope.find(*ast.id);
    //全局作用域内，一定使用常量，全局作用域下访问到LValAST，那么use_const一定被置为了true
    if (scope.in_global()) {
        //不是数组，直接返回该常量
        if (ast.arrays.empty()) {
            recentVal = var;
            return;
        }
        //若是数组，则var一定是全局常量数组
        vector<int> index;
        for (auto &exp : ast.arrays) {
            exp->accept(*this);
            index.push_back(dynamic_cast<ConstantInt*>(recentVal)->value_);
        }
        recentVal = ((GlobalVariable *)var)->init_val_; //使用var的初始化数组查找常量元素
        for (int i : index) {
            //某数组元素为ConstantZero，则该数一定是0
            if (dynamic_cast<ConstantZero*>(recentVal)) {
                Type* arrayTy = recentVal->type_;
                //找数组元素标签
                while (dynamic_cast<ArrayType*>(arrayTy)) {
                    arrayTy = dynamic_cast<ArrayType*>(arrayTy)->contained_;
                }
                if (arrayTy == INT32_T) recentVal = CONST_INT(0);
                else recentVal = CONST_FLOAT(0);
                return;
            }
            if (dynamic_cast<ConstantArray*>(recentVal)) {
                recentVal = ((ConstantArray*)recentVal)->const_array[i];
            }
        }
        return;
    }

    //局部作用域
    if (var->type_->tid_ == Type::IntegerTyID || var->type_->tid_ == Type::FloatTyID) { //说明为局部常量
        recentVal = var;
        return;
    }
    // 不是常量那么var一定是指针类型
    Type* varType = static_cast<PointerType*>(var->type_)->contained_; //所指的类型
    if (!ast.arrays.empty()) { //说明是数组
        vector<Value *> idxs;
        for (auto &exp : ast.arrays) {
            exp->accept(*this);
            idxs.push_back(recentVal);
        }
        // 当函数传入参数i32 *，会生成类型为i32 **的局部变量，即此情况
        if (varType->tid_ == Type::PointerTyID) {
            var = builder->create_load(var);
        } else if (varType->tid_ == Type::ArrayTyID) {
            idxs.insert(idxs.begin(), CONST_INT(0));
        }
        var = builder->create_gep(var, idxs); //获取的一定是指针类型
        varType = ((PointerType*)var->type_)->contained_;
    }

    //指向的还是数组,那么一定是传数组参,数组若为x[2], 参数为int a[]，需要传i32 *
    if (varType->tid_ == Type::ArrayTyID) {
        recentVal = builder->create_gep(var, {CONST_INT(0), CONST_INT(0)});
    } else if (!isTrueLVal) { //如果不是取左值，那么load
        recentVal = builder->create_load(var);
    } else { //否则返回地址值
        recentVal = var;
    }
}

void GenIR::visit(NumberAST &ast) {
    if (ast.isInt) recentVal = CONST_INT(ast.intval);
    else recentVal = CONST_FLOAT(ast.floatval);
}

void GenIR::visit(CallAST &ast) {
    auto fun = (Function *)scope.find(*ast.id);
    //引用函数返回值
    if (fun->basic_blocks_.size() && !is_single_exp)
        fun->use_ret_cnt ++ ;
    is_single_exp = false;
    vector<Value *> args;
    for (int i = 0; i < ast.funcCParamList.size(); i++) {
        ast.funcCParamList[i]->accept(*this);
        //检查函数形参与实参类型是否匹配
        if (recentVal->type_ == INT32_T && fun->arguments_[i]->type_ == FLOAT_T) {
            recentVal = builder->create_sitofp(recentVal, FLOAT_T);
        } else if (recentVal->type_ == FLOAT_T && fun->arguments_[i]->type_ == INT32_T) {
            recentVal = builder->create_fptosi(recentVal, INT32_T);
        }
        args.push_back(recentVal);
    }
    recentVal = builder->create_call(fun, args);
}

void GenIR::visit(RelExpAST &ast) {
    if (ast.relExp == nullptr) {
        ast.addExp->accept(*this);
        return;
    }
    Value* val[2];
    ast.relExp->accept(*this);
    val[0] = recentVal;
    ast.addExp->accept(*this);
    val[1] = recentVal;
    checkCalType(val);
    if (val[0]->type_ == INT32_T) {
        switch (ast.op) {
            case ROP_LTE:
                recentVal = builder->create_icmp_le(val[0], val[1]);
                break;
            case ROP_LT:
                recentVal = builder->create_icmp_lt(val[0], val[1]);
                break;
            case ROP_GT:
                recentVal = builder->create_icmp_gt(val[0], val[1]);
                break;
            case ROP_GTE:
                recentVal = builder->create_icmp_ge(val[0], val[1]);
                break;
        }
    } else {
        switch (ast.op) {
            case ROP_LTE:
                recentVal = builder->create_fcmp_le(val[0], val[1]);
                break;
            case ROP_LT:
                recentVal = builder->create_fcmp_lt(val[0], val[1]);
                break;
            case ROP_GT:
                recentVal = builder->create_fcmp_gt(val[0], val[1]);
                break;
            case ROP_GTE:
                recentVal = builder->create_fcmp_ge(val[0], val[1]);
                break;
        }
    }
}

void GenIR::visit(EqExpAST &ast) {
    if (ast.eqExp == nullptr) {
        ast.relExp->accept(*this);
        return;
    }
    Value* val[2];
    ast.eqExp->accept(*this);
    val[0] = recentVal;
    ast.relExp->accept(*this);
    val[1] = recentVal;
    checkCalType(val);
    if (val[0]->type_ == INT32_T) {
        switch (ast.op) {
            case EOP_EQ:
                recentVal = builder->create_icmp_eq(val[0], val[1]);
                break;
            case EOP_NEQ:
                recentVal = builder->create_icmp_ne(val[0], val[1]);
                break;
        }
    } else {
        switch (ast.op) {
            case EOP_EQ:
                recentVal = builder->create_fcmp_eq(val[0], val[1]);
                break;
            case EOP_NEQ:
                recentVal = builder->create_fcmp_ne(val[0], val[1]);
                break;
        }
    }
}

void GenIR::visit(LAndExpAST &ast) {
    if (ast.lAndExp == nullptr) {
        ast.eqExp->accept(*this);
        return;
    }
    auto tempTrue = trueBB; //防止嵌套and导致原trueBB丢失。用于生成短路模块
    trueBB = new BasicBlock(module.get(), to_string(id++), currentFunction);
    ast.lAndExp->accept(*this);

    if (recentVal->type_ == INT32_T) {
        recentVal = builder->create_icmp_ne(recentVal, CONST_INT(0));
    } else if (recentVal->type_ == FLOAT_T) {
        recentVal = builder->create_fcmp_ne(recentVal, CONST_FLOAT(0));
    }
    builder->create_cond_br(recentVal, trueBB, falseBB);
    builder->BB_ = trueBB;
    has_br = false;
    trueBB = tempTrue; //还原原来的true模块

    ast.eqExp->accept(*this);
}

void GenIR::visit(LOrExpAST &ast) {
    if (ast.lOrExp == nullptr) {
        ast.lAndExp->accept(*this);
        return;
    }
    auto tempFalse = falseBB; //防止嵌套and导致原trueBB丢失。用于生成短路模块
    falseBB = new BasicBlock(module.get(), to_string(id++), currentFunction);
    ast.lOrExp->accept(*this);
    if (recentVal->type_ == INT32_T) {
        recentVal = builder->create_icmp_ne(recentVal, CONST_INT(0));
    } else if (recentVal->type_ == FLOAT_T) {
        recentVal = builder->create_fcmp_ne(recentVal, CONST_FLOAT(0));
    }
    builder->create_cond_br(recentVal, trueBB, falseBB);
    builder->BB_ = falseBB;
    has_br = false;
    falseBB = tempFalse;

    ast.lAndExp->accept(*this);
}
