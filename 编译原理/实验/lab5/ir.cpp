#include "ir.h"
std::map<Instruction::OpID, std::string> instr_id2string_ = {
        {Instruction::Ret,"ret"},{Instruction::Br,"br"},{Instruction::FNeg,"fneg"},
        {Instruction::Add,"add"},{Instruction::Sub,"sub"},{Instruction::Mul,"mul"},{Instruction::SDiv,"sdiv"},{Instruction::SRem,"srem"},{Instruction::UDiv,"udiv"},{Instruction::URem,"urem"},
        {Instruction::FAdd,"fadd"},{Instruction::FSub,"fsub"},{Instruction::FMul,"fmul"},{Instruction::FDiv,"fdiv"},
        {Instruction::Shl,"shl"},{Instruction::LShr,"lshr"},{Instruction::AShr,"ashr"},{Instruction::And,"and"},{Instruction::Or,"or"},{Instruction::Xor,"xor"},
        {Instruction::Alloca,"alloca"},{Instruction::Load,"load"},{Instruction::Store,"store"},{Instruction::GetElementPtr,"getelementptr"},
        {Instruction::ZExt,"zext"},{Instruction::FPtoSI,"fptosi"},{Instruction::SItoFP,"sitofp"}, {Instruction::BitCast, "bitcast"},
        {Instruction::ICmp,"icmp"},{Instruction::FCmp,"fcmp"},{Instruction::PHI,"phi"},{Instruction::Call,"call"}
};   // Instruction from opid to string
const std::map<ICmpInst::ICmpOp, std::string> ICmpInst::ICmpOpName = {{ICmpInst::ICmpOp::ICMP_EQ, "eq"}, {ICmpInst::ICmpOp::ICMP_NE, "ne"}, {ICmpInst::ICmpOp::ICMP_UGT, "hi"}, {ICmpInst::ICmpOp::ICMP_UGE, "cs"}, {ICmpInst::ICmpOp::ICMP_ULT, "cc"}, {ICmpInst::ICmpOp::ICMP_ULE, "ls"}, {ICmpInst::ICmpOp::ICMP_SGT, "gt"}, {ICmpInst::ICmpOp::ICMP_SGE, "ge"}, {ICmpInst::ICmpOp::ICMP_SLT, "lt"}, {ICmpInst::ICmpOp::ICMP_SLE, "le"}};
const std::map<FCmpInst::FCmpOp, std::string> FCmpInst::FCmpOpName = {{FCmpInst::FCmpOp::FCMP_FALSE, "nv"}, {FCmpInst::FCmpOp::FCMP_OEQ, "eq"}, {FCmpInst::FCmpOp::FCMP_OGT, "gt"}, {FCmpInst::FCmpOp::FCMP_OGE, "ge"}, {FCmpInst::FCmpOp::FCMP_OLT, "cc"}, {FCmpInst::FCmpOp::FCMP_OLE, "ls"}, {FCmpInst::FCmpOp::FCMP_ONE, "ne"}, {FCmpInst::FCmpOp::FCMP_ORD, "vc"}, {FCmpInst::FCmpOp::FCMP_UNO, "vs"}, {FCmpInst::FCmpOp::FCMP_UEQ, "eq"}, {FCmpInst::FCmpOp::FCMP_UGT, "hi"}, {FCmpInst::FCmpOp::FCMP_UGE, "cs"}, {FCmpInst::FCmpOp::FCMP_ULT, "lt"}, {FCmpInst::FCmpOp::FCMP_ULE, "le"}, {FCmpInst::FCmpOp::FCMP_UNE, "ne"}, {FCmpInst::FCmpOp::FCMP_TRUE, "al"}};
std::string print_as_op( Value *v, bool print_ty );
std::string print_cmp_type( ICmpInst::ICmpOp op);
std::string print_fcmp_type( FCmpInst::FCmpOp op);
//-----------------------------------------------Type-----------------------------------------------
std::string Type::print(){
    std::string type_ir;
    switch (this->tid_)
    {
        case VoidTyID:
            type_ir += "void";
            break;
        case LabelTyID:
            type_ir += "label";
            break;
        case IntegerTyID:
            type_ir += "i";
            type_ir += std::to_string( static_cast<IntegerType *>(this)->num_bits_);
            break;
        case FloatTyID:
            type_ir += "float";
            break;
        case FunctionTyID:
            type_ir += static_cast<FunctionType *>(this)->result_->print();
            type_ir += " (";
            for (size_t i = 0 ; i < static_cast<FunctionType *>(this)->args_.size() ;i ++ ) {
                if (i) type_ir += ", ";
                type_ir += static_cast<FunctionType *>(this)->args_[i]->print();
            }
            type_ir += ")";
            break;
        case PointerTyID:
            type_ir += static_cast<PointerType *>(this)->contained_->print();
            type_ir += "*";
            break;
        case ArrayTyID:
            type_ir += "[";
            type_ir += std::to_string( static_cast<ArrayType *>(this)->num_elements_);
            type_ir += " x ";
            type_ir += static_cast<ArrayType *>(this)->contained_->print();
            type_ir += "]";
            break;
        default:
            break;
    }
    return type_ir;
}

//-----------------------------------------------Value-----------------------------------------------
void Value::replace_all_use_with(Value *new_val) {
    for (auto use : use_list_) {
        auto val = dynamic_cast<Instruction *>(use.val_);
    #ifdef DEBUG
        assert(val && "new_val is not a user");
    #endif
        val->set_operand(use.arg_no_, new_val);
    }
}
bool Value::remove_used(Instruction* user,unsigned int i)
{
    if(this!=user->operands_[i]) {
        return false;
    }
    auto pos = user->use_pos_[i];
    use_list_.erase( pos);
    user->operands_[i] = nullptr; //表示user->use_pos_[i]失效了，提示set_operand不要再删除
    return true;
}


//-----------------------------------------------Constant-----------------------------------------------
std::string ConstantInt::print()
{
    std::string const_ir;
    if (this->type_->tid_ == Type::IntegerTyID && static_cast<IntegerType *>(this->type_)->num_bits_ == 1 ) {
        //int1
        const_ir += (this->value_ == 0) ? "0" : "1";
    }
    else //int32
        const_ir += std::to_string(this->value_);
    return const_ir;
}

std::string ConstantFloat::print()
{
    std::stringstream fp_ir_ss;
    std::string fp_ir;
    double val = this->value_;
    fp_ir_ss << "0x"<< std::hex << *(uint64_t *)&val << std::endl;
    fp_ir_ss >> fp_ir;
    return fp_ir;
}

std::string ConstantArray::print()
{
    std::string const_ir;
    const_ir += "[";
    const_ir += static_cast<ArrayType *>(this->type_)->contained_->print();
    const_ir += " ";
    const_ir += const_array[0]->print();
    for (size_t i = 1 ; i < this->const_array.size() ; i++ ){
        const_ir += ", ";
        const_ir += static_cast<ArrayType *>(this->type_)->contained_->print();
        const_ir += " ";
        const_ir += const_array[i]->print();
    }
    const_ir += "]";
    return const_ir;
}

std::string ConstantZero::print()
{
    return "zeroinitializer";
}

//-----------------------------------------------Module-----------------------------------------------
std::string Module::print()
{
    std::string module_ir;
    for ( auto global_val : this->global_list_) {
        module_ir += global_val->print();
        module_ir += "\n";
    }
    for ( auto func : this->function_list_) {
        module_ir += func->print();
        module_ir += "\n";
    }
    return module_ir;
}

Function *Module::getMainFunc() {
    for (auto f : function_list_) {
        if (f->name_ == "main") {
            return f;
        }
    }
    return nullptr;
}

//-----------------------------------------------GlobalVariable-----------------------------------------------
std::string GlobalVariable::print()
{
    std::string global_val_ir;
    global_val_ir += print_as_op(this, false);
    global_val_ir += " = ";
    global_val_ir += (this->is_const_ ? "constant " : "global ");
    global_val_ir += static_cast<PointerType *>(this->type_)->contained_->print();
    global_val_ir += " ";
    global_val_ir += this->init_val_->print();
    return global_val_ir;
}

//-----------------------------------------------Function-----------------------------------------------
std::string Function::print()
{
    if (this->name_ == "llvm.memset.p0.i32") {
        std::string func_ir = "declare void @llvm.memset.p0.i32(i32*, i8, i32, i1)";
        return func_ir;
    }
    set_instr_name();
    std::string func_ir;
    if ( this->is_declaration() )
        func_ir += "declare ";
    else
        func_ir += "define ";

    func_ir += this->get_return_type()->print();
    func_ir += " ";
    func_ir += print_as_op(this, false);
    func_ir += "(";

    //print arg
    if (this->is_declaration()) {
        for (size_t i = 0 ; i < this->arguments_.size() ; i ++ ) {
            if(i)
                func_ir += ", ";
            func_ir += static_cast<FunctionType *>(this->type_)->args_[i]->print();
        }
    }
    else {
        for (auto arg = this->arguments_.begin(); arg != arguments_.end() ; arg++ ) {
            if( arg != this->arguments_.begin()) {
                func_ir += ", ";
            }
            func_ir += static_cast<Argument *>(*arg)->print();
        }
    }
    func_ir += ")";

    //print bb
    if(!this->is_declaration()) {
        func_ir += " {";
        func_ir += "\n";
        for ( auto bb : this->basic_blocks_) {
            func_ir += bb->print();
        }
        func_ir += "}";
    }

    return func_ir;
}

std::string Argument::print()
{
    std::string arg_ir;
    arg_ir += this->type_->print();
    arg_ir += " %";
    arg_ir += this->name_;
    return arg_ir;
}

void Function::remove_bb(BasicBlock* bb)
{ 
    //basic_blocks_.remove(bb); 
    basic_blocks_.erase(std::remove(basic_blocks_.begin(), basic_blocks_.end(), bb) , basic_blocks_.end());
    for (auto pre : bb->pre_bbs_) {
        pre->remove_succ_basic_block(bb);
    }
    for (auto succ : bb->succ_bbs_) {
        succ->remove_pre_basic_block(bb);
    }
}

BasicBlock *Function::getRetBB() {
    for (auto bb : basic_blocks_) {
        if (bb->get_terminator()->is_ret()) {
            return bb;
        }
    }
    return nullptr;
}

//-----------------------------------------------BasicBlock-----------------------------------------------
std::string BasicBlock::print()
{
    std::string bb_ir;
    bb_ir += this->name_;
    bb_ir += ":";
    // print prebb
    if(!this->pre_bbs_.empty()) {
        bb_ir += "                                                ; preds = ";
    }
    for (auto bb : this->pre_bbs_) {
        if( bb != *this->pre_bbs_.begin() )
            bb_ir += ", ";
        bb_ir += print_as_op(bb, false);
    }

    // print prebb
    if ( !this->parent_) {
        bb_ir += "\n";
        bb_ir += "; Error: Block without parent!";
    }
    bb_ir += "\n";
    for (auto instr : this->instr_list_) {
        bb_ir += "  ";
        bb_ir += instr->print();
        bb_ir += "\n";
    }

    return bb_ir;
}

Instruction *BasicBlock::get_terminator() {
    if (instr_list_.empty())
        return nullptr;
    switch (instr_list_.back()->op_id_) {
        case Instruction::Ret:
        case Instruction::Br:
            return instr_list_.back();
        default:
            return nullptr;
    }
}

bool BasicBlock::delete_instr(Instruction *instr) {
    //******************--------instvec2list-----qwc20220814
    //instr_list_.remove(instr);
    if((!instr)||instr->pos_in_bb.size()!=1||instr->parent_!= this)
        return false;
    this->instr_list_.erase(instr->pos_in_bb.back());
    //instr_list_.erase(std::remove(instr_list_.begin(), instr_list_.end(), instr) , instr_list_.end());
    instr->remove_use_of_ops();
    instr->pos_in_bb.clear();//保证指令自由身
    instr->parent_ = nullptr;
    return true;
}

bool BasicBlock::add_instruction(Instruction *instr) {
//******************--------instvec2list-----qwc20220814

    if(instr->pos_in_bb.size()!=0){ //指令已经插入到某个地方了
        return false;
    }else{
        instr_list_.push_back(instr);
        std::list<Instruction*>::iterator tail = instr_list_.end();
        instr->pos_in_bb.emplace_back(--tail);
        instr->parent_ = this;
        return true;
    }

}

bool BasicBlock::add_instruction_front(Instruction *instr) {
//******************--------instvec2list-----qwc20220814
    if(instr->pos_in_bb.size()!=0){ //指令已经插入到某个地方了
        return false;
    }else{
        instr_list_.push_front(instr);
        std::list<Instruction*>::iterator head = instr_list_.begin();
        instr->pos_in_bb.emplace_back(head);
        instr->parent_ = this;
        return true;
    }
}



//插入到倒数第二位
bool BasicBlock::add_instruction_before_terminator(Instruction *instr) {
    if(instr->pos_in_bb.size()!=0){ //指令已经插入到某个地方了
        return false;
    }else if(instr_list_.empty()){ //没有“倒数第1位”何来的倒数第二位
        return false;
    }else{
        auto it = std::end(instr_list_);//最后一位的后一位位置
        instr_list_.emplace(--it, instr);//插入使得代替最后一位（--it）的位置，此时it是最后一位，插入的是倒数第二位
        instr->pos_in_bb.emplace_back(--it);//记录插入的结果
        instr->parent_ = this;
        return true;
    }
}

bool BasicBlock::add_instruction_before_inst(Instruction* new_instr,Instruction* instr) {
    if((!instr)||instr->pos_in_bb.size()!=1||instr->parent_!= this)
        return false;
    if(new_instr->pos_in_bb.size()!=0) //指令已经插入到某个地方了
        return false;
    else if(instr_list_.empty()) //bb原本没有指令，那instr不在bb内，那为啥instr->parent_== this
        return false;
    else{
        auto it = instr->pos_in_bb.back();//
        instr_list_.emplace( it, new_instr);//插入使得代替最后一位（--it）的位置，此时it是最后一位，插入的是倒数第二位
        new_instr->pos_in_bb.emplace_back(--it);//记录插入的结果
        new_instr->parent_ = this;
        return true;
    }
}

//从bb移出一个指令，但是不删指令的use关系，因为还要插入其他bb
bool BasicBlock::remove_instr(Instruction *instr) {
    //instr_list_.remove(instr);
    if((!instr)||instr->pos_in_bb.size()!=1||instr->parent_!= this)
        return false;
    this->instr_list_.erase(instr->pos_in_bb.back());
//    instr->remove_use_of_ops();
    instr->pos_in_bb.clear(); //保证指令自由身
    instr->parent_ = nullptr;
    return true;
}


//-----------------------------------------------Instruction-----------------------------------------------
std::string BinaryInst::print()
{
    std::string instr_ir;
    instr_ir += "%";
    instr_ir += this->name_;
    instr_ir += " = ";
    instr_ir += instr_id2string_[this->op_id_];
    instr_ir += " ";
    instr_ir += this->operands_[0]->type_->print();
    instr_ir += " ";
    instr_ir += print_as_op(this->get_operand(0), false);
    instr_ir += ", ";
    assert(this->get_operand(0)->type_->tid_ == this->get_operand(1)->type_->tid_);
    instr_ir += print_as_op(this->get_operand(1), false);
    //instr_ir += print_as_op(this->get_operand(1), true);
    return instr_ir;
}

std::string UnaryInst::print()
{
    std::string instr_ir;
    instr_ir += "%";
    instr_ir += this->name_;
    instr_ir += " = ";
    instr_ir += instr_id2string_[this->op_id_];
    instr_ir += " ";
    instr_ir += this->operands_[0]->type_->print();
    instr_ir += " ";
    instr_ir += print_as_op(this->get_operand(0), false);
    switch (this->op_id_) {
        case Instruction::ZExt:
            assert(this->type_->tid_ == Type::IntegerTyID);
            instr_ir += " to i32";
            break;
        case Instruction::FPtoSI:
            assert(this->type_->tid_ == Type::IntegerTyID);
            instr_ir += " to i32";
            break;
        case Instruction::SItoFP:
            assert(this->type_->tid_ == Type::FloatTyID);
            instr_ir += " to float";
            break;
        default:
            assert(0 && "UnaryInst opID invalid!");
            break;
    }
    return instr_ir;
}

std::string ICmpInst::print()
{
    std::string instr_ir;
    instr_ir += "%";
    instr_ir += this->name_;
    instr_ir += " = ";
    instr_ir += instr_id2string_[this->op_id_];
    instr_ir += " ";
    instr_ir += print_cmp_type(this->icmp_op_);
    instr_ir += " ";
    instr_ir += this->get_operand(0)->type_->print();
    instr_ir += " ";
    instr_ir += print_as_op(this->get_operand(0), false);
    instr_ir += ", ";
    if (this->get_operand(0)->type_->tid_ == this->get_operand(1)->type_->tid_) {
        instr_ir += print_as_op(this->get_operand(1), false);
    }
    else {
        instr_ir += print_as_op(this->get_operand(1), true);
    }
    return instr_ir;
}

std::string FCmpInst::print()
{
    std::string instr_ir;
    instr_ir += "%";
    instr_ir += this->name_;
    instr_ir += " = ";
    instr_ir += instr_id2string_[this->op_id_];
    instr_ir += " ";
    instr_ir += print_fcmp_type(this->fcmp_op_);
    instr_ir += " ";
    instr_ir += this->get_operand(0)->type_->print();
    instr_ir += " ";
    instr_ir += print_as_op(this->get_operand(0), false);
    instr_ir += ", ";
    if (this->get_operand(0)->type_->tid_ == this->get_operand(1)->type_->tid_)
    {
        instr_ir += print_as_op(this->get_operand(1), false);
    }
    else
    {
        instr_ir += print_as_op(this->get_operand(1), true);
    }
    return instr_ir;
}

std::string CallInst::print()
{
    std::string instr_ir;
    if(!(this->type_->tid_ == Type::VoidTyID))
    {
        instr_ir += "%";
        instr_ir += this->name_;
        instr_ir += " = ";
    }
    instr_ir += instr_id2string_[this->op_id_];
    instr_ir += " ";
    unsigned int numops = this->num_ops_;
    instr_ir += static_cast<FunctionType *>(this->get_operand(numops-1)->type_)->result_->print();

    instr_ir += " ";
    assert(dynamic_cast<Function *>(this->get_operand(numops - 1)) && "Wrong call operand function");
    //__aeabi_memclr4 -> llvm_memset
    if (dynamic_cast<Function *>(this->get_operand(numops - 1))->name_ == "__aeabi_memclr4") {
        instr_ir += "@llvm.memset.p0.i32(";
        //i32* 目的内存地址
        instr_ir += this->get_operand(0)->type_->print();
        instr_ir += " ";
        instr_ir += print_as_op(this->get_operand(0), false);
        //i8 0
        instr_ir += ", i8 0, ";
        //i32 修改总字节数
        instr_ir += this->get_operand(1)->type_->print();
        instr_ir += " ";
        instr_ir += print_as_op(this->get_operand(1), false);
        //i1 false
        instr_ir += ", i1 false)";
        return instr_ir;
    }

    instr_ir += print_as_op(this->get_operand(numops - 1), false);
    instr_ir += "(";
    for (unsigned int i = 0; i < numops - 1; i++) {
        if( i > 0 )
            instr_ir += ", ";
        instr_ir += this->get_operand(i)->type_->print();
        instr_ir += " ";
        instr_ir += print_as_op(this->get_operand(i), false);
    }
    instr_ir += ")";
    return instr_ir;
}

std::string BranchInst::print()
{
    std::string instr_ir;
    instr_ir += instr_id2string_[this->op_id_];
    instr_ir += " ";
    instr_ir += print_as_op(this->get_operand(0), true);

    if( this->num_ops_ == 3 ) {
        instr_ir += ", ";
        instr_ir += print_as_op(this->get_operand(1), true);
        instr_ir += ", ";
        instr_ir += print_as_op(this->get_operand(2), true);
    }
    return instr_ir;
}

std::string ReturnInst::print()
{
    std::string instr_ir;
    instr_ir += instr_id2string_[this->op_id_];
    instr_ir += " ";
    if ( this->num_ops_ != 0 ) {
        instr_ir += this->get_operand(0)->type_->print();
        instr_ir += " ";
        instr_ir += print_as_op(this->get_operand(0), false);
    }
    else {
        instr_ir += "void";
    }

    return instr_ir;
}

std::string GetElementPtrInst::print()
{
    std::string instr_ir;
    instr_ir += "%";
    instr_ir += this->name_;
    instr_ir += " = ";
    instr_ir += instr_id2string_[this->op_id_];
    instr_ir += " ";
    assert(this->get_operand(0)->type_->tid_==Type::PointerTyID);
    instr_ir += static_cast<PointerType *>(this->get_operand(0)->type_)->contained_->print();
    instr_ir += ", ";
    for (unsigned int i = 0; i < this->num_ops_; i++)
    {
        if( i > 0 )
            instr_ir += ", ";
        instr_ir += this->get_operand(i)->type_->print();
        instr_ir += " ";
        instr_ir += print_as_op(this->get_operand(i), false);
    }
    return instr_ir;
}

std::string StoreInst::print()
{
    std::string instr_ir;
    instr_ir += instr_id2string_[this->op_id_];
    instr_ir += " ";
    instr_ir += this->get_operand(0)->type_->print();
    instr_ir += " ";
    instr_ir += print_as_op(this->get_operand(0), false);
    instr_ir += ", ";
    instr_ir += print_as_op(this->get_operand(1), true);
    return instr_ir;
}

std::string LoadInst::print()
{
    std::string instr_ir;
    instr_ir += "%";
    instr_ir += this->name_;
    instr_ir += " = ";
    instr_ir += instr_id2string_[this->op_id_];
    instr_ir += " ";
    assert(this->get_operand(0)->type_->tid_ == Type::PointerTyID);
    instr_ir += static_cast<PointerType *>(this->get_operand(0)->type_)->contained_->print();
    instr_ir += ",";
    instr_ir += " ";
    instr_ir += print_as_op(this->get_operand(0), true);
    return instr_ir;
}

std::string AllocaInst::print()
{
    std::string instr_ir;
    instr_ir += "%";
    instr_ir += this->name_;
    instr_ir += " = ";
    instr_ir += instr_id2string_[this->op_id_];
    instr_ir += " ";
    instr_ir += alloca_ty_->print();
    return instr_ir;
}

std::string ZextInst::print()
{
    std::string instr_ir;
    instr_ir += "%";
    instr_ir += this->name_;
    instr_ir += " = ";
    instr_ir += instr_id2string_[this->op_id_];
    instr_ir += " ";
    instr_ir += this->get_operand(0)->type_->print();
    instr_ir += " ";
    instr_ir += print_as_op(this->get_operand(0), false);
    instr_ir += " to ";
    instr_ir += this->dest_ty_->print();
    return instr_ir;
}

std::string FpToSiInst::print()
{
    std::string instr_ir;
    instr_ir += "%";
    instr_ir += this->name_;
    instr_ir += " = ";
    instr_ir += instr_id2string_[this->op_id_];
    instr_ir += " ";
    instr_ir += this->get_operand(0)->type_->print();
    instr_ir += " ";
    instr_ir += print_as_op(this->get_operand(0), false);
    instr_ir += " to ";
    instr_ir += this->dest_ty_->print();
    return instr_ir;
}

std::string SiToFpInst::print()
{
    std::string instr_ir;
    instr_ir += "%";
    instr_ir += this->name_;
    instr_ir += " = ";
    instr_ir += instr_id2string_[this->op_id_];
    instr_ir += " ";
    instr_ir += this->get_operand(0)->type_->print();
    instr_ir += " ";
    instr_ir += print_as_op(this->get_operand(0), false);
    instr_ir += " to ";
    instr_ir += this->dest_ty_->print();
    return instr_ir;
}

std::string Bitcast::print() {
    std::string instr_ir;
    instr_ir += "%";
    instr_ir += this->name_;
    instr_ir += " = ";
    instr_ir += instr_id2string_[this->op_id_];
    instr_ir += " ";
    instr_ir += this->get_operand(0)->type_->print();
    instr_ir += " ";
    instr_ir += print_as_op(this->get_operand(0), false);
    instr_ir += " to ";
    instr_ir += this->dest_ty_->print();
    return instr_ir;
}

std::string PhiInst::print()
{
    std::string instr_ir;
    instr_ir += "%";
    instr_ir += this->name_;
    instr_ir += " = ";
    instr_ir += instr_id2string_[this->op_id_];
    instr_ir += " ";
    instr_ir += this->get_operand(0)->type_->print();
    instr_ir += " ";
    for (int i = 0; i < this->num_ops_ / 2; i++) {
        if( i > 0 )
            instr_ir += ", ";
        instr_ir += "[ ";
        instr_ir += print_as_op(this->get_operand(2*i), false);
        instr_ir += ", ";
        instr_ir += print_as_op(this->get_operand(2*i+1), false);
        instr_ir += " ]";
    }
    if ( this->num_ops_ / 2 < this->parent_->pre_bbs_.size() ) {
        for ( auto pre_bb : this->parent_->pre_bbs_ ) {
            if (std::find(this->operands_.begin(), this->operands_.end(), static_cast<Value *>(pre_bb)) == this->operands_.end()) {
                // find a pre_bb is not in phi
                instr_ir += ", [ undef, " +print_as_op(pre_bb, false)+" ]";
            }
        }
    }
    return instr_ir; 
}

std::string print_as_op( Value *v, bool print_ty )
{
    std::string op_ir;
    if( print_ty ) {
        op_ir += v->type_->print();
        op_ir += " ";
    }

    if (dynamic_cast<GlobalVariable *>(v)) {
        op_ir += "@" + v->name_;
    }
    else if ( dynamic_cast<Function *>(v) ) {
        op_ir += "@" + v->name_;
    }
    else if ( dynamic_cast<Constant *>(v)) {
        op_ir += v->print();
    }
    else {
        op_ir += "%" + v->name_;
    }

    return op_ir;
}

std::string print_cmp_type( ICmpInst::ICmpOp op )
{
    switch (op)
    {
        case ICmpInst::ICMP_SGE:
            return "sge";
            break;
        case ICmpInst::ICMP_SGT:
            return "sgt";
            break;
        case ICmpInst::ICMP_SLE:
            return "sle";
            break;
        case ICmpInst::ICMP_SLT:
            return "slt";
            break;
        case ICmpInst::ICMP_EQ:
            return "eq";
            break;
        case ICmpInst::ICMP_NE:
            return "ne";
            break;
        default:
            break;
    }
    return "wrong cmpop";
}

std::string print_fcmp_type( FCmpInst::FCmpOp op )
{
    switch (op)
    {
        case FCmpInst::FCMP_UGE:
            return "uge";
            break;
        case FCmpInst::FCMP_UGT:
            return "ugt";
            break;
        case FCmpInst::FCMP_ULE:
            return "ule";
            break;
        case FCmpInst::FCMP_ULT:
            return "ult";
            break;
        case FCmpInst::FCMP_UEQ:
            return "ueq";
            break;
        case FCmpInst::FCMP_UNE:
            return "une";
            break;
        default:
            break;
    }
    return "wrong fcmpop";
}

void Function::set_instr_name() {
    std::map<Value*, int> seq;
    for (auto arg : this->arguments_) {
        if (!seq.count(arg)) {
            auto seq_num = seq.size() + seq_cnt_;
            if (arg->name_ == "") {
                arg->name_ = "arg_" + std::to_string(seq_num);
                seq.insert({ arg, seq_num });
            }
        }
    }
    for (auto bb : basic_blocks_) {
        if (!seq.count(bb)) {
            auto seq_num = seq.size() + seq_cnt_;
            if (bb->name_.length() <= 6 || bb->name_.substr(0, 6) != "label_") {
                bb->name_ = "label_" + std::to_string(seq_num);
                seq.insert({ bb, seq_num });
            }
        }
        for (auto instr : bb->instr_list_) {
            if (instr->type_->tid_ != Type::VoidTyID && !seq.count(instr)) {
                auto seq_num = seq.size() + seq_cnt_;
                if (instr->name_ == "") {
                    instr->name_ = "v" + std::to_string(seq_num);
                    seq.insert({ instr, seq_num });
                }
            }
        }
    }
    seq_cnt_ += seq.size();
}
