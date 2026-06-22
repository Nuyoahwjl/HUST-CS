#include "codegen.h"
#include <memory>
#include <optional>
#include <string>

using namespace llvm;
using namespace llvm::sys;

namespace codegen {

bool codeGenerate(const std::string &ir_filename,
                  const CodeGenFileType &gen_filetype) {
  SMDiagnostic error_smdiagnostic;
  LLVMContext context;
  std::unique_ptr<Module> module =
      parseIRFile(ir_filename, error_smdiagnostic, context);

  if (!module) {
    error_smdiagnostic.print(ir_filename.c_str(), errs());
    return false;
  }

  // 补充代码1 - 初始化目标
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  // 补充代码2 - 指定目标平台
  auto target_triple = "riscv64-unknown-elf";
  // 如果评测要求 ARMv7，可改为：
  // auto target_triple = "armv7-unknown-linux-gnueabihf";

  module->setTargetTriple(target_triple);

  std::string error_string;
  auto target = TargetRegistry::lookupTarget(target_triple, error_string);

  if (!target) {
    errs() << error_string;
    return false;
  }

  auto cpu = "generic-rv64";
  // 如果评测要求 ARMv7，可改为：
  // auto cpu = "generic";

  auto features = "";

  TargetOptions opt;
  auto RM = Optional<Reloc::Model>();
  auto TheTargetMachine =
      target->createTargetMachine(target_triple, cpu, features, opt, RM);

  module->setDataLayout(TheTargetMachine->createDataLayout());

  // 补充代码3 - 初始化 addPassesToEmitFile() 的参数
  auto filename = getGenFilename(ir_filename, gen_filetype);

  std::error_code EC;
  raw_fd_ostream dest(filename, EC, sys::fs::OF_None);

  if (EC) {
    errs() << "Could not open file: " << EC.message() << "\n";
    return false;
  }

  legacy::PassManager pass;
  auto file_type = gen_filetype;

  if (TheTargetMachine->addPassesToEmitFile(pass, dest, nullptr, file_type)) {
    errs() << "TheTargetMachine can't emit a file of this type";
    return false;
  }

  pass.run(*module);
  dest.flush();

  return true;
}

std::string getGenFilename(const std::string &ir_filename,
                           const CodeGenFileType &gen_filetype) {
  if (gen_filetype == CGFT_Null) {
    return nullptr;
  }

  return ir_filename.substr(0, ir_filename.find(".")) +
         (gen_filetype == CGFT_AssemblyFile ? ".s" : ".o");
}

} // namespace codegen