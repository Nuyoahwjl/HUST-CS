/*
  注意：应与你在SysyLex.g4中定义的token顺序完全一致。由于antlr4会从1开始编号，故种别名称数组的0号元素为空串。
*/
std::string tokenTypeName[] = {"", "INT", "FLOAT", "VOID", "CONST", "RETURN", "IF", "ELSE", 
     "WHILE", "BREAK", "CONTINUE", "LP", "RP", 
    "LB", "RB", "LC", "RC", "COMMA", "SEMICOLON", "QUESTION", 
    "COLON", "MINUS", "NOT", "ASSIGN", "ADD", "MUL", "DIV", 
    "MOD", "AND", "OR", "EQ", "NEQ", "LT", "LE", "GT", 
    "GE", "INT_LIT", "FLOAT_LIT", "ID", "STRING", "", 
    "", "", "LEX_ERR"};

