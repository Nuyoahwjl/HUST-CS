lexer grammar SysyLex;

fragment HexPrefix
    : '0x'
    | '0X'
    ;
fragment OctPrefix : '0';

fragment NonzeroDigit : [1-9];
fragment Digit : [0-9];
fragment HexDigit : [0-9a-fA-F];
fragment OctDegit : [0-7];

DecIntConst : NonzeroDigit Digit*;
OctIntConst : OctPrefix OctDegit*;
HexIntConst : HexPrefix HexDigit+;

fragment Dot : '.';

fragment Sign : '+' | '-' ;

fragment Exponent : 'e' | 'E' ;
fragment HexExponent : 'p' | 'P' ;

fragment DecFloatFrac : Digit* Dot Digit+ | Digit+ Dot   ;
fragment HexFloatFrac : HexDigit* Dot HexDigit+ | HexDigit+ Dot ;

fragment DecFloatExp : Exponent Sign? Digit+;
fragment BinFloatExp : HexExponent Sign? Digit+;

DecFloatConst : DecFloatFrac DecFloatExp? | Digit+ DecFloatExp ;
HexFloatConst : HexPrefix HexFloatFrac BinFloatExp
    | HexPrefix HexDigit+ BinFloatExp
    ;

fragment Escaped : '\\'['"?\\abfnrtv];

StringConst : '"' (~['"\\\r\n] | Escaped)* '"';

Int : 'int';
Float : 'float';
Void : 'void';

Const : 'const';

If : 'if';
Else : 'else';
While : 'while';
Break : 'break';
Continue : 'continue';
Return : 'return';

Assign : '=';

Add : '+';
Sub : '-';
Mul : '*';
Div : '/';
Mod : '%';

Eq : '==';
Neq : '!=';
Lt : '<';
Gt : '>';
Leq : '<=';
Geq : '>=';

Not : '!';
And : '&&';
Or : '||';

Comma : ',';
Semicolon : ';';
Lparen : '(';
Rparen : ')';
Lbracket : '[';
Rbracket : ']';
Lbrace : '{';
Rbrace : '}';

Ident : [A-Za-z_][_0-9A-Za-z]*;

Whitespace : [ \t\r\n]+ -> skip;

LineComment : '//' ~[\r\n]* -> skip;
BlockComment : '/*' .*? '*/' -> skip;
