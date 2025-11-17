fun printInt (a:int) =
    print(Int.toString(a)^" ");

fun printIntInf (a:IntInf.int) =
    print(IntInf.toString(a)^" ");

fun printReal (a:real) =
    print(Real.toString(a)^" ");

fun printString (a:string) =
    print(a^" ");

fun getInt () =
    Option.valOf (TextIO.scanStream (Int.scan StringCvt.DEC) TextIO.stdIn);

fun getIntInf () =
    Option.valOf (TextIO.scanStream (IntInf.scan StringCvt.DEC) TextIO.stdIn);

fun getReal () =
    Option.valOf (TextIO.scanStream (Real.scan) TextIO.stdIn);

(*****Begin*****)
(* 矩阵乘法，计算两个2x2矩阵的乘积并对 m 取模 *)
fun matMult ((a11, a12, a21, a22): (IntInf.int * IntInf.int * IntInf.int * IntInf.int), 
             (b11, b12, b21, b22): (IntInf.int * IntInf.int * IntInf.int * IntInf.int), 
             m: IntInf.int): (IntInf.int * IntInf.int * IntInf.int * IntInf.int) =
    (
        IntInf.mod(IntInf.+(IntInf.*(a11, b11), IntInf.*(a12, b21)), m),
        IntInf.mod(IntInf.+(IntInf.*(a11, b12), IntInf.*(a12, b22)), m),
        IntInf.mod(IntInf.+(IntInf.*(a21, b11), IntInf.*(a22, b21)), m),
        IntInf.mod(IntInf.+(IntInf.*(a21, b12), IntInf.*(a22, b22)), m)
    );

(* 矩阵快速幂，计算矩阵 base 的 exp 次幂并对 m 取模 *)
fun matPow ((a11, a12, a21, a22): (IntInf.int * IntInf.int * IntInf.int * IntInf.int), 
            exp: IntInf.int, 
            m: IntInf.int): (IntInf.int * IntInf.int * IntInf.int * IntInf.int) =
    if exp = IntInf.fromInt 1 then
        (a11, a12, a21, a22)
    else if IntInf.mod(exp, IntInf.fromInt 2) = IntInf.fromInt 0 then
        let
            val halfPow = matPow((a11, a12, a21, a22), IntInf.div(exp, IntInf.fromInt 2), m)
        in
            matMult(halfPow, halfPow, m)
        end
    else
        matMult((a11, a12, a21, a22), matPow((a11, a12, a21, a22), IntInf.-(exp, IntInf.fromInt 1), m), m);

(* 计算斐波那契数列的第 N 项，并对 M 取模 *)
fun fibMod (n: IntInf.int, m: IntInf.int): IntInf.int =
    if n = IntInf.fromInt 0 then IntInf.fromInt 0
    else if n = IntInf.fromInt 1 orelse n = IntInf.fromInt 2 then IntInf.fromInt 1
    else
        let
            val (f11, f12, f21, f22) = matPow((IntInf.fromInt 1, IntInf.fromInt 1, IntInf.fromInt 1, IntInf.fromInt 0), n, m)
        in
            IntInf.mod(f21, m)
        end;

val N = getIntInf();
val M = getIntInf();

val result = fibMod(N, M);
printIntInf(result);
(*****End*****)
