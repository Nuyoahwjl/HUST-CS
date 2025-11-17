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
(* 如果 N 是偶数，则 A^N = (A^(N/2))^2。 *)
(* 如果 N 是奇数，则 A^N = A * A^(N-1)。 *)

fun modExp (a:IntInf.int, n:IntInf.int, m:IntInf.int): IntInf.int =
    if n = 0 then
        IntInf.fromInt 1
    else if IntInf.mod(n, IntInf.fromInt 2) = IntInf.fromInt 0 then
        let
            val half = modExp(a, IntInf.div(n, IntInf.fromInt 2), m)
        in
            IntInf.mod(IntInf.*(half, half), m)
        end
    else
        IntInf.mod(IntInf.*(a, modExp(a, IntInf.-(n, IntInf.fromInt 1), m)), m);

val A = getIntInf();
val N = getIntInf();
val M = getIntInf();

val result = modExp(A, N, M);
printIntInf(result);
(*****End*****)
