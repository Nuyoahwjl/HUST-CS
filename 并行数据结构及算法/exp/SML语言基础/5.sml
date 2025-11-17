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
fun fibMod (n:IntInf.int, m:IntInf.int): IntInf.int =
    let
        (* 初始化 F(1) = F(2) = 1 *)
        fun iter (i:IntInf.int, prev:IntInf.int, curr:IntInf.int): IntInf.int =
            if i = n then curr
            else iter(IntInf.+(i, IntInf.fromInt 1), curr, IntInf.mod(IntInf.+(prev, curr), m))
    in
        if n = IntInf.fromInt 1 orelse n = IntInf.fromInt 2 then IntInf.fromInt 1
        else iter(IntInf.fromInt 3, IntInf.fromInt 1, IntInf.fromInt 2)
    end;
val N = getIntInf();
val M = getIntInf();

val result = fibMod(N, M);
printIntInf(result);
(*****End*****)
