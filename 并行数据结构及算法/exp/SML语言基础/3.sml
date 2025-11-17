fun printInt (a:int) =
    print(Int.toString(a)^" ");

fun printReal (a:real) =
    print(Real.toString(a)^" ");

fun printString (a:string) =
    print(a^" ");

fun getInt () =
    Option.valOf (TextIO.scanStream (Int.scan StringCvt.DEC) TextIO.stdIn);
fun getReal () =
    Option.valOf (TextIO.scanStream (Real.scan) TextIO.stdIn);

type vector = int * int;

fun det(v1:vector, v2:vector):int =
    (#1 v1 * #2 v2 - #1 v2 * #2 v1);

(*****Begin*****)
fun solveX (a:int, b:int, c:int, d:int, k1:int, k2:int):string =
    let
        val detA = det((a, b), (c, d))
        val detX = det((k1, b), (k2, d))
    in
        if detA = 0 then "No Solution"
        else Int.toString(detX div detA)
    end;

val a = getInt();
val b = getInt();
val k1 = getInt();
val c = getInt();
val d = getInt();
val k2 = getInt();

val result = solveX(a, b, c, d, k1, k2);
printString(result);    
(*****End*****)
