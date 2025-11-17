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
fun gcd (a:IntInf.int, b:IntInf.int): IntInf.int =
    if b = IntInf.fromInt 0 then a
    else gcd (b, IntInf.mod (a, b));

fun simplify (numerator: IntInf.int, denominator: IntInf.int): (IntInf.int * IntInf.int) =
    let
        val g = gcd (numerator, denominator) 
    in
        (IntInf.div (numerator, g), IntInf.div (denominator, g)) 
    end;

val A1 = getIntInf(); 
val A2 = getIntInf(); 
val B1 = getIntInf(); 
val B2 = getIntInf(); 

val numerator = IntInf.+(IntInf.*(A1, B2), IntInf.*(B1, A2));
val denominator = IntInf.*(A2, B2);

val (simpleNumerator, simpleDenominator) = simplify(numerator, denominator);

printIntInf(simpleNumerator);
printIntInf(simpleDenominator);
(*****End*****)
