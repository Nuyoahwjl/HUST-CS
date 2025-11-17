
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

(*****Begin*****)
fun gcd (n:int, m:int) = 
    if m= 0 then n
    else gcd(m,n mod m);
val a = getInt();
val b = getInt();
val c = gcd(a,b);
printInt(c);
(*****End*****)
