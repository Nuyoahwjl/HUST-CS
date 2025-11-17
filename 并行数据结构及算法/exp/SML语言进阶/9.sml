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

fun printEndOfLine () =
    print("\n");

fun printIntTable ( [] ) = ()
  | printIntTable ( x::xs ) = 
    let
	val tmp = printInt(x)
    in
	printIntTable(xs)
    end;

fun printIntInfTable ( [] ) = ()
  | printIntInfTable ( x::xs ) = 
    let
	val tmp = printIntInf(x)
    in
	printIntInfTable(xs)
    end;

fun getIntTable ( 0 ) = []
  | getIntTable ( N:int) = getInt()::getIntTable(N-1);

fun getIntInfTable ( 0 ) = []
  | getIntInfTable ( N:int) = getIntInf()::getIntInfTable(N-1);

fun getIntVector ( 0 ) =  Vector.fromList []
  | getIntVector ( N:int) = Vector.fromList(getIntTable(N));

fun getIntInfVector ( 0 ) = Vector.fromList []
  | getIntInfVector ( N:int) = Vector.fromList(getIntInfTable(N));

(*****Begin*****)
val numberToTest = getIntInf(); 
val primeBases:IntInf.int list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97];

if (List.exists (fn prime => prime = numberToTest) primeBases) 
    then printString("True") 
else
    let
        fun modularExponentiation (base:IntInf.int, exponent:IntInf.int, modulus:IntInf.int) = 
            case exponent of 
                0 => 1
              | 1 => base mod modulus
              | _ =>
                    let
                        val half = modularExponentiation (base, exponent div 2, modulus)
                        val halfSq = half * half mod modulus
                    in
                        if exponent mod 2 = 0 
                            then halfSq
                        else halfSq * base mod modulus
                    end;

        fun factorOutPowersOfTwo (x:IntInf.int, s:int) =
            if x mod 2 = 0 
                then factorOutPowersOfTwo (x div 2, s + 1) 
            else (x, s); (* Returns (d, s) where n-1 = d * 2^s *)

        val (d, s) = factorOutPowersOfTwo (numberToTest - 1, 0);

        fun millerRabinTest(base:IntInf.int, allPassed:bool) =
            if not allPassed
                then false
            else
                let
                    val x = modularExponentiation(base, d, numberToTest);

                    fun checkSquares(i, (isProbablePrime, prevX:IntInf.int)) =
                        if not isProbablePrime
                            then (false, prevX) 
                        else
                            let
                                val currentX = prevX * prevX mod numberToTest;
                                (* If currentX is 1, then prevX must have been 1 or n-1 *)
                                val passed = not(currentX = 1 andalso prevX <> 1 andalso prevX <> numberToTest - 1);
                            in
                                (passed, currentX) 
                            end; 

                    val (passedAllSquares, finalX) = List.foldl checkSquares (true, x) (List.tabulate(s, fn i => i));
                in
                    passedAllSquares andalso finalX = 1
                end;
    in
        if List.foldl millerRabinTest true primeBases then printString("True") else printString("False") 
    end;
(*****End*****)

