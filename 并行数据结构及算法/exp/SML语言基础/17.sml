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

(*****Begin*****)
fun reverse (ys, zs) = case ys of
      [] => zs
    | y::ys => reverse (ys, y::zs)

fun find_k (xs, k) = 
    if k < 0 then ~1
    else if List.nth(xs, k) < List.nth(xs, k + 1) then k
    else find_k (xs, k - 1)

fun find_l (xs, l, k) = 
    if List.nth(xs, k) < List.nth(xs, l) then l
    else find_l (xs, l - 1, k)

fun next_permutation (xs : int list) =
    let
        val k = find_k (xs, List.length(xs) - 2)
        val l = if k = ~1 then ~1 else find_l (xs, List.length(xs) - 1, k)
        
        val xs_new = if k = ~1 orelse l = ~1 then xs
                     else List.take(xs, k) @ [List.nth(xs, l)] @ List.take(List.drop(xs, k + 1), l - k - 1) @ [List.nth(xs, k)] @ List.drop(xs, l + 1)
    in
        if k = ~1 orelse l = ~1 then reverse (xs, [])
        else List.take(xs_new, k + 1) @ reverse (List.drop(xs_new, k + 1), [])
    end

val N = getInt();
val permutation = getIntTable(N);
val reversedpermutation = List.rev(permutation);
val next_perm = next_permutation(reversedpermutation);
val reversednext_perm = List.rev(next_perm);
printIntTable(reversednext_perm);
(*****End*****)
