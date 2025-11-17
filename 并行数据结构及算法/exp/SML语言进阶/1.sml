fun printInt (a:int) =
    print(Int.toString(a)^" ");

fun getInt () =
    Option.valOf (TextIO.scanStream (Int.scan StringCvt.DEC) TextIO.stdIn);

fun printIntTable ( [] ) = ()
  | printIntTable ( x::xs ) = 
    let
	val tmp = printInt(x)
    in
	printIntTable(xs)
    end;

fun getIntTable ( 0 ) = []
  | getIntTable ( N:int) = getInt()::getIntTable(N-1);

(*****Begin*****)			 
fun quicksort ([]) = []
  | quicksort (x::xs) =
      let
          val (left, right) = List.partition (fn y => y < x) xs
      in
          quicksort(left) @ [x] @ quicksort(right)
      end;

val n = getInt();
val arr = getIntTable(n);
val sortedArr = quicksort(arr);
printIntTable(sortedArr);		
(*****End*****)
