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

fun printArray ( Arr ) =
    let
	val cur = ref 0
	val len = Array.length(Arr)
    in
	while !cur < len
	do
	(
	  printInt(Array.sub(Arr,!cur));
	  cur := !cur + 1)
    end;

fun printString( s ) = print(s ^ " ");

(*****Begin*****)			 			 
val N = getInt();
val s = getIntTable(N);

fun parenDist((pos, x), (stack, max)) =
    if x = 0 then 
        (pos::stack, max)
    else if stack = [] then 
        (stack, max) 
    else
        let 
            val top = hd stack
            val tmp = Int.max(max, pos - top + 1)
        in
            (tl stack, tmp)
        end;

val indexed = ListPair.zip(List.tabulate(N, fn x => x), s);
val res = #2(foldl parenDist ([], 0) indexed);
printInt(res);
(*****End*****)


