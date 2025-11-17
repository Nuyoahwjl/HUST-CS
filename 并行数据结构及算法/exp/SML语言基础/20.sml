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

fun getIntVector ( 0 ) =  Vector.fromList []
  | getIntVector ( N:int) = Vector.fromList(getIntTable(N));

fun getIntInfVector ( 0 ) = Vector.fromList []
  | getIntInfVector ( N:int) = Vector.fromList(getIntInfTable(N));

(*****Begin*****)
val horse_moves = [(2, 1), (1, 2), (~1, 2), (~2, 1), (~2, ~1), (~1, ~2), (1, ~2), (2, ~1)];

fun check (x:int, y:int, i:int, j:int, n:int, m:int) =
    if x = i andalso y = j then false
    else
        let
            fun isControlled (dx, dy) =
                let
                    val a1 = x + dx
                    val b1 = y + dy
                in
                    a1 >= 0 andalso a1 <= n andalso b1 >= 0 andalso b1 <= m andalso a1 = i andalso b1 = j
                end
        in
            (* 遍历 horse_moves 列表，并且检查 (i, j) 是否在马的控制范围内 *)
            not (List.exists isControlled horse_moves)
        end;

fun fillDP (dp: int Array2.array, n:int, m:int, x:int, y:int) =
    let
        val _ = Array2.update(dp, 0, 0, 1)
        
        val _ =
            let
                fun fillCol (i:int) =
                    if i > n orelse not (check(x, y, i, 0, n, m)) then ()
                    else (Array2.update(dp, i, 0, 1); fillCol (i + 1))
            in
                fillCol (1)
            end

        val _ =
            let
                fun fillRow (j:int) =
                    if j > m orelse not (check(x, y, 0, j, n, m)) then ()
                    else (Array2.update(dp, 0, j, 1); fillRow (j + 1))
            in
                fillRow (1)
            end
        
        (* 递归填充整个表格 *)
        val _ =
            let
                fun fillGrid (i:int) (j:int) =
                    if i > n then ()
                    else if j > m then fillGrid (i + 1) 1
                    else
                        let
                            val _ =
                                if not (check(x, y, i, j, n, m)) then
                                    Array2.update(dp, i, j, 0)
                                else
                                    let
                                        val up = Array2.sub(dp, i - 1, j)
                                        val left = Array2.sub(dp, i, j - 1)
                                        val sum = (if up <> ~1 then up else 0) + (if left <> ~1 then left else 0)
                                    in
                                        Array2.update(dp, i, j, sum)
                                    end
                        in
                            fillGrid i (j + 1)
                        end
            in
                fillGrid 1 1
            end
    in
        dp
    end

fun chessPath (n:int, m:int, x:int, y:int) =
    let
        val dp = Array2.array(n+1, m+1, 0)
        val _ = fillDP(dp, n, m, x, y)
    in
        if Array2.sub(dp, n, m) = ~1 then 0 else Array2.sub(dp, n, m)
    end

val n = getInt()
val m = getInt()
val x = getInt()
val y = getInt()
val result = chessPath(n, m, x, y);
printInt(result);
(*****End*****)



(*****Begin*****)
val horse_moves = [(2, 1), (1, 2), (~1, 2), (~2, 1), (~2, ~1), (~1, ~2), (1, ~2), (2, ~1)];

fun check (x:int, y:int, i:int, j:int, n:int, m:int) =
    if x = i andalso y = j then false
    else
        let
            fun isControlled (dx, dy) =
                let
                    val a1 = x + dx
                    val b1 = y + dy
                in
                    a1 >= 0 andalso a1 <= n andalso b1 >= 0 andalso b1 <= m andalso a1 = i andalso b1 = j
                end
        in
            not (List.exists isControlled horse_moves)
        end;

fun fillDPReverse (dp: int Array2.array, n:int, m:int, x:int, y:int, i:int, j:int) =
    if i < 0 orelse j < 0 then ~1  (* 越界返回-1 *)
    else
        case Array2.sub(dp, i, j) of
            ~1 => (* 如果该位置还未计算 *)
                if not (check(x, y, i, j, n, m)) then
                    (Array2.update(dp, i, j, 0); 0)
                else if i = 0 andalso j = 0 then
                    (Array2.update(dp, i, j, 1); 1)
                else if i = 0 then
                    let
                        val left = fillDPReverse(dp, n, m, x, y, i, j-1)
                    in
                        Array2.update(dp, i, j, left);
                        left
                    end
                else if j = 0 then
                    let
                        val up = fillDPReverse(dp, n, m, x, y, i-1, j)
                    in
                        Array2.update(dp, i, j, up);
                        up
                    end
                else
                    let
                        val up = fillDPReverse(dp, n, m, x, y, i-1, j)
                        val left = fillDPReverse(dp, n, m, x, y, i, j-1)
                        val sum = (if up <> ~1 then up else 0) + (if left <> ~1 then left else 0)
                    in
                        Array2.update(dp, i, j, sum);
                        sum
                    end
          | value => value  (* 如果已经计算过，直接返回值 *)

fun chessPath (n:int, m:int, x:int, y:int) =
    let
        val dp = Array2.array(n+1, m+1, ~1)  (* 初始化为-1表示未计算 *)
        val result = fillDPReverse(dp, n, m, x, y, n, m)
    in
        if result = ~1 then 0 else result
    end

val n = getInt()
val m = getInt()
val x = getInt()
val y = getInt()
val result = chessPath(n, m, x, y);
printInt(result);
(*****End*****)