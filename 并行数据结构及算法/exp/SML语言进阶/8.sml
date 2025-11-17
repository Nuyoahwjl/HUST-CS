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
(* 线段树节点的定义 *)
datatype segtree = Leaf of int | Node of int * segtree * segtree;

(* 创建线段树 *)
fun buildTree (arr: int array, start: int, End: int): segtree =
    if start = End then
        Leaf (Array.sub(arr, start))   (* 如果是叶子节点，返回该元素 *)
    else
        let
            val mid = (start + End) div 2
            val left = buildTree (arr, start, mid)  (* 构建左子树 *)
            val right = buildTree (arr, mid + 1, End)  (* 构建右子树 *)
            val maxVal = case (left, right) of
                (Leaf v1, Leaf v2) => Int.max(v1, v2)  (* 计算最大值 *)
              | (Node (v1, _, _), Leaf v2) => Int.max(v1, v2)
              | (Leaf v1, Node (v2, _, _)) => Int.max(v1, v2)
              | (Node (v1, _, _), Node (v2, _, _)) => Int.max(v1, v2)
        in
            Node (maxVal, left, right)  (* 返回当前节点 *)
        end;

(* 查询线段树区间最大值 *)
fun rangeMaxQuery (tree: segtree, L: int, R: int, start: int, End: int): int =
    if R < start orelse L > End then
        ~1  (* 返回 -1，表示无效查询 *)
    else if L <= start andalso R >= End then
        case tree of
            Leaf v => v
          | Node (v, _, _) => v  (* 当前节点最大值 *)
    else
        case tree of
            Node (_, left, right) =>
                let
                    val mid = (start + End) div 2
                    val leftMax = rangeMaxQuery (left, L, R, start, mid)  (* 查询左子树 *)
                    val rightMax = rangeMaxQuery (right, L, R, mid + 1, End)  (* 查询右子树 *)
                in
                    Int.max(leftMax, rightMax)  (* 返回左右子树的最大值 *)
                end;

val N = getInt();
val M = getInt();
val NList = getIntTable(N);
val tree = buildTree (Array.fromList NList, 0, N-1);
val ans = List.tabulate(M, fn _ => (let val L = getInt() and R = getInt() in rangeMaxQuery(tree, L-1, R-1, 0, N-1) end));
printIntTable(ans);
(*****End*****)



(*******
```mermaid
graph TD
    Q1[Start rangeMaxQuery] --> Q2{No overlap? <br> R < start or L > end};
    Q2 -- Yes --> Q3["Return sentinel value (e.g., -1)"];
    Q2 -- No --> Q4{Total overlap? <br> L <= start and R >= end};
    Q4 -- Yes --> Q5[Return value of current tree node];
    Q4 -- No --> Q6[Partial overlap];
    Q6 --> Q7["mid = (start + end) / 2"];
    Q7 --> Q8["leftMax = query on left <br> (left, L, R, start, mid)"];
    Q7 --> Q9["rightMax = query on right <br> (right, L, R, mid+1, end)"];
    Q8 & Q9 --> Q10["Return <br> max(leftMax, rightMax)"];
    Q3 --> Q11[End rangeMaxQuery];
    Q5 --> Q11;
    Q10 --> Q11;
```
*******)