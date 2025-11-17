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
        val _ = printInt(x)
    in
        printIntTable(xs)
    end;

fun printIntInfTable ( [] ) = ()
  | printIntInfTable ( x::xs ) = 
    let
        val _ = printIntInf(x)
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
(* 读取图的节点数 n 和边数 m *)
val n = getInt() and m = getInt(); 
(* 初始化图的邻接表，使用数组存储，每个节点对应一个 (邻居节点, 边编号) 的列表 *)
val graph: (int * int) list array = Array.array(n, []);

(* 根据输入的边信息构建图 *)
fun buildGraphFromEdge edgeIndex =
    let 
        val u = getInt() - 1 and v = getInt() - 1
    in
        (* 将边加入到 u 和 v 的邻接表中 *)
        Array.update(graph, u, (v, edgeIndex)::(Array.sub(graph, u)));
        Array.update(graph, v, (u, edgeIndex)::(Array.sub(graph, v)));
        0
    end;

(* 读取 m 条边并构建图 *)
List.tabulate(m, buildGraphFromEdge); 

(* 初始化割点和割边的标记数组 *)
val isArticulationPoint = Array.array(n, false); 
val isBridge = Array.array(m, false); 

(* 初始化时间戳数组 *)
val dfn = Array.array(n, 0); 
val low = Array.array(n, 0); 
val timestamp = Array.array(1, 1);

(* Tarjan 算法，用于求解关节点和桥 *)
fun tarjan parentNode currentNode =
    let
        (* 获取当前时间戳 *)
        val currentTimestamp = Array.sub(timestamp, 0) 
        (* 更新时间戳 *)
        val _ = Array.update(timestamp, 0, currentTimestamp+1) 
        (* 初始化当前节点的 dfn 和 low 值 *)
        val _ = Array.update(dfn, currentNode, currentTimestamp) 
        val _ = Array.update(low, currentNode, currentTimestamp) 
        (* 根节点的子节点计数器 *)
        val rootChildrenCount = Array.array(1, 0)
        (* 遍历当前节点的所有邻居 *)
        fun traverseNeighbors (neighbor, edgeId) =
            if (Array.sub (dfn, neighbor)) = 0 then
                (* 如果邻居节点未被访问 *)
                let 
                    (* 如果是根节点，增加子节点计数 *)
                    val _ = Array.update (rootChildrenCount, 0, (Array.sub (rootChildrenCount, 0)) + 1) 
                    (* 递归访问邻居节点 *)
                    val _ = tarjan currentNode neighbor 
                    (* 更新当前节点的 low 值 *)
                    val _ = Array.update(low, currentNode, Int.min(Array.sub (low, currentNode), Array.sub (low, neighbor))) 
                    (* 判断根节点是否为关节点 *)
                    val _ = 
                        if currentNode = 0 andalso (Array.sub (rootChildrenCount, 0)) > 1 
                            then (Array.update (isArticulationPoint, currentNode, true)) 
                        else () 
                    (* 判断非根节点是否为关节点 *)
                    val _ = 
                        if currentNode <> 0 andalso (Array.sub (dfn, currentNode)) <= (Array.sub(low, neighbor)) 
                            then (Array.update (isArticulationPoint, currentNode, true)) 
                        else () 
                    (* 判断是否为桥 *)
                    val _ = 
                        if (Array.sub (dfn, currentNode)) < (Array.sub (low, neighbor)) 
                            then (Array.update (isBridge, edgeId, true)) else ()
                in () 
                end
            else if neighbor <> parentNode 
                (* 如果邻居节点已访问且不是父节点，更新 low 值 *)
                then Array.update (low, currentNode, Int.min(Array.sub (low, currentNode), Array.sub (dfn, neighbor)))             
            else () 
        (* 遍历当前节点的所有邻居 *)
        val _ = List.app traverseNeighbors (Array.sub (graph, currentNode))
    in () 
    end;

(* 从节点 0 开始运行 Tarjan 算法 *)
tarjan ~1 0;

(* 输出关节点的数量 *)
printInt (Array.foldl (fn(isPoint, count) => if isPoint then count + 1 else count) 0 isArticulationPoint);
(* 输出桥的数量 *)
printInt (Array.foldl (fn(isBridge, count) => if isBridge then count + 1 else count) 0 isBridge);
printEndOfLine();

(*****End*****)