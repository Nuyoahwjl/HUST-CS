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
fun shortestPaths() =
    let
        val (N, M, T) = (getInt(), getInt(), getInt())
        
        (* 创建邻接表 *)
        val adj = Array.array(N+1, [])
        
        (* 读取边并构建图 *)
        fun readEdges(0) = ()
          | readEdges(count) =
            let
                val u = getInt()
                val v = getInt()
                val w = getInt()
                val _ = Array.update(adj, u, (v, w)::Array.sub(adj, u))
                val _ = Array.update(adj, v, (u, w)::Array.sub(adj, v))
            in
                readEdges(count-1)
            end
        
        (* Dijkstra算法 *)
        val INF = 1000000000
        val dist = Array.array(N+1, INF)
        val visited = Array.array(N+1, false)
        
        fun dijkstra() =
            let
                (* 初始化源点 *)
                val _ = Array.update(dist, T, 0)
                
                (* 寻找未访问的最小距离节点 *)
                fun findMinNode() =
                    let
                        fun findMin(i, minDist, minNode) =
                            if i > N then minNode
                            else if not(Array.sub(visited, i)) andalso Array.sub(dist, i) < minDist 
                                 then findMin(i+1, Array.sub(dist, i), i)
                            else findMin(i+1, minDist, minNode)
                    in
                        findMin(1, INF, ~1)
                    end
                
                (* 主循环 *)
                fun mainLoop() =
                    let
                        val u = findMinNode()
                    in
                        if u = ~1 then ()  (* 所有节点都已访问或不可达 *)
                        else 
                            let
                                val _ = Array.update(visited, u, true)
                                
                                (* 更新邻居节点的距离 *)
                                fun updateNeighbors([]) = ()
                                  | updateNeighbors((v, w)::neighbors) =
                                    let
                                        val newDist = Array.sub(dist, u) + w
                                        val currentDist = Array.sub(dist, v)
                                    in
                                        if newDist < currentDist then
                                            Array.update(dist, v, newDist)
                                        else ();
                                        updateNeighbors(neighbors)
                                    end
                            in
                                updateNeighbors(Array.sub(adj, u));
                                mainLoop()
                            end
                    end
            in
                mainLoop()
            end
        
        (* 输出结果 *)
        fun printResults(i) =
            if i > N then ()
            else 
                let
                    val d = Array.sub(dist, i)
                in
                    if d = INF then print("~1 ")
                    else print(Int.toString(d) ^ " ");
                    printResults(i+1)
                end
    
    in
        readEdges(M);
        dijkstra();
        printResults(1);
        printEndOfLine()
    end

val _ = shortestPaths()
(*****End*****)

