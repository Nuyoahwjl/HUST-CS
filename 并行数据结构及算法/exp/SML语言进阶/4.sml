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
type building = {l:int, r:int, h:int}
type point = int * int

(* 合并两个子天际线 *)
fun mergeSkylines (s1: point list, s2: point list) : point list =
  let
    (*
      h1, h2: 扫描线在当前位置时，来自s1和s2的各自高度
      result: 存储合并后的点（倒序）
    *)
    fun merge (sl1, sl2, h1, h2, result) =
      case (sl1, sl2) of
        ([], []) => result (* 两个列表都处理完毕 *)
      | (p::ps, []) => merge(ps, [], #2 p, h2, p::result) (* s2已空，处理s1剩余部分 *)
      | ([], p::ps) => merge([], ps, h1, #2 p, p::result) (* s1已空，处理s2剩余部分 *)
      | (p1 as (x1, y1)::ps1, p2 as (x2, y2)::ps2) =>
        let
          val lastMaxH = Int.max(h1, h2) (* 进入此点前的轮廓高度 *)
        in
          if x1 < x2 then
            let val newMaxH = Int.max(y1, h2) in
              if newMaxH <> lastMaxH then merge(ps1, sl2, y1, h2, (x1, newMaxH)::result)
              else merge(ps1, sl2, y1, h2, result)
            end
          else if x2 < x1 then
            let val newMaxH = Int.max(h1, y2) in
              if newMaxH <> lastMaxH then merge(sl1, ps2, h1, y2, (x2, newMaxH)::result)
              else merge(sl1, ps2, h1, y2, result)
            end
          else (* x1 = x2, 在同一点上都有关键点 *)
            let val newMaxH = Int.max(y1, y2) in
              if newMaxH <> lastMaxH then merge(ps1, ps2, y1, y2, (x1, newMaxH)::result)
              else merge(ps1, ps2, y1, y2, result)
            end
        end
  in
    List.rev(merge(s1, s2, 0, 0, []))
  end

(* 分治主函数 *)
fun skyline (buildings: building list) : point list =
    case buildings of
      [] => []
    | [b] => [(#l b, #h b), (#r b, 0)] (* Base Case: 单个建筑的天际线 *)
    | _ =>
      let
        val n = List.length buildings
        val mid = n div 2
        val leftBuildings = List.take (buildings, mid)
        val rightBuildings = List.drop (buildings, mid)

        val leftSkyline = skyline leftBuildings
        val rightSkyline = skyline rightBuildings
      in
        mergeSkylines (leftSkyline, rightSkyline)
      end

(* 读取输入 *)
val n = getInt()
fun readBuildings 0 = []
  | readBuildings i =
    let val l=getInt() val h=getInt() val r=getInt()
    in {l=l, r=r, h=h} :: readBuildings(i-1) end
val buildings = readBuildings n

(* 计算并打印结果 *)
val finalSkyline = skyline buildings
fun printPoint (x, y) = (printInt x; printInt y; print "\n")
val _ = List.app printPoint finalSkyline

(*****End*****)


