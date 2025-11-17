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

fun printIntTable ( []) = ()
  | printIntTable ( x::xs ) = (printInt(x); printIntTable(xs));

fun printIntInfTable ( [] ) = ()
  | printIntInfTable ( x::xs ) = (printIntInf(x); printIntInfTable(xs));

fun getIntTable ( 0 ) = []
  | getIntTable ( N:int) = getInt()::getIntTable(N-1);

fun getIntInfTable ( 0 ) = []
  | getIntInfTable ( N:int) = getIntInf()::getIntInfTable(N-1);

fun getIntVector ( 0 ) =  Vector.fromList []
  | getIntVector ( N:int) = Vector.fromList(getIntTable(N));

fun getIntInfVector ( 0 ) = Vector.fromList []
  | getIntInfVector ( N:int) = Vector.fromList(getIntInfTable(N));

(*****Begin*****)
(* 移除列表前导的0 *)
fun trimLeadingZeros [] = [0]
  | trimLeadingZeros (0::xs) = trimLeadingZeros xs
  | trimLeadingZeros list = list;

(* --- 数据读入与预处理 --- *)
val numDigitsA = getInt(); 
val numA = List.rev (trimLeadingZeros (getIntTable(numDigitsA))); 
val numDigitsB = getInt(); 
val numB = List.rev (trimLeadingZeros (getIntTable(numDigitsB)));

(* --- 高精度加法 --- *)
fun addStep (carry, (digitA, result, [])) = 
    let val sum = digitA + carry
    in (sum div 10, (sum mod 10)::result, []) end
  | addStep (carry, (digitA, result, digitB::restB)) = 
    let val sum = digitA + digitB + carry
    in (sum div 10, (sum mod 10)::result, restB) end;

val (finalCarry, sumResult, _) = List.foldl addStep (0, [], numB) numA;
if finalCarry > 0 then printIntTable(finalCarry::sumResult) 
else printIntTable(sumResult);
print("\n");

(* --- 高精度减法 (假设 numA >= numB) --- *)
fun subtractStep (borrow, (digitB, result, [])) =
    let val diff = 0 - digitB - borrow
    in if diff < 0 then (1, (10 + diff)::result, []) else (0, diff::result, []) end
  | subtractStep (borrow, (digitB, result, digitA::restA)) =
    let val diff = digitA - digitB - borrow
    in if diff < 0 then (1, (10 + diff)::result, restA) else (0, diff::result, restA) end;

val (borrowAfterFold, partialDiff, restA) = List.foldl subtractStep (0, [], numA) numB;

fun finishSubtraction (borrow, result, []) = result
  | finishSubtraction (borrow, result, digitA::rest) =
    let val diff = digitA - borrow
    in if diff < 0 then finishSubtraction(1, (10 + diff)::result, rest)
       else finishSubtraction(0, diff::result, rest)
    end;

val finalDiff = finishSubtraction(borrowAfterFold, partialDiff, restA);

printIntTable (trimLeadingZeros finalDiff);
print("\n");

(* --- 高精度乘法 --- *)
fun multiplyStep ((power, digitFromB), accumulatedProduct) =
    let
        (* 1. 计算部分积: numA * digitFromB *)
        fun multiplyByDigit (digitFromA, (carry, partialProduct)) = 
            let val product = digitFromA * digitFromB + carry
            in (product div 10, (product mod 10)::partialProduct) end;
        
        val (partialCarry, partialProductDigits) = List.foldl multiplyByDigit (0, []) numA;
        val fullPartialProduct = if partialCarry > 0 then partialCarry::partialProductDigits else partialProductDigits;

        (* 2. 补零实现错位, 并反转以进行加法 (最低位在前) *)
        val shiftedPartialProduct = List.rev(fullPartialProduct @ List.tabulate (power, fn _ => 0));
        
        (* 3. 将当前部分积加到累计总和上 (确保总是用长列表驱动fold) *)
        val (listToFold, initialRest) = 
            if List.length shiftedPartialProduct > List.length accumulatedProduct
            then (shiftedPartialProduct, accumulatedProduct)
            else (accumulatedProduct, shiftedPartialProduct);

        val (carryFromSum, sumResult, _) = List.foldl addStep (0, [], initialRest) listToFold;
        val newAccumulatedProduct = if carryFromSum > 0 then carryFromSum::sumResult else sumResult;
    in
        (* 返回反转后的结果 (最低位在前) 以便下次迭代 *)
        List.rev newAccumulatedProduct
    end;

(* 准备 (索引, 位) 对用于迭代, 例如: [(0, d0), (1, d1), ...] *)
val digitIndexPairs = ListPair.zip (List.tabulate (numDigitsB, fn i => i), numB);

(* 使用 foldl 进行迭代乘法和累加 *)
(* 初始累积值为 [0] (最低位在前) *)
val finalProductReversed = List.foldl multiplyStep [0] digitIndexPairs;

(* 最终结果需要反转为正序 (最高位在前) 以便打印 *)
val finalProduct = List.rev finalProductReversed;

printIntTable (trimLeadingZeros finalProduct);
print("\n");

(*****End*****)