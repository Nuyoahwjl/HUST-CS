-- 17) 查询2022年2月购买基金的高峰期。至少连续三个交易日，
--     所有投资者购买基金的总金额超过100万(含)，则称这段
--     连续交易日为投资者购买基金的高峰期。只有交易日才能
--     购买基金,但不能保证每个交易日都有投资者购买基金。
--     2022年春节假期之后的第1个交易日为2月7日,周六和周日
--     是非交易日，其余均为交易日。请列出高峰时段的日期和
--     当日基金的总购买金额，按日期顺序排序。总购买金额命名为total_amount。
select
    t3.t as pro_purchase_time,
    t3.amount as total_amount
from (
    select 
        *,
        count(*) over(partition by t2.workday - t2.rownum) cnt
    from (
        select
            *,
            row_number() over(order by workday) rownum
        from (
            select
                pro_purchase_time t,
                sum(pro_quantity * f_amount) amount,
                @row := datediff(pro_purchase_time, "2021-12-31") - 2 * week(pro_purchase_time) workday
            from property, fund, (select @row) a
            where pro_purchase_time like "2022-02-%"
            and pro_type = 3
            and pro_pif_id = f_id
            group by pro_purchase_time
        ) as t1
        where amount > 1000000
    ) as t2
) as t3
where t3.cnt >= 3;





/*  end  of  your code  */