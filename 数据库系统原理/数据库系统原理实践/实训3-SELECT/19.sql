-- 19) 以日历表格式列出2022年2月每周每日基金购买总金额，输出格式如下：
-- week_of_trading Monday Tuesday Wednesday Thursday Friday
--               1
--               2    
--               3
--               4
select
    wk as week_of_trading,
    sum(if(dayId = 0, amount, null)) as Monday,
    sum(if(dayId = 1, amount, null)) as Tuesday,
    sum(if(dayId = 2, amount, null)) as Wednesday,
    sum(if(dayId = 3, amount, null)) as Thursday,
    sum(if(dayId = 4, amount, null)) as Friday
from (
    select
        week(pro_purchase_time) - 5 as wk,
        weekday(pro_purchase_time) as dayId,
        sum(pro_quantity * f_amount) as amount
    from property, fund
    where
        pro_type = 3
        and pro_pif_id = f_id
        and pro_purchase_time like "2022-02-%"
    group by pro_purchase_time
) as t
group by wk;