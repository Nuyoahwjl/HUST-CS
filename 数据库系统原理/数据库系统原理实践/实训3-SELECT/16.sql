-- 16) 查询持有相同基金组合的客户对，如编号为A的客户持有的基金，
--     编号为B的客户也持有，反过来，编号为B的客户持有的基金，编
--     号为A的客户也持有，则(A,B)即为持有相同基金组合的二元组，
--     请列出这样的客户对。为避免过多的重复，如果(1,2)为满足条件
--     的元组，则不必显示(2,1)，即只显示编号小者在前的那一对，
--     这一组客户编号分别命名为c_id1,c_id2。
with pro(c_id, f_id) as (
    select
        pro_c_id as c_id,
        group_concat(distinct pro_pif_id order by pro_pif_id) as f_id 
        -- group_concat函数将同一客户的基金编号连接成一个字符串
    from property
    where pro_type = 3
    group by pro_c_id
)
select
    t1.c_id as c_id1,
    t2.c_id as c_id2
from pro as t1, pro as t2
where t1.c_id < t2.c_id
and t1.f_id = t2.f_id;