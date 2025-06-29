select pro_pif_id, count(*) as cc, dense_rank() over(order by count(*) desc) as prank
from property
where 
    pro_type = 1 and
    pro_pif_id in (
        select distinct pro_pif_id
        from property 
        where
            pro_type = 1 and
            pro_pif_id <> 14 and
            pro_c_id in (
                select pro_c_id
                from (
                    select pro_c_id, dense_rank() over(order by pro_quantity) as rk
                    from property
                    where 
                        pro_type = 1 and
                        pro_pif_id = 14
                ) as fin_rk
            where fin_rk.rk <= 3))
group by pro_pif_id
order by cc desc, pro_pif_id;