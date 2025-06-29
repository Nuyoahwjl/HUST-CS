-- 3) 查询购买了所有畅销理财产品的客户
select distinct pro_c_id
from property as p1
where not exists(
   select *
   from (
      select pro_pif_id
      from property
      where pro_type = 1
      group by pro_pif_id
      having count(*) > 2
   ) as t1
   where t1.pro_pif_id not in (
      select pro_pif_id
      from property as p2
      where p1.pro_c_id = p2.pro_c_id
      and p2.pro_type = 1
   )
);