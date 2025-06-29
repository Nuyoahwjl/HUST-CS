-- 2) 投资积极且偏好理财类产品的客户select t1.pro_c_id 
select t1.pro_c_id 
from (
(
  select pro_c_id, count(distinct(pro_pif_id)) as cnt1
  from property, finances_product
  where pro_type = 1 
  and pro_pif_id = p_id 
  group by pro_c_id
) as t1,
(
  select pro_c_id, count(distinct(pro_pif_id)) as cnt2
  from property, fund
  where pro_type = 3
  and pro_pif_id = f_id 
  group by pro_c_id
) as t2
)
where t1.pro_c_id = t2.pro_c_id
and t1.cnt1 > t2.cnt2
order by t1.pro_c_id;

