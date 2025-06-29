-- 6) 查询资产表中所有资产记录里商品收益的众数和它出现的次数。
select pro_income, count(*) as presence
from property
group by pro_income
having count(*) >= all (
    select count(*)
    from property 
    group by pro_income
); 