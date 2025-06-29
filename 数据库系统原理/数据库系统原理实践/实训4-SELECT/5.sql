-- 请用一条SQL语句完成以下查询任务：
-- 查询任意两个客户之间持有的相同理财产品种数，
-- 并且结果仅保留相同理财产品数至少2种的用户对。
-- 注意结果输出要求：第一列和第二列输出客户编号(pro_c_id,pro_c_id)，
-- 第三列输出他们持有的相同理财产品数(total_count)，按照第一列的客户编号的升序排列。

select a.pro_c_id, b.pro_c_id,count(*) as total_count
from property a, property b
where a.pro_c_id <> b.pro_c_id
    and a.pro_type = 1
    and b.pro_type = 1
    and a.pro_pif_id = b.pro_pif_id
group by a.pro_c_id, b.pro_c_id
having count(*) >= 2
order by a.pro_c_id;