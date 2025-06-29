 -- 基于上一关创建的视图v_insurance_detail进行分组统计查询，列出每位客户的姓名，身份证号，保险投资总额(insurance_total_amount)和保险投资总收益(insurance_total_revenue),结果依保险投资总额降序排列。
--  请用一条SQL语句实现该查询：
select c_name, c_id_card, sum(i_amount*pro_quantity) as insurance_total_amount, sum(pro_income) as insurance_total_revenue
from v_insurance_detail
group by c_name,c_id_card
order by insurance_total_amount DESC;