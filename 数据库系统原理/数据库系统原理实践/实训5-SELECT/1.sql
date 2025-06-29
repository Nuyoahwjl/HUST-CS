-- 本关任务：将客户年度从各单位获得的酬劳进行排序
-- 综合客户表(client)、薪资表(wage)，列出客户的名称、
-- 年份、身份证号、全职酬劳总金额（full_t_amount）、兼职酬劳总金额（part_t_amount）。
-- 查询结果按全职和兼职总金额降序排序。数据确保不存在全职和兼职总金额相等的客户。
-- 提示：确保只统计与有效客户关联的记录(客户表可能存在悬浮元组)
-- 输出表结构为：c_name, year, c_id_card, full_t_amount，part_t_amount

select c_name, year, c_id_card, full_t_amount, part_t_amount
from(
select c_name, extract(year from w_time) as year, c_id_card,
       sum(case when w_type = 1 then w_amount else 0 end) as full_t_amount,
       sum(case when w_type = 2 then w_amount else 0 end) as part_t_amount,
       sum(w_amount) as amount
from client, wage
where c_id = w_c_id
group by c_id, year
order by amount desc
) as temp;