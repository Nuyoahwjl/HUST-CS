-- 查询发放兼职酬劳前三个单位名字，及发放兼职酬劳，
-- 按酬劳降序输出，总酬劳命名为total_salary。
-- 不需要考虑并列排名情形;
-- 并确保只统计与有效客户关联的记录(客户表可能存在悬浮元组)
-- 输出格式：w_org, total_salary

select w.w_org, sum(w.w_amount) as total_salary
from wage w
inner join client c on w.w_c_id = c.c_id
where w.w_type = 2
group by w.w_org
order by total_salary desc
limit 3;