-- 查询当前兼职总酬劳前三名的客户的姓名、身份证号及其总酬劳，
-- 按酬劳降序输出，总酬劳命名为total_salary。不需要考虑并列排名情形。
-- 1)本题目不需要考虑并列排名情形，这意味着前5名为16000，12900，
-- 12900，8000，8000时，也只需要给出16000，12900，12900即可。
-- 2)确保只统计与有效客户关联的记录(客户表可能存在悬浮元组)
-- 输出表结构为：c_name, c_id_card, total_salary

select c_name, c_id_card, total_salary
from(
select c_name, c_id_card, sum(w_amount) as total_salary
from client, wage
where c_id = w_c_id and w_type = 2
group by c_id
order by total_salary desc, c_id 
) as temp limit 3;