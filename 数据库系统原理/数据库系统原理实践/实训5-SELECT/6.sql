update wage
join (
    select 
        w_c_id,
        sum(w_amount) as total_salary,
        greatest(sum(w_amount) - 60000, 0) * 0.2 as total_tax
    from wage
    where w_c_id = (select c_id from client where c_id_card = '420108199702144323')
      and year(w_time) = 2023
    group by w_c_id
) as t using (w_c_id)
set 
    wage.w_amount = wage.w_amount - (t.total_tax * (wage.w_amount / nullif(t.total_salary, 0))), -- 修复括号闭合
    wage.w_tax = if(t.total_tax > 0, 'y', 'n')
where 
    year(wage.w_time) = 2023;