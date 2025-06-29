-- 7) 查询身份证隶属武汉市没有买过任何理财产品的客户的名称、电话号、邮箱。
select c_name, c_phone, c_mail
from client
where c_id_card like "4201%"
and not exists (
    select *
    from property
    where pro_c_id = c_id
    and pro_type = 1
)
order by c_id;
