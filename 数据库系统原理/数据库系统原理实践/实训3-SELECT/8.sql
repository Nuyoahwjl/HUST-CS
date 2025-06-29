-- 8) 查询持有两张(含）以上信用卡的用户的名称、身份证号、手机号。
select c_name, c_id_card, c_phone
from client
where (c_id, '信用卡') in (
    select b_c_id, b_type
    from bank_card
    group by b_c_id, b_type
    having count(*) >= 2
);


select c_name, c_id_card, c_phone
from client
where c_id in (
    select b_c_id
    from bank_card
    where b_type = '信用卡'
    group by b_c_id
    having count(*) >= 2
);