use finance1;
-- 已知33号客户部分信息如下:
-- c_id(编号):33
-- c_name(名称):蔡依婷
-- c_phone(电话):18820762130
-- c_id_card(身份证号):350972199204227621
-- c_password(密码):MKwEuc1sc6

-- 请用一条SQL语句将这名客户的信息插入到客户表(client)：
insert nto client (c_id, c_name, c_phone, c_id_card, c_password) 
values(33, "蔡依婷", "18820762130", "350972199204227621", "MKwEuc1sc6");