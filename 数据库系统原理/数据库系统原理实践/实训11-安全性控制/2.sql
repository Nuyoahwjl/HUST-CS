-- 请填写语句，完成以下功能：
-- (1) 创建角色client_manager和fund_manager；
drop role if exists client_manager, fund_manager;
create role client_manager, fund_manager;
-- (2) 授予client_manager对client表拥有select,insert,update的权限；
grant select, insert, update
on table client
to client_manager;
-- (3) 授予client_manager对bank_card表拥有查询除银行卡余额外的select权限；
grant select(b_c_id, b_number, b_type)
on table bank_card
to client_manager;
-- (4) 授予fund_manager对fund表的select,insert,update权限；
grant select, insert, update
on table fund
to fund_manager;
-- (5) 将client_manager的权限授予用户tom和jerry；
grant client_manager
to tom, jerry;
-- (6) 将fund_manager权限授予用户Cindy.
grant fund_manager
to Cindy;