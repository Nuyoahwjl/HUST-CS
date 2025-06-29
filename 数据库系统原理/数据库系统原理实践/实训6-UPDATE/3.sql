use finance1;
-- 已知表new_client保存了一批新客户信息，该表与client表结构完全相同。请用一条SQL语句将new_client表的全部客户信息插入到客户表(client):
insert into client 
select * from new_client;

