-- 请不要在本代码文件中添加空行！！！ 
use testdb1;
# 设置事务的隔离级别为 read uncommitted
set session transaction isolation level read uncommitted;
-- 开启事务
start transaction;
insert into dept(name) values('运维部');
# 回滚事务：
rollback;
/* 结束 */