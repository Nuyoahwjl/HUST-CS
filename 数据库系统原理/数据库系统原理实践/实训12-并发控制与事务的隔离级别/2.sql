-- 事务1:
use testdb1;
-- ## 请设置适当的事务隔离级别
set session transaction isolation level read uncommitted;
start transaction;
-- 时刻2 - 事务1读航班余票,发生在事务2修改之后
-- ## 添加等待代码，确保读脏
set @n = sleep(1);
select tickets from ticket where flight_no = 'CA8213';
commit;

-- 事务2
use testdb1;
-- ## 请设置适当的事务隔离级别
set session transaction isolation level read uncommitted;
start transaction;
-- 时刻1 - 事务2修改航班余票
update ticket set tickets = tickets - 1 where flight_no = 'CA8213';
-- 时刻3 - 事务2 取消本次修改
-- ## 请添加代码，使事务1在事务2撤销前读脏;
set @n = sleep(2);
rollback;