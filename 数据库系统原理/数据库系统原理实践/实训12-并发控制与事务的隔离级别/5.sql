-- 事务1:
use testdb1;
set session transaction isolation level read uncommitted;
start transaction;
-- # 第1次查询航班'MU2455'的余票
select tickets from ticket where flight_no = "MU2455" for update;
set @n = sleep(5);
-- # 第2次查询航班'MU2455'的余票
select tickets from ticket where flight_no = "MU2455";
commit;
-- 第3次查询所有航班的余票，发生在事务2提交后
set @n = sleep(1);
select * from ticket;

-- 事务2:
use testdb1;
set session transaction isolation level read uncommitted;
start transaction;
set @n = sleep(1);
-- # 在事务1的第1，2次查询之间，试图出票1张(航班MU2455)：
update ticket set tickets = tickets-1 where flight_no = "MU2455";
commit;