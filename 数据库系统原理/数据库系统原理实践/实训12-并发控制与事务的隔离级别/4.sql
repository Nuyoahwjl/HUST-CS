-- 事务1（采用默认的事务隔离级别- repeatable read）:
use testdb1;
select @@transaction_isolation;
start transaction;
## 第1次查询余票超过300张的航班信息
select * from ticket where tickets > 300;
set @n = sleep(2);
-- 修改航班MU5111的执飞机型为A330-300：
update ticket set aircraft = 'A330-300' where flight_no = 'MU5111'; 
-- 第2次查询余票超过300张的航班信息
select * from ticket where tickets > 300;
commit;