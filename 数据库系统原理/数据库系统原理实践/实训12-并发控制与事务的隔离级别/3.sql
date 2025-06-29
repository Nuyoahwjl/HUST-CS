-- 事务1:
-- ## 请设置适当的事务隔离级别
set session transaction isolation level read uncommitted;
-- 开启事务
start transaction;
-- 时刻1 - 事务1读航班余票:
insert into result 
select now(),1 t, tickets from ticket where flight_no = 'CZ5525';
-- ## 添加等待代码，确保事务2的第一次读取在事务1修改前发生
set @n = sleep(2);
-- 时刻3 - 事务1修改余票，并立即读取:
update ticket set tickets = tickets - 1 where flight_no = 'CZ5525';
insert into result
select now(),1 t, tickets from ticket where flight_no = 'CZ5525';
-- ## 添加代码，使事务2 的第2次读取在事务1修改之后，提交之前发生
set @n = sleep(3);
commit;
-- 时刻6 - 事务1在t2也提交后读取余票
-- ## 添加代码，确保事务1在事务2提交后读取
set @n = sleep(2);
insert into result
select now(), 1 t, tickets from ticket where flight_no = 'CZ5525';

-- 事务2
-- ## 请设置适当的事务隔离级别以构造不可重复读
set session transaction isolation level read uncommitted;
start transaction;
-- 时刻2 - 事务2在事务1读取余票之后也读取余票
-- ## 添加代码，确保事务2的第1次读发生在事务1读之后，修改之前
set @n = sleep(1);
insert into result 
select now(),2 t, tickets from ticket where flight_no = 'CZ5525';
-- 时刻4 - 事务2在事务1修改余票但未提交前再次读取余票，事务2的两次读取结果应该不同
-- ## 添加代码，确保事务2的读取时机
set @n = sleep(3);
insert into result 
select now(), 2 t, tickets from ticket where flight_no = 'CZ5525';
-- 事务2立即修改余票
update ticket set tickets = tickets - 1 where flight_no = 'CZ5525';
-- 时刻5 - 事务2 读取余票（自己修改但未交的结果）:
-- set @n = sleep(1);
insert into result 
select now(), 2 t, tickets from ticket where flight_no = 'CZ5525';
commit;
