create database TestDb;
use TestDb;
create table t_emp(
    id int primary key,
    name varchar(32),
    deptId int,
    salary float
);