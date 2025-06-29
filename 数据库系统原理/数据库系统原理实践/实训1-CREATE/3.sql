drop database MyDb;
create database MyDb;
use MyDb;
create table dept(
    deptNo int primary key,
    deptName varchar(32)
);
create table staff(
    staffNo int primary key,
    staffName varchar(32),
    gender char(1),
    dob date,
    salary numeric(8, 2),
    deptNo int,
    constraint FK_staff_deptNo foreign key(deptNo) references dept(deptNo)
);