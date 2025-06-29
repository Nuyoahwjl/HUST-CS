drop database MyDb;
create database MyDb;
use MyDb;
create table hr(
    id char(10) primary key,
    name varchar(32) not null,
    mz char(16) default '汉族'
);