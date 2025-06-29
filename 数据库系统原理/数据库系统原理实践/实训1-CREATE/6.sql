drop database MyDb;
create database MyDb;
use MyDb;
create table s(
    sno char(10) primary key,
    name varchar(32) not null,
    ID char(18) unique
);