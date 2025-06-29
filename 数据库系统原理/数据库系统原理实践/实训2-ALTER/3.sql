-- 将QQ号的数据类型改为char(12);将列名weixin改为wechat。
alter table addressBook modify QQ char(12), rename column weixin to wechat;