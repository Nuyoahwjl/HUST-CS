-- test2_1.sh
-- 这是shell脚本，将在linux命令行上执行
-- 命令行上可省略密码的指定
-- 请写出对数据库train作逻辑备份并新开日志文件的命令，备份文件你可以自己命名(如train_bak.sql)：
mysqldump -h127.0.0.1 -uroot --flush-logs --databases train > train_bak.sql

-- test2_2.sh
-- 这是shell脚本，将在linux命令行上执行
-- 命令行上可省略密码的指定
-- 请写出利用逻辑备份和日志恢复数据库的命令：
mysql -h127.0.0.1 -uroot < train_bak.sql
mysqlbinlog --no-defaults log/binlog.000018 | mysql -h127.0.0.1 -u root