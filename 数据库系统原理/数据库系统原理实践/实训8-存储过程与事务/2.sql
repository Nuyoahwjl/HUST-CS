-- 编写一存储过程，自动安排某个连续期间的大夜班的值班表:

delimiter $$
create procedure sp_night_shift_arrange(in start_date date, in end_date date)
begin
declare done, waitdir int default false;
declare nowdate date;
declare waitdr, dr, nr1, nr2 char(30);
declare drtype int;
declare drlist cursor for select e_name, e_type from employee where e_type < 3;
declare nrlist cursor for select e_name from employee where e_type = 3;
declare continue handler for not found set done = true;

open drlist;
open nrlist;
set nowdate = start_date;
while nowdate <= end_date do
    if weekday(nowdate) < 5 and waitdir then
        set dr = waitdr, waitdir = false;
    else
        fetch drlist into dr, drtype;
        if done then
            close drlist;
            open drlist;
            fetch drlist into dr, drtype;
            set done = false;
        end if;
        if weekday(nowdate) >= 5 and drtype = 1 then
            set waitdir = true, waitdr = dr;
            fetch drlist into dr, drtype;
            if done then
                close drlist;
                open drlist;
                fetch drlist into dr, drtype;
                set done = false;
            end if;
        end if;
    end if;

    fetch nrlist into nr1;
    if done then 
        close nrlist;
        open nrlist;
        fetch nrlist into nr1;
        set done = false;
    end if;

    fetch nrlist into nr2;
    if done then 
        close nrlist;
        open nrlist;
        fetch nrlist into nr2;
        set done = false;
    end if;
    insert into night_shift_schedule values (nowdate, dr, nr1, nr2);
    set nowdate = date_add(nowdate, interval 1 day);
end while;
end$$

delimiter ;

/*  end  of  your code  */ 