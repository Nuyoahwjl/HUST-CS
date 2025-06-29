-- (1) 为表Staff添加主码
alter table Staff add primary key (staffNo);
-- (2) Dept.mgrStaffNo是外码，对应的主码是Staff.staffNo,请添加这个外码，名字为FK_Dept_mgrStaffNo:
alter table Dept add constraint FK_Dept_mgrStaffNo foreign key(mgrStaffNo) references Staff(staffNo);
-- (3) Staff.dept是外码，对应的主码是Dept.deptNo. 请添加这个外码，名字为FK_Staff_dept:
alter table Staff add constraint FK_Staff_dept foreign key(dept) references Dept(deptNo);
-- (4) 为表Staff添加check约束，规则为：gender的值只能为F或M；约束名为CK_Staff_gender:
alter table Staff add constraint CK_Staff_gender check (gender in ('F', 'M'));
-- (5) 为表Dept添加unique约束：deptName不允许重复。约束名为UN_Dept_deptName：
alter table Dept add constraint UN_Dept_deptName unique(deptName);