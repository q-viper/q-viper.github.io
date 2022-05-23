---
title:  "MySQL: Triggers"
date:   2022-05-22 09:29:17 +0545
categories:
    - MySQL
    - Triggers
    
tags:
    - mysql
    - triggers
---

## Triggers in SQL
Triggers in SQL is a way to invoke something as a response to the events on the table in which Trigger is attached. The example of the event can be Insert, Update, Delete. Triggers are of two type, Row Level and Statement Level. The row level trigger is triggered for each row while statement level trigger is triggered once per transaction or execution. 

### Why do we need trigger?

* In Data Engineering or Data Pipelining, to reflect the change of the data without having to listen.
* To perform data validation with the by executing trigger Before inserting data. Examples can be performing integrity checks.
* To handle database layer errors.
* To record the history of the data changes.
* To achieve some kind of table monitoring functionalities.


## Triggers in MySQL
MySQL provides only row level triggers.

### Syntax

```sql
CREATE TRIGGER name_of_trigger
{BEFORE | AFTER} {INSERT | UPDATE| DELETE }
ON table_name FOR EACH ROW
body_of_trigger;
```

Trigger's body can be a single line to multiple and it is enclosed inside `BEGIN` and `END` for multiple line body.

* While using Update, we can access existing value and new value (existing as `Old` and new as `New`)and we can compare between them too. Example: to compare old and new value of a column `age`, we can do `OLD.age` != `NEW.age`.
* While using Insert, we can access new value using `New` keyword.
* While using Delete, we can access old value using `Old` keyword.



### Alert After Insert
Lets insert into logs after inserting the values.

1. First of all, lets create a database, `Student` via MySQL. 

```sql
create database Student;
```

2. Create table, `student_bio`.

```sql
create table Student.student_bio (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        `name` varchar(255),
                        class varchar(255),
                        age float
                        );
```

3. Create table, `student_logs`

```sql
CREATE TABLE Student.student_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_name varchar(255) NOT NULL,
    student_age float NOT NULL,
    created_date DATETIME DEFAULT NULL,
    operation VARCHAR(50) DEFAULT NULL
);

```

4. Create a trigger to log info on logs on inserting.

```sql
CREATE TRIGGER Student.after_student_insert 
    after insert ON Student.student_bio
    FOR EACH ROW 
 INSERT INTO Student.student_logs
 SET operation = 'insert',
     student_name = new.name,
     student_age = new.age,
     created_date = NOW();
```

3. Insert few data into it.

```sql
INSERT into Student.student_bio values(1,'John', 5, 15), (1,'Johnny', 7, 25);
```

4. Now look into `Student.student_logs`


![png](after_insert1.png)



### Alert Before Insert
Lets insert the logs before inserting the values.

1. Define a trigger as:

```sql
delimiter // 
CREATE TRIGGER Student.before_student_insert 
    before insert ON Student.student_bio
    FOR EACH ROW 
 
 begin
 INSERT INTO Student.student_logs (student_name, student_age, created_date, operation) values(new.name, new.age,now(), 'insert_before');
 end
 //
 delimiter ;
```

2. Now insert few data as:

```sql
INSERT into Student.student_bio(`name`, class, age) values('Diwo', 5, 15), ('Ben', 7, 25);
```

3. Now see the data of `student_logs`

![]({{site.url}}/assets/mysql/before_insert1.png)

### Alert Before Update

Lets create a trigger which checks the new value before inserting. If new value is greater than old, then set age as average of them. Else set age as old age. And additionally, insert the logs too.

1. Create a trigger as:
```sql
 delimiter // 
CREATE TRIGGER Student.before_student_update
    before update ON Student.student_bio
    FOR EACH ROW 
 
 begin
if old.age<new.age then set new.age=(old.age+new.age)/2;
	else set new.age=old.age; 
 end if;
 INSERT INTO Student.student_logs (student_name, student_age, created_date, operation) values(old.name, new.age,now(), 'update_before');
 end
 //
 delimiter ;
```
2. Now update `student_bio` as:

```sql
update student.student_bio set age =10 where class=5; 
```

![]({{site.url}}/assets/mysql/update_before1.png)

3. Again, update `student_bio` as:

```sql
update student.student_bio set age =20 where class=5; 
```

![]({{site.url}}/assets/mysql/update_before2.png)


In first update, the condition was False so the age was not changed. But in the second update, the condition is True and thus the age was set to average of two.

### Alert Before Delete

Will be updated sooon....


```python

```

## Drawbacks
Now we knew its benefits and the use cases, lets get into the drawbacks of Triggers:
1. It increases the server overhead and can cause server hang ups.
2. It is difficult to test triggers because they are run by Database itself.
3. Can be used for advanced data validation but simple ones can be achieved by constraints like Unique, Null, Check, foreign key etc.