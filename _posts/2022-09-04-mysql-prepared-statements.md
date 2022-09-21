---
title:  "MySQL Prepared Statements"
date:   2022-09-04 01:29:17 +0545
categories:
    - MySQL
tags:
    - MySQL
header:
  teaser: assets/mysql/p2.png
---

MySQL Prepared Statements are the queries or statements they are prepared in a way that they can be used later on. They are complied while on creation and can be executed later as desired. We might need to run prepared statements many times once its created. Lets get little bit deep into it and how can we used MySQL Prepared Statements to make SQLing easily.

## What Do We need?
We only need mysql server but for the ease, I am going to use MySQL Workbench because of its nice GUI.

## Why do we need Prepared Statements?

### To make dynamic queries.

Often SQL queries are predetermined and we only have to work with whats known to use only. But when time comes for us to run some query that we do not know yet, and the query itself is not static, we might have to generate a dynamic query and run.

For this blog, I am going to use new database `School`. Lets create a table student inside a school db.

```sql  
CREATE TABLE school.student(name VARCHAR(255), age INT, gender VARCHAR(255));
```

Now lets insert some entries in it.

```sql
use school;
INSERT INTO student VALUES ('John', 14, 'male');
INSERT INTO student VALUES ('Jean', 11, 'female');
INSERT INTO student VALUES ('Sandra', 17, 'female');
```

Now lets create a prepared statement which will show us only students with gender as 'female'.

```sql
PREPARE ptmt FROM 'select * from student  where  gender =?;';
set @gen = "female";
execute ptmt using @gen;
```

And result is:

![]({{site.url}}/assets/mysql/p1.png)

Here, we have just run the statement with dynamic value in where clause. But Can we run dynamic query with table name and column name too? Answer is yes but we have to use `CONCAT` for this too.

```sql
set @gen = "female";
set @tb = "student";
set @stmt = concat("select * from ", @tb, " where gender = ?;");

PREPARE ptmt FROM @stmt;
execute ptmt using @gen;
```

![]({{site.url}}/assets/mysql/p2.png)

Here we just created a dynamic statement and used it too. Below is another example of it.

```sql
set @gen = "male";
set @tb = "student";
set @col = 'age';
set @stmt = concat(concat("select ", @col), " from ", @tb, " where gender = ?;");

PREPARE ptmt FROM @stmt;
execute ptmt using @gen;
```

![]({{site.url}}/assets/mysql/p3.png)


Using multiple concats, we can make even more complex statements and run them as prepared statements.

### To save our time from writing redundant queries

One of the main reason of using MySQL Prepared Statements is to save time from running redundant queries. We might have to run queries with similar nature and mostly we search the term and replace them by our new keyword. Like, we want to search for the number of customers from Sydney with age 35 and below. Queries like above can be run without any hassle but what if we want to view different columns too. We want to view columns like number of max subscription per month of customer from Sydney and number of least subscription per month of customer from California. We have to modify the query itself. But there is an easier way to do that. But first, lets add another table to our example.

```sql
CREATE TABLE school.teacher(name VARCHAR(255), age INT, gender VARCHAR(255), location varchar(255));

INSERT INTO teacher VALUES ('Harvey', 43, 'male', "Dubai");
INSERT INTO teacher VALUES ('Joanna', 51, 'female', "California");
INSERT INTO teacher VALUES ('Harris', 37, 'male', "Sydney");
INSERT INTO teacher VALUES ('Holly', 43, 'female', "Dubai");
INSERT INTO teacher VALUES ('Mark', 51, 'male', "California");
INSERT INTO teacher VALUES ('Henry', 37, 'male', "Sydney");
```

Now lets search for the average age  of teachers from Dubai. And number of teachers from Sydney. But using single type of statement.

First Statement:

```sql
set @place = "Dubai";
set @tb = "teacher";
set @col = 'avg(age)';
set @stmt = concat(concat("select ", @col), " from ", @tb, " where location = ?;");

PREPARE ptmt FROM @stmt;
execute ptmt using @place;
```

Second statement:

```sql
set @place = "Sydney";
set @tb = "teacher";
set @col = 'count(name)';
set @stmt = concat(concat("select ", @col), " from ", @tb, " where location = ?;");

PREPARE ptmt FROM @stmt;
execute ptmt using @place;
```


### Can wrap statements
With the help of Stored Procedure, MySQL Prepared Statements can be used to wrap statements too.

```sql
drop procedure if exists query_runner;

delimiter //
create procedure query_runner(In tb varchar(255), 
							IN scol varchar(255), 
                            IN ocol varchar(255),
                            IN op varchar(255),
                            IN oval varchar(255)
                            ) 
begin
set @oval = oval;
set @stmt = concat ("select ", scol, " from ", tb);
set @stmt = concat(concat(@stmt, " where "), ocol);
set @stmt = concat(@stmt, op, "?");

prepare pstmt from @stmt;
execute pstmt using @oval;

end //
delimiter ; 

call query_runner("teacher", "*", "age", "<",50);
```

Result is:
![]({{site.url}}/assets/mysql/p4.png)


MySQL Prepared Statements has better application when used along with Stored Procedure as SP works like a function. We can achieve a lot of benefits using MySQL Prepared Statements and Stored Procedures. But that will be covered in next part.


```python

```
