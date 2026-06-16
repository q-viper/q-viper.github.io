---
layout: single
title: "MySQL Prepared Statements: Beginner Guide with Dynamic Queries and Stored Procedures"
date: 2022-09-04 01:29:17 +0545
last_modified_at: 2026-06-16
categories:
  - MySQL
  - SQL
  - Database
tags:
  - "MySQL"
  - "Prepared statements"
  - "SQL"
  - "Dynamic SQL"
  - "Stored procedures"
  - "SQL injection"
  - "Database tutorial"
description: "A beginner-friendly guide to MySQL prepared statements, including parameterized queries, dynamic SQL, PREPARE, EXECUTE, DEALLOCATE PREPARE, and stored procedure examples."
excerpt: "Learn how MySQL prepared statements work, why they are useful, how to use dynamic query values safely, and how to combine prepared statements with stored procedures."
header:
  teaser: assets/mysql/p2.png
  og_image: assets/mysql/p2.png
  image_description: "MySQL prepared statement result shown in MySQL Workbench"
toc: true
toc_label: "MySQL Prepared Statements"
toc_icon: "database"
toc_sticky: true
read_time: true
share: true
related: true
classes: wide
---

**MySQL prepared statements** are SQL statements that are written once, prepared by MySQL, and executed later with different values. They are useful when we need to run the same kind of query multiple times or when parts of the query need to be dynamic.

Prepared statements are commonly used for:

- parameterized queries
- repeated SQL execution
- dynamic filtering
- dynamic SQL in stored procedures
- reducing repeated query writing
- safer handling of user-provided values

In this post, we will learn how prepared statements work in MySQL using simple examples with a `school` database.

## What Is a MySQL Prepared Statement?

A prepared statement is a SQL statement that is prepared first and executed later.

The basic flow is:

```sql
PREPARE statement_name FROM @sql_query;
EXECUTE statement_name USING @value;
DEALLOCATE PREPARE statement_name;
```

A prepared statement can contain placeholders. In MySQL, the placeholder is written as `?`.

Example:

```sql
PREPARE pstmt FROM 'SELECT * FROM student WHERE gender = ?';

SET @gen = 'female';

EXECUTE pstmt USING @gen;

DEALLOCATE PREPARE pstmt;
```

Here, `?` is replaced by the value stored in `@gen` during execution.

## Why Use Prepared Statements?

Prepared statements are useful for several reasons.

### 1. Reuse Similar Queries

If we need to run the same query many times with different values, prepared statements save time.

For example:

```sql
SELECT * FROM student WHERE gender = 'female';
SELECT * FROM student WHERE gender = 'male';
```

Instead of writing the query again and again, we can prepare it once and only change the value.

### 2. Work with Dynamic Values

Prepared statements make it easy to pass different values into a query.

```sql
SET @gen = 'female';
EXECUTE pstmt USING @gen;

SET @gen = 'male';
EXECUTE pstmt USING @gen;
```

### 3. Use Dynamic SQL

Sometimes the table name, column name, or operation may change. In such cases, we can build a SQL string using `CONCAT()` and then prepare it.

### 4. Improve Safety for Values

Prepared statements help separate SQL logic from data values. This helps reduce SQL injection risk when values are passed as parameters.

However, there is an important warning:

> Prepared statement placeholders can be used for values, but not directly for table names or column names.

For dynamic table names or column names, we usually build the SQL string manually. In that case, we must validate or whitelist the table and column names carefully.

## What Do We Need?

For this tutorial, we need:

- MySQL Server
- MySQL Workbench or any SQL client

I will use MySQL Workbench because it has a simple GUI and makes it easy to view query results.

## Create a School Database

Let's create a small database named `school`.

```sql
CREATE DATABASE IF NOT EXISTS school;

USE school;
```

Now create a `student` table.

```sql
CREATE TABLE IF NOT EXISTS student (
    name VARCHAR(255),
    age INT,
    gender VARCHAR(255)
);
```

Insert some sample data.

```sql
INSERT INTO student VALUES ('John', 14, 'male');
INSERT INTO student VALUES ('Jean', 11, 'female');
INSERT INTO student VALUES ('Sandra', 17, 'female');
```

Now we have a simple table to test prepared statements.

## Basic Prepared Statement Example

Let's create a prepared statement to show only students with a given gender.

```sql
PREPARE pstmt FROM 'SELECT * FROM student WHERE gender = ?';

SET @gen = 'female';

EXECUTE pstmt USING @gen;

DEALLOCATE PREPARE pstmt;
```

Result:

![]({{site.url}}/assets/mysql/p1.png)

Here, the SQL query is prepared first. Then the value `female` is passed during execution.

## Execute the Same Statement with a Different Value

One benefit of prepared statements is that the same statement can be executed with different values.

```sql
PREPARE pstmt FROM 'SELECT * FROM student WHERE gender = ?';

SET @gen = 'female';
EXECUTE pstmt USING @gen;

SET @gen = 'male';
EXECUTE pstmt USING @gen;

DEALLOCATE PREPARE pstmt;
```

This avoids rewriting the full query.

## Dynamic Query with Table Name

Can we make the table name dynamic too?

Yes, but not with `?`.

This will not work:

```sql
PREPARE pstmt FROM 'SELECT * FROM ? WHERE gender = ?';
```

Placeholders are for values, not identifiers such as table names and column names.

To make a table name dynamic, we need to build the SQL string with `CONCAT()`.

```sql
SET @gen = 'female';
SET @tb = 'student';

SET @stmt = CONCAT('SELECT * FROM ', @tb, ' WHERE gender = ?;');

PREPARE pstmt FROM @stmt;
EXECUTE pstmt USING @gen;

DEALLOCATE PREPARE pstmt;
```

Result:

![]({{site.url}}/assets/mysql/p2.png)

This works because we created the SQL query string first and then prepared it.

## Important Safety Note for Dynamic Table Names

When building SQL with dynamic table names or column names, be careful.

Do not directly use user input like this:

```sql
SET @tb = user_input;
```

This can be dangerous if the value is not trusted.

A safer approach is to allow only known table names.

For example:

```sql
SET @tb = 'student';

-- In real applications, validate @tb against allowed table names.
```

If the table name comes from an application, validate it in the application code before sending it to MySQL.

## Dynamic Query with Column Name

We can also make the selected column dynamic.

```sql
SET @gen = 'male';
SET @tb = 'student';
SET @col = 'age';

SET @stmt = CONCAT('SELECT ', @col, ' FROM ', @tb, ' WHERE gender = ?;');

PREPARE pstmt FROM @stmt;
EXECUTE pstmt USING @gen;

DEALLOCATE PREPARE pstmt;
```

Result:

![]({{site.url}}/assets/mysql/p3.png)

Here, the column name is dynamic and the gender value is passed as a prepared parameter.

Again, the column name should be validated or selected from a whitelist.

## Better Formatting for Dynamic SQL

When queries become longer, using multiple `CONCAT()` calls can become hard to read.

This is easier:

```sql
SET @gen = 'male';
SET @tb = 'student';
SET @col = 'age';

SET @stmt = CONCAT(
    'SELECT ',
    @col,
    ' FROM ',
    @tb,
    ' WHERE gender = ?;'
);

PREPARE pstmt FROM @stmt;
EXECUTE pstmt USING @gen;

DEALLOCATE PREPARE pstmt;
```

This makes the query-building logic clearer.

## Prepared Statements with Multiple Parameters

A prepared statement can use more than one placeholder.

Example:

```sql
PREPARE pstmt FROM 'SELECT * FROM student WHERE gender = ? AND age > ?';

SET @gen = 'female';
SET @age = 12;

EXECUTE pstmt USING @gen, @age;

DEALLOCATE PREPARE pstmt;
```

This finds female students older than 12.

## Add a Teacher Table

Now let's add another table named `teacher`.

```sql
CREATE TABLE IF NOT EXISTS teacher (
    name VARCHAR(255),
    age INT,
    gender VARCHAR(255),
    location VARCHAR(255)
);
```

Insert sample data.

```sql
INSERT INTO teacher VALUES ('Harvey', 43, 'male', 'Dubai');
INSERT INTO teacher VALUES ('Joanna', 51, 'female', 'California');
INSERT INTO teacher VALUES ('Harris', 37, 'male', 'Sydney');
INSERT INTO teacher VALUES ('Holly', 43, 'female', 'Dubai');
INSERT INTO teacher VALUES ('Mark', 51, 'male', 'California');
INSERT INTO teacher VALUES ('Henry', 37, 'male', 'Sydney');
```

Now we can test prepared statements on a second table.

## Reusing Dynamic Query Structure

Suppose we want to find the average age of teachers from Dubai.

```sql
SET @place = 'Dubai';
SET @tb = 'teacher';
SET @col = 'AVG(age)';

SET @stmt = CONCAT(
    'SELECT ',
    @col,
    ' FROM ',
    @tb,
    ' WHERE location = ?;'
);

PREPARE pstmt FROM @stmt;
EXECUTE pstmt USING @place;

DEALLOCATE PREPARE pstmt;
```

Now suppose we want to count teachers from Sydney.

```sql
SET @place = 'Sydney';
SET @tb = 'teacher';
SET @col = 'COUNT(name)';

SET @stmt = CONCAT(
    'SELECT ',
    @col,
    ' FROM ',
    @tb,
    ' WHERE location = ?;'
);

PREPARE pstmt FROM @stmt;
EXECUTE pstmt USING @place;

DEALLOCATE PREPARE pstmt;
```

The query shape is the same, but the selected expression and location are different.

This is one reason prepared statements are useful.

## Prepared Statements and Stored Procedures

Prepared statements become even more powerful when combined with stored procedures.

A stored procedure works like a database function. We can pass arguments to it and run SQL logic inside it.

Let's create a procedure that builds and runs a dynamic query.

```sql
DROP PROCEDURE IF EXISTS query_runner;

DELIMITER //

CREATE PROCEDURE query_runner(
    IN tb VARCHAR(255),
    IN scol VARCHAR(255),
    IN ocol VARCHAR(255),
    IN op VARCHAR(255),
    IN oval VARCHAR(255)
)
BEGIN
    SET @oval = oval;

    SET @stmt = CONCAT('SELECT ', scol, ' FROM ', tb);
    SET @stmt = CONCAT(@stmt, ' WHERE ', ocol);
    SET @stmt = CONCAT(@stmt, ' ', op, ' ?');

    PREPARE pstmt FROM @stmt;
    EXECUTE pstmt USING @oval;
    DEALLOCATE PREPARE pstmt;
END //

DELIMITER ;
```

Now call the procedure.

```sql
CALL query_runner('teacher', '*', 'age', '<', '50');
```

Result:

![]({{site.url}}/assets/mysql/p4.png)

This procedure lets us dynamically choose:

- table name
- selected columns
- condition column
- operator
- comparison value

## Improve Safety in the Stored Procedure

The above stored procedure is useful for learning, but it is not fully safe for production because it accepts table names, column names, and operators directly.

For production, we should validate inputs.

For example, we can restrict operators:

```sql
IF op NOT IN ('=', '<', '>', '<=', '>=', '<>') THEN
    SIGNAL SQLSTATE '45000'
    SET MESSAGE_TEXT = 'Invalid operator';
END IF;
```

For table and column names, it is often better to validate them in application code. Another option is to check against known allowed values inside the stored procedure.

## Example with Operator Validation

```sql
DROP PROCEDURE IF EXISTS safe_query_runner;

DELIMITER //

CREATE PROCEDURE safe_query_runner(
    IN tb VARCHAR(255),
    IN scol VARCHAR(255),
    IN ocol VARCHAR(255),
    IN op VARCHAR(10),
    IN oval VARCHAR(255)
)
BEGIN
    IF op NOT IN ('=', '<', '>', '<=', '>=', '<>') THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Invalid operator';
    END IF;

    SET @oval = oval;

    SET @stmt = CONCAT('SELECT ', scol, ' FROM ', tb);
    SET @stmt = CONCAT(@stmt, ' WHERE ', ocol, ' ', op, ' ?');

    PREPARE pstmt FROM @stmt;
    EXECUTE pstmt USING @oval;
    DEALLOCATE PREPARE pstmt;
END //

DELIMITER ;
```

Call it:

```sql
CALL safe_query_runner('teacher', '*', 'age', '<', '50');
```

This is still a simple example, but it shows the idea of validating dynamic parts.

## PREPARE, EXECUTE, and DEALLOCATE

MySQL prepared statements usually use three main commands.

### PREPARE

`PREPARE` creates a prepared statement.

```sql
PREPARE pstmt FROM @stmt;
```

### EXECUTE

`EXECUTE` runs the prepared statement.

```sql
EXECUTE pstmt USING @value;
```

### DEALLOCATE PREPARE

`DEALLOCATE PREPARE` removes the prepared statement.

```sql
DEALLOCATE PREPARE pstmt;
```

It is a good habit to deallocate prepared statements after using them.

## Prepared Statements vs Normal SQL Queries

| Feature | Normal Query | Prepared Statement |
|---|---|---|
| Query is written once | No | Yes |
| Can reuse with different values | Manual changes needed | Yes |
| Supports placeholders | No | Yes |
| Useful for dynamic values | Limited | Yes |
| Useful in stored procedures | Yes | Yes |
| Safer value handling | Depends | Better |

## Common Mistakes

Here are some common mistakes when using MySQL prepared statements.

### Mistake 1: Using ? for Table Names

This does not work:

```sql
PREPARE pstmt FROM 'SELECT * FROM ? WHERE gender = ?';
```

The `?` placeholder is for values, not table names.

### Mistake 2: Not Deallocating Prepared Statements

Always clean up when done.

```sql
DEALLOCATE PREPARE pstmt;
```

### Mistake 3: Trusting Dynamic Table or Column Names

Values can be passed safely with placeholders, but table names and column names must be validated.

### Mistake 4: Forgetting Spaces in CONCAT

This can create invalid SQL.

Bad:

```sql
SET @stmt = CONCAT('SELECT * FROM', @tb, 'WHERE age > ?');
```

Good:

```sql
SET @stmt = CONCAT('SELECT * FROM ', @tb, ' WHERE age > ?');
```

### Mistake 5: Making Dynamic SQL Too Complicated

Prepared statements are useful, but too much dynamic SQL can become hard to debug. Use them only when they make the code clearer or more flexible.

## When Should You Use MySQL Prepared Statements?

Use prepared statements when:

- you need to run the same query with different values
- you want parameterized filtering
- you need dynamic SQL inside stored procedures
- you want to reduce repeated query writing
- you want safer value handling
- your application sends repeated queries to MySQL

Avoid prepared statements when:

- the query is simple and only runs once
- dynamic SQL makes the logic harder to understand
- table and column names come from untrusted input
- the same result can be achieved with simpler SQL

## Final Thoughts

In this post, we learned the basics of **MySQL prepared statements**. We created prepared statements with placeholders, executed them with different values, built dynamic SQL using `CONCAT()`, and wrapped dynamic logic inside stored procedures.

Prepared statements are useful for dynamic values and repeated SQL execution. They become even more powerful with stored procedures. But when using dynamic table names, column names, or operators, always validate inputs carefully.

This is just the beginning. Prepared statements and stored procedures can be used to build more flexible database logic, but they should be written carefully to keep SQL safe, readable, and maintainable.
