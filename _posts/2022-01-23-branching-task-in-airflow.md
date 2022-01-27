---
title:  "Branching Tasks in Airflow"
date:   2022-01-23 10:29:17 +0545
last_modified_at: 2022-01-23 12:29:17 +0545
categories:
    - apache airflow
    - data engineering
    - data pipelining
tags:
    - data pipelining
    - tasks
    - branching
    - airflow for beginners
    - tutorial
header:
  teaser: assets/airflow_blog/graph_task.png
---

## Branching  Task in Airflow
When do we need to make a branch like flow of a task? 


A simple example could be, lets assume that we are in a Media Company and
our task is to provide personalized content experience. Lets assume that we will have 3 different sets of rules for 3 different types of customers.
One for new comers, another for subscribed but not active and last for subscribed and active customer. And we also have to treat these 3 as distinct.
How would we manage to send a first ever content recommendation to each? A simple example could be, we make a distinct flow of tasks for distinct group of customers.
That will be much more efficient and logically easier to do. And Airflow allows us to do so. Lets see it how. 

This blog is a continuation of previous blogs
* [Getting Started With Airflow in WSL]({{site.url}}/2021/12/01/running-airflow-in-wsl-and-getting-started-with-it/)
* [Dynamic Tasks in Airflow]({{site.url}}/2022/01/09/airflow-dynamic-tasks/)

There are different of [Branching operators](https://airflow.apache.org/docs/apache-airflow/2.1.0/search.html?q=branch&check_keywords=yes&area=default) available in Airflow:
1. [Branch Python Operator](https://airflow.apache.org/docs/apache-airflow/2.1.0/_api/airflow/operators/python/index.html?highlight=branch#airflow.operators.python.BranchPythonOperator)
2. [Branch SQL Operator](https://airflow.apache.org/docs/apache-airflow/2.1.0/_api/airflow/operators/sql/index.html?highlight=branch#airflow.operators.sql.BranchSQLOperator)
3. [Branch Datetime Operator](https://airflow.apache.org/docs/apache-airflow/2.1.0/_api/airflow/operators/datetime/index.html?highlight=branch#airflow.operators.datetime.BranchDateTimeOperator)


### Airflow `BranchPythonOperator`
In this example, we will again take previous code and update it. Lets decide that, 
* If a customer is new, then we will use MySQL DB,
* If a customer is active, then we will use SQL DB,
* Else, we will use Sqlite DB.

> We have to return a task_id to run if a condition meets. A Branch always should return something (task_id).

```python
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.models import Variable
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow.decorators import task, dag
from airflow.operators.subdag import SubDagOperator
from airflow.utils.task_group import TaskGroup
from airflow.operators.dummy import DummyOperator

from datetime import datetime, timedelta
from typing import Dict
# from learning_project_DAG.subdag.subdag_factory import subdag_factory
from learning_project_DAG.groups.validate_tasks import validate_tasks


default_args = {
    "start_date": datetime(2021, 1, 1)
}

db_details = {
    "MySQL": 
        {
            "uname": "MySQL",
            "password": "adminadmin"
        },
        "SQL": 
        {
            "uname": "SQL",
            "password": "pass"
        },
        "Sqlite": 
        {
            "uname": "Sqlite",
            "password": "admin"
        }
}

@task.python(task_id="extract_uinfo", multiple_outputs=True, do_xcom_push=False)
def extract(name, pwd):
    return {"uname":name, "password": pwd}

@task.python
def authenticate(uname, pwd):
    print(uname, pwd)

@task.python
def validate(uname, pwd):
    print(uname, pwd)

def _choose_db_based_on_utype(utype):
    if utype == "new":
        return 'extract_unifo_MySQL'
    elif utype == "active":
        return 'extract_unifo_SQL'
    else:
        return 'extract_unifo_Sqlite'

@dag(description="DAG for showing nothing.", 
         default_args=default_args, schedule_interval="@daily", #timedelta(minutes=5)
         dagrun_timeout=timedelta(minutes=10), tags=["learning_dag"], catchup=False)
def my_dag():
    start = DummyOperator(task_id="start")
    stop = DummyOperator(task_id="stop")
    log_info = DummyOperator(task_id="log_info", trigger_rule="none_failed_or_skipped")
    
    choose_db = BranchPythonOperator(
        task_id = "choose_db_based_on_utype",
        python_callable=_choose_db_based_on_utype
    )
    
    choose_db >> stop
    for name,detail in db_details.items():
        @task.python(task_id=f"extract_uinfo_{name}")
        def extract(name, pwd):
            return {"uname":name,"password":pwd}

        extracted = extract(detail["uname"], detail["password"])
        start >> choose_db >> extracted 
        validate_tasks(extracted) >> log_info  
    
    
md = my_dag()
```

From the last time, we have done few changes:
* Imported `BranchPythonOperator`.
* Made a new task, `choose_db = BranchPythonOperator(task_id = "choose_db_based_on_utype", python_callable=_choose_db_based_on_utype)` .
* Then made a new python callable function. Where certain task_id is returned based on utype.

```python
def _choose_db_based_on_utype(utype):
    if utype == "new":
        return 'extract_unifo_MySQL'
    elif utype == "active":
        return 'extract_unifo_SQL'
    else:
        return 'extract_unifo_Sqlite'
```
* Made a new task to run `stop`. In case choose_db fails.
* Made a new flow of tasks.
```python
choose_db >> stop
```
* At the end, we want to log the info so we have a `log_info` task.

Now, the Graph should look like below:

![]({{site.url}}/assets/airflow_blog/python_branching_op.png)

But as we can see above (green boxes) it is not working as it should be. Our last should be running regardless of whether its parents runs or not.

If we run our DAG. we will be seeing error in choose_db_based_on_type.

Now to make it little bit more logical, we will pass a arguments to it.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.models import Variable
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow.decorators import task, dag
from airflow.operators.subdag import SubDagOperator
from airflow.utils.task_group import TaskGroup
from airflow.operators.dummy import DummyOperator

from datetime import datetime, timedelta
from typing import Dict
# from learning_project_DAG.subdag.subdag_factory import subdag_factory
from learning_project_DAG.groups.validate_tasks import validate_tasks


default_args = {
    "start_date": datetime(2021, 1, 1)
}

db_details = {
    "MySQL": 
        {
            "uname": "MySQL",
            "password": "adminadmin"
        },
        "SQL": 
        {
            "uname": "SQL",
            "password": "pass"
        },
        "Sqlite": 
        {
            "uname": "Sqlite",
            "password": "admin"
        }
}

@task.python(task_id="extract_uinfo", multiple_outputs=True, do_xcom_push=False)
def extract(name, pwd):
    
    return {"uname":name, "password": pwd}

@task.python
def authenticate(uname, pwd):
    print(uname, pwd)

@task.python
def validate(uname, pwd):
    print(uname, pwd)

def _choose_db_based_on_utype(utype):
    if utype == "new":
        return 'extract_uinfo_MySQL'
    elif utype == "active":
        return 'extract_uinfo_SQL'
    else:
        return 'extract_uinfo_Sqlite'

@dag(description="DAG for showing nothing.", 
         default_args=default_args, schedule_interval="@daily", #timedelta(minutes=5)
         dagrun_timeout=timedelta(minutes=10), tags=["learning_dag"], catchup=False)
def my_dag():
    
    start = DummyOperator(task_id="start")
    stop = DummyOperator(task_id="stop")
    log_info = DummyOperator(task_id="log_info", trigger_rule="none_failed_or_skipped")
    
    choose_db = BranchPythonOperator(
        task_id = "choose_db_based_on_utype",
        python_callable=_choose_db_based_on_utype,
        op_args=["new"]
    )
    
    choose_db >> stop
    for name,detail in db_details.items():
        @task.python(task_id=f"extract_uinfo_{name}", multiple_outputs=True, do_xcom_push=False)
        def extract(name, pwd):
            return {"uname":name,"password":pwd}

        extracted = extract(detail["uname"], detail["password"])
        start >> choose_db >> extracted 
        validate_tasks(extracted) >> log_info 
md = my_dag()
```

Now, if we run our DAG, Graph should look like below:

![]({{site.url}}/assets/airflow_blog/python_branching_op_new.png)



### Airflow `BranchSQLOperator`
Before diving into making a SQL Branch Operator, I am going to make a dummy DB in MySQL and dummy table. 

```sql
create table airflow_test.user_info(
id int auto_increment,
`name` varchar(255),
`password` varchar(255),
primary key (id)
);

insert into airflow_test.user_info(`name`, `password`) values("admin", "admin123");
```

To use BranchSQLOperator, we have to install connector. Follow this [link](https://airflow.apache.org/docs/apache-airflow-providers-mysql/stable/index.html) to do so.

```shell
pip install apache-airflow-providers-mysql
```

While trying to make a MySQL Connection from WSL to windows, I was facing too many errors and I might not be able to share errors here,
but I have written a solid way about **[How to use MySQL Server (from WSL) that is in Windows?]()**. Please follow the link for more info. 

```python
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.models import Variable
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow.decorators import task, dag
from airflow.operators.subdag import SubDagOperator
from airflow.utils.task_group import TaskGroup
from airflow.operators.dummy import DummyOperator
from airflow.operators.sql import BranchSQLOperator

from datetime import datetime, timedelta
from typing import Dict
from learning_project_DAG.groups.validate_tasks import validate_tasks


default_args = {
    "start_date": datetime(2021, 1, 1)
}

db_details = {
    "MySQL": 
        {
            "uname": "MySQL",
            "password": "adminadmin"
        },
        "SQL": 
        {
            "uname": "SQL",
            "password": "pass"
        },
        "Sqlite": 
        {
            "uname": "Sqlite",
            "password": "admin"
        }
}

@task.python(task_id="extract_uinfo", multiple_outputs=True, do_xcom_push=False)
def extract(name, pwd):    
    return {"uname":name, "password": pwd}

@task.python
def authenticate(uname, pwd):
    print(uname, pwd)

@task.python
def validate(uname, pwd):
    print(uname, pwd)

def _choose_db_based_on_utype(utype):
    if utype == "new":
        return 'extract_uinfo_MySQL'
    elif utype == "active":
        return 'extract_uinfo_SQL'
    else:
        return 'extract_uinfo_Sqlite'

@dag(description="DAG for showing nothing.", 
         default_args=default_args, schedule_interval="@daily", #timedelta(minutes=5)
         dagrun_timeout=timedelta(minutes=10), tags=["learning_dag"], catchup=False)
def my_dag():
    start = DummyOperator(task_id="start")
    stop = DummyOperator(task_id="stop")
    log_info = DummyOperator(task_id="log_info", trigger_rule="none_failed_or_skipped")
    
    authenticate_success = DummyOperator(task_id="authenticate_success")
    authenticate_failure = DummyOperator(task_id="authenticate_failure")
    
    check_uname = BranchSQLOperator(
            task_id="check_uname",
            conn_id="airflow_db",
            sql="SELECT count(1) FROM `airflow_test`.`user_info` where `name`='admin';",
            follow_task_ids_if_true="authenticate_success",
            follow_task_ids_if_false="authenticate_failure",
            trigger_rule="none_failed_or_skipped"
            )
    
    
    choose_db = BranchPythonOperator(
        task_id = "choose_db_based_on_utype",
        python_callable=_choose_db_based_on_utype,
        op_args=["new"]
    )
    
    choose_db >> stop
    for name,detail in db_details.items():
        @task.python(task_id=f"extract_uinfo_{name}", multiple_outputs=True, do_xcom_push=False)
        def extract(name, pwd):
            return {"uname":name,"password":pwd}

        extracted = extract(detail["uname"], detail["password"])
        start >> choose_db >> extracted 
        validate_tasks(extracted) >> check_uname >>[authenticate_success, authenticate_failure]>> log_info 
md = my_dag()
```

Above code is slightly changed version of `BranchPythonOperator` and main changes are on:
* Make a mysql connection using a UI. Admin > Connections > Add New

![]({{site.url}}/assets/airflow_blog/create_connection.png)

* Make sure to use the same configuration that we setup earlier. Use host as the IPv4 from `Go to Settings -> Network and Internet -> Status -> View Hardware and connection properties`.
* `wsl_root` the username that we created for WSL. ([Please follow this blog for how we did it?]())
* And use default port of MySQL 3306.
* Also put password. 
* We imported a new Operator `BranchSQLOperator` as `from airflow.operators.sql import BranchSQLOperator`
* Created a BranchSQLOperator:

```python
    authenticate_success = DummyOperator(task_id="authenticate_success")
    authenticate_failure = DummyOperator(task_id="authenticate_failure")
    
    check_uname = BranchSQLOperator(
            task_id="check_uname",
            conn_id="airflow_db",
            sql="SELECT count(1) FROM `airflow_test`.`user_info` where `name`='admin';",
            follow_task_ids_if_true="authenticate_success",
            follow_task_ids_if_false="authenticate_failure",
            trigger_rule="none_failed_or_skipped"
            )
```
* Made 2 dummy tasks which are self explained.
* In above step, we would choose the connection id as the connection id that we set while making a connection configuration in above step.
    * `sql` is either sql file or query. In our case a SQL query and it is simple as get 1 if there is a field name which value is admin. Which was already set on earlier part of this section.
    * Two similar parameters,  `follow_task_ids_if_true` and `follow_task_ids_if_false` are self explained.
    * `trigger_rule` is to run this task regardless of whatever this task's parent happens.
* Then for making a flow of task, `validate_tasks(extracted) >> check_uname >>[authenticate_success, authenticate_failure]>> log_info ` is done. Which means that either one of task has to be executed among tasks inside `[]`.
* Now our graph will look like:

![]({{site.url}}/assets/airflow_blog/sql_branching_operator.png)

I have already triggered the DAG and as you can see there, the task seems to be following a trail. i.e. it is running as expected.


### Airflow `Branch Datetime Operator`
TBD

## References
* [Astronomer Certification Apache Airflow DAG Authoring Preparation](https://academy.astronomer.io/astronomer-certification-apache-airflow-dag-authoring-preparation)
* [Airflow XCOM](https://marclamberti.com/blog/airflow-xcom/)
    
