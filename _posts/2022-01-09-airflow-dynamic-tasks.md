---
title:  "Dynamic Tasks in Airflow"
date:   2022-01-09 10:29:17 +0545
last_modified_at: 2022-01-09 12:29:17 +0545
categories:
    - apache airflow
    - data engineering
    - data pipelining
tags:
    - data pipelining
    - ubuntu and windows
    - tasks
    - airflow for beginners
    - tutorial
header:
  teaser: assets/airflow_blog/graph_task.png
---

This blog is a continuation of previous blog **[Getting Started With Airflow in WSL]({{site.url}}/2021/12/01/running-airflow-in-wsl-and-getting-started-with-it/)**.

## Dynamic Tasks in Airflow
Sometimes there will be a need to create different task for different purpose within a DAG and those task has to be run dynamically. Not only run but has to be created dynamically also. A simple example could be, we want to connect to different database to pipeline data from different source and we have to connect to them manually. It will not be a much hassle if we are working on few databases but what if there are 100 different sources? Creating a distinct task for each from the scratch is not a right way to do it. And there is a simple solution to it i.e. **Dynamic Task**. One way we could achieve it is by creating a common function that will authenticate and gives us the session and we will pass different credentials to that function via loop. Lets see it in action. (All of the codes will be continued from the previous part.)

### DAG File: `first_dag.py`
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow.decorators import task, dag
from airflow.operators.subdag import SubDagOperator
from airflow.utils.task_group import TaskGroup
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
    return {"uname":name,"password":pwd}

@task.python
def authenticate(uname, pwd):
    print(uname, pwd)

@task.python
def validate(uname, pwd):
    print(uname, pwd)
    
@dag(description="DAG for showing nothing.", 
         default_args=default_args, schedule_interval="@daily", #timedelta(minutes=5)
         dagrun_timeout=timedelta(minutes=10), tags=["learning_dag"], catchup=False)
def my_dag():    
    for name,detail in db_details.items():
        validate_tasks(extract(detail["uname"], detail["password"]))
 
    
    
    
md = my_dag()
```

### Validate File: `validate_tasks.py`

```python

from airflow.utils.task_group import TaskGroup
from airflow.decorators import task,task_group

@task.python
def authenticate(uname, pwd):
    print(uname, pwd)

@task.python
def validate(uname, pwd):
    print(uname, pwd)
    
@task.python
def check_uname(uname):
    print(f"Entered Uname: {uname}")

@task.python
def check_password(pwd):
    print(f"Entered Password: {pwd}")

def validate_tasks(uinfo):
    # @task_group(group_id="validate_tasks")
    # def validate_tasks():
    with TaskGroup(group_id="validate_tasks", add_suffix_on_collision=True) as validate_tasks:
            
        uname = uinfo["uname"]
        pwd = uinfo["password"]
        
        with TaskGroup(group_id="checks") as checks:
            check_uname(uname)
            check_password(pwd)
        
        checks >> validate(uname, pwd)
        checks >> authenticate(uname, pwd)
        
    return validate_tasks        
```

Comparing with previous code, we have done few minor changes. Those includes:
1. Created a dictionary `db_details` and stored name, username and password of database.
2. In `my_dag` function, we ran a loop inside db_details and passed username and password to `extract`. Furthermore, we passed that `extract` task's object to `validate_tasks` task.
3. In `validate_tasks.py`, we added a parameter `add_suffix_on_collision=True` which allows us to use suffix on similar task running twice. Else we would be getting an error while making multiple task with same id inside same DAG.

Going over a UI, we could see something like below:

Tree View

![]({{site.url}}/assets/airflow_blog/graph_view_loop.png)

Graph View

![]({{site.url}}/assets/airflow_blog/dynamic_task_loop.png)

Tweaking it little bit to make it look more logical and friendly. Inside `my_dag.py`, 

```python
from airflow.operators.dummy import DummyOperator

@dag(description="DAG for showing nothing.", 
         default_args=default_args, schedule_interval="@daily", #timedelta(minutes=5)
         dagrun_timeout=timedelta(minutes=10), tags=["learning_dag"], catchup=False)
def my_dag():
    start = DummyOperator(task_id="start")
    for name,detail in db_details.items():
        @task.python(task_id=f"extract_uinfo_{name}", multiple_outputs=True, do_xcom_push=False)
        def extract(name, pwd):
            return {"uname":name,"password":pwd}

        extracted = extract(detail["uname"], detail["password"])
        start >> extracted
        validate_tasks(extracted)
```

What we basically did is, made a dynamic task according to the name of database from dictionary and made it run after a `start` task. `start` task is a dummy task that does nothing and it is here just to combine 3 different tasks. Now our graph looks like:

![]({{site.url}}/assets/airflow_blog/dynamic_with_dummy.png)

Which is much more readable and pleasing to see.

It does seem like not actually a dynamic tasks we made here because everything was defined inside a dictionary already. In earlier verisons, Airflow did not support dynamic tasks made from output of some other task. Which means that Airflow has to know something already. But lets check if it works in current version (i.e. ).



## References
* [Astronomer Certification Apache Airflow DAG Authoring Preparation](https://academy.astronomer.io/astronomer-certification-apache-airflow-dag-authoring-preparation)
* [Airflow XCOM](https://marclamberti.com/blog/airflow-xcom/)
    
