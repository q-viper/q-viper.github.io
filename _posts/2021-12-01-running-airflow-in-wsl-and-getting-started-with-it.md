---
title:  "Running Airflow in Windows with WSL"
date:   2021-12-01 10:29:17 +0545
update_date: 2021-12-02 12:29:17 +0545
categories:
    - airflow
    - data engineering
tags:
    - data pipelining
    - wsl
header:
  teaser: assets/airflow_blog/scheduler.PNG
---

# Getting Started With Airflow in WSL

## Introduction
Airflow is a data pipelining tool used for ETL operations. It is a hot requirement in the field of data related jobs.
Airflow schedules task on the concept of graph thus, there will be a collection of related task called as a DAG (Directed Acyclic Graph). 


## Installing WSL
Using airflow in Windows machine is hard way to go but with the use of Docker one can do it easily. But I am using [Ubuntu in WSL](https://www.microsoft.com/store/productId/9NBLGGH4MSV6) (Windows Subsystem for Linux) to use Airflow in my Windows.

## Installing Airflow
(Referenced from [here](https://towardsdatascience.com/run-apache-airflow-on-windows-10-without-docker-3c5754bb98b4).)
* Open the Ubuntu.
* Update system packages.
    ```
    sudo apt update
    sudo apt upgrade
    ```

* Installing PIP.
    ```
    sudo apt-get install software-properties-common
    sudo apt-add-repository universe
    sudo apt-get update
    sudo apt-get install python-setuptools
    sudo apt install python3-pip
    ```

* Run `sudo nano /etc/wsl.conf` then, insert the block below, save and exit with `ctrl+s` `ctrl+x`
```
[automount]
root = /
options = "metadata"
```

* To setup a airflow home, first make sure where to install it. Run `nano ~/.bashrc`, insert the line below, save and exit with `ctrl+s` `ctrl+x`

    ```export AIRFLOW_HOME=c/users/YOURNAME/airflowhome```

    Mine is, `/mnt/c/users/dell/myName/documents/airflow`

* Install virtualenv to create environment.
    ```
    sudo apt install python3-virtualenv
    ```

* Create and activate environment.
    ```
    virtualenv airflow_env
    source airflow_env/bin/activate
    ```

* Install airflow
    ```
    install apache-airflow
    ```

* Make sure if Airflow is installed properly.
    ```
    airflow info
    ```

    If no error pops up, proceed else install missing packages.

* Initialize DB. By default, sqlite is used.
    ```
    airflow db init
    ```

* Create airflow user.
    ```
    airflow users create [-h] -e EMAIL -f FIRSTNAME -l LASTNAME [-p PASSWORD] -r
                         ROLE [--use-random-password] -u USERNAME
    ```

* Start webserver.
    ```
    airflow webserver
    ```
    * Go to URL `http://localhost:8080/`. If error pops up, check what is missing. Below page will be seen.

        ![img]({{site.url}}/assets/airflow_blog/login.PNG)

    * Next page might be something like below.

        ![img]({{site.url}}/assets/airflow_blog/dags.PNG)

* In another terminal, enable virtual environment and then start scheduler.
    ```
    airflow scheduler
    ```

## Ways to Define a DAG
### Way 1 

```python
with DAG(..) as dag:
    DummyOperator()
```
### Way 2

```python
dag = DAG(..)
DummyOperator(dag=dag)
```

### Our Dag

```python
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.models import Variable


with DAG(dag_id="my_dag", description="DAG for showing nothinig.", 
        start_date=datetime(2021, 1, 1), schedule_interval=timedelta(minutes=5),
        dagrun_timeout=timedelta(minutes=10), tags=["learning_dag"], catchup=False) as dag:

```

First way is best way to do it because we do not have to give dag name to tasks everytime we define it.

### Most Useful Parameters While Defininig a DAG
* `dag_id`: should be unique else scheduler will randomly choose one dag with that id.
* `start_date`: should be used else error is shown but it is not default. It is a date at which task starts being scheduled. All DAG operators will use start_date of DAG
* `schedule_interval`: At which interval do we need to run. Possible values `@daily`, `@weekly`, cron job expression (ex. */10 * * * * for every 10min), timedelta,
Interval of time from min(start_date) at which DAG is triggered. It waits for start_date + schedule_interval and then triggers, execution date is start_date.
* `dagrun_timeout`: What to do when dag is running for longer time?  By default, it keeps running even if the previous dag is not complete. But we could stop DAG run after specified time.
* `tags`: To filter dags in UI. My DAG's tag will be `learning_dag` 
* `catchup`: Should we run all the schedules that we missed? If True, scheduler will automatically trigger all the previous task runs that are available between this date and start date.

#### Cron Expression vs timedelta in `schedule_interval`
* Cron Expn: Stateless. Trigger according to the expression. 0 0 * * * is for each day's 00:00:00 AM. 
* timedelta: Stateful. Trigger according to previous execution date.

#### Task's Rule
All task should be:
* **Idempotence**: If excuted multiple times, it should always have same side effect. If tried to create SQL table twice, error comes so to make idempothence, create if not exists".
* **Determinism**: If task run with same input, output should always be same. If 

#### Backfilling:
How to run on the previous date i.e past dates?
* Airflow will try to trigger all the non triggered tasks in the dates between current date and start date.
* Altered by `catchup` parameter. If set to `False`, no previous task runs are triggered. But we can forcefully backfill from airflow CLI using command `airflow dags backfill -s 2021-10-01 -e 202011-01`. 

#### Running Limited DAGs at a Time
Using parameter `max_active_runs`, we could only run number of tasks that we intend to. We could limit resources usage by doing this and also we could manage dependencies between tasks.

## Variables In Airflow
Object with key and a value stored in metadatabse. 
We can create via:
* Airflow UI: In Admin -> Variables.
* CLI
* APIs

Get variable via, `Variable.get()`. To make it secret, add `_secret` on the last of variable name.

### Properly Fetch Variable
* Never use `Variable.get()` outside a Task. Else we would be making a useless connection everytime our DAG is parsed. We will be making tons of useless variables.
* How to retreive multiple relative variables? Instead of making connection request for each of variables, use JSON as value in Variable and pass deserialize_json=True to access json as dictionary. 
* Passing variable only once. Instead of passing `Variable.get()` in `op_args`, we could pass "{{ var.json.variable_name.variable_key}}". Doing this, we wont be making fetch more than once.

#### Examples
* Create 3 variables from UI. `data_folder`, `test_df` and `user_info` then pass values accordingly. Make sure `user_info` is in JSON format i.e. '{"uname":"admin","password":"password"}'.
* Create a function outside DAG,

    ```python
    def _extract():
        file_path = Variable.get("data_folder") + "/" + Variable.get("test_df")
        uinfo = Variable.get("user_info", deserialize_json=True)
        print(uinfo, file_path)
        print(uinfo["uname"], uinfo["password"])
    ```
* Inside a DAG create a task,

    ```python
        extract = PythonOperator(
                task_id="extract", 
                python_callable=_extract
                )
    ```

To see this task in action, 
    * Re-run scheduler and see the DAG with name `my_dag` then enable it. 
    * Go inside the DAG and hit the trigger by clicking on play icon.
    * To see the output, go to the log by clicking on the green rectangle. And then logs.

    ![PNG]({{site.url}}/assets/airflow_blog/run_dag.PNG)

    * Logs output will be something like below

    ![PNG]({{site.url}}/assets/airflow_blog/log_op.PNG)

* Using `{{var.json.variable_name.variable_key}}`. Alternatively, we could do `{{var.value.variable_name}}. Outside DAG.
    
```python
    def _extract2(uname):
        print(f"Username: {uname}")
    ```

* Inside DAG.

    ```python
        extract2 = PythonOperator(
                task_id="extract2", 
                python_callable=_extract2,
                op_args = ["{{ var.json.user_info.uname}}"])
    ```

### Environment Variable
Why do we need environment variable? Well, first reason is that we will be hiding our variables from unwanted users and second reason is that we won't have to make database connection everytime we want to access this variable.

Any airflow environment variable will start with `AIRFLOW_VAR_` and will be in JSON format. ex
`AIRFLOW_VAR_VARNAME='{"uname":"admin","password":"password"}'`. To setup this variable, we have to create an environment variable first. To do so, `export AIRFLOW_VAR_USER_INFO2='{"uname":"admin","password":"password"}'`. The variable will not be permanent though so we need to insert it into `.bashrc` by `nano ~/.bashrc`. 

Insert line `export AIRFLOW_VAR_USER_INFO2='{"uname":"admin","password":"password"}'` in bashrc file.

#### Examples
* Outside DAG.

    ```python 
    def _extract_env():
        print(Variable.get("user_info2", deserialize_json=True))
    
    ```
* Inside DAG.

    ```python 
        extract_env = PythonOperator(
                    task_id="extract_env", 
                    python_callable=_extract_env
                    )
            
    ```
    
## Final Codes

    ```python
    
    from airflow import DAG
    from datetime import datetime, timedelta
    from airflow.operators.python import PythonOperator
    from airflow.models import Variable
    
    
    def _extract():
        """[summary]
        """
        file_path = Variable.get("data_folder") + "/" + Variable.get("test_df")
        uinfo = Variable.get("user_info", deserialize_json=True)
        # print(uinfo, file_path)
        print(uinfo["uname"], uinfo["password"])
    
    def _extract2(uname):
        """[summary]
        """
        print(f"Username: {uname}")
    
    def _extract_env():
        """[summary]
        """
        print(Variable.get("user_info2", deserialize_json=True))
    
    with DAG(dag_id="my_dag", description="DAG for showing nothinig.", 
             start_date=datetime(2021, 1, 1), schedule_interval=timedelta(minutes=5),
             dagrun_timeout=timedelta(minutes=10), tags=["learning_dag"], catchup=False) as dag:
        
        extract = PythonOperator(
                task_id="extract", 
                python_callable=_extract
                )
        
        extract2 = PythonOperator(
                task_id="extract2", 
                python_callable=_extract2,
                op_args = ["{{ var.json.user_info.uname}}"])
        
        extract_env = PythonOperator(
                task_id="extract_env", 
                python_callable=_extract_env
                )
        
    ```
    
