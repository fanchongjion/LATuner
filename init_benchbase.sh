#!/usr/bin/env bash
# create twitter database
MYSQL_BIN=$MYSQL_HOME/bin/mysql
RDS_HOST='localhost'
MYSQL_PORT=3306
SOCK="$MYSQL_HOME/mysql.sock"
DBNAME=benchbase
BENCHNAME=tpcc

$MYSQL_BIN -uroot -S$SOCK -h $RDS_HOST -P$MYSQL_PORT -e "drop database if exists $DBNAME"
$MYSQL_BIN -uroot -S$SOCK -h $RDS_HOST -P$MYSQL_PORT -e "create database $DBNAME"
cd /home/root3/Tuning/benchbase-mysql
java -jar benchbase.jar -b $BENCHNAME -c config/mysql/sample_${BENCHNAME}_config.xml --create=true --load=true