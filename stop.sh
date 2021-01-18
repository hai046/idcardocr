#!/usr/bin/env sh

ps -axf|grep "idcard_recognize.py"|grep -v grep|awk 'NR==1{print $1}'|xargs kill -TERM

