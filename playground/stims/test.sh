#!/bin/bash

if [ `ls -1 /var/log/apache2/access.log.* 2>/dev/null | wc -l ` -gt 0 ];
then
    echo "ok"
else
    echo "ko"
fi
